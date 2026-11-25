"""Iris: can the answer-logit nullspace predict when to re-read a frozen LM?

Run: python iris.py --self-test
     python iris.py --device cuda
No training of the LM. See IRIS.md for the locked protocol and limits.
"""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import common as C

# More families than Theory's original 8, fixed before any model run.
EXTRA = {
    "fruit": ["apple", "pear", "peach", "plum", "orange", "grape", "lemon", "melon"],
    "metal": ["iron", "gold", "silver", "copper", "zinc", "steel", "tin", "lead"],
    "color": ["red", "blue", "green", "yellow", "purple", "black", "white", "pink"],
    "sport": ["tennis", "golf", "soccer", "rugby", "boxing", "hockey", "skiing", "swimming"],
    "furniture": ["chair", "table", "bed", "desk", "sofa", "stool", "bench", "cabinet"],
    "clothing": ["shirt", "coat", "dress", "skirt", "pants", "socks", "hat", "jacket"],
    "emotion": ["joy", "anger", "fear", "sadness", "shame", "pride", "envy", "grief"],
    "profession": ["doctor", "nurse", "teacher", "lawyer", "pilot", "chef", "farmer", "dentist"],
    "river": ["Nile", "Amazon", "Thames", "Rhine", "Danube", "Mississippi", "Volga", "Congo"],
    "country": ["France", "Germany", "Spain", "Italy", "Japan", "China", "Canada", "Brazil"],
    "language": ["English", "French", "German", "Spanish", "Italian", "Japanese", "Arabic", "Russian"],
    "currency": ["dollar", "euro", "yen", "pound", "peso", "ruble", "franc", "rupee"],
}
BANK = {**{k: v[1] for k, v in C.FAMILIES.items()}, **EXTRA}
# Exclude words appearing as items in the few-shot examples. Ambiguous words
# are retained and disclosed; this is a pilot, not an ontology benchmark.
EXCLUDE = {"rose", "lion", "chair", "wolf", "cup"}


def unit(x):
    return F.normalize(x.float(), dim=-1, eps=1e-12)


def row_basis(W):
    """Orthonormal basis of the candidate output weight row span."""
    _, s, v = torch.linalg.svd(W.double().cpu(), full_matrices=False)
    return v[s > s.max() * 1e-10].float()


def split(h, basis):
    read = (h @ basis.T) @ basis
    return read, h - read


def neighbors(X, i, k=3):
    """No labels, no native answer, no self. Fixed cosine metric."""
    sim = unit(X) @ unit(X[i])
    sim[i] = -float("inf")
    return torch.argsort(sim, descending=True, stable=True)[:k].tolist()


@torch.no_grad()
def measure(r, word, qs, targets=None):
    """Capture exactly the input of the LM's output projection and its logits.
    All prompts are right padded by common.Reader.margins; use true last index.
    """
    ends = [len(r.prompt(word, q)[0]) - 1 for _, q in qs]
    box = {}
    def hook(m, inp, out):
        h = inp[0]
        idx = torch.arange(len(ends), device=h.device)
        j = torch.tensor(ends, device=h.device)
        box["h"] = h[idx, j].detach().float().cpu()
        z = out[idx, j].detach().float()
        ids = sorted(set(r.yes + r.no))
        lp = z.log_softmax(-1)
        box["z"] = z[:, ids].cpu()
        box["aux"] = torch.stack([
            -(lp.exp() * lp).sum(-1),
            lp.max(-1).values,
            torch.logsumexp(lp[:, ids], -1),
        ], -1).cpu()
    handle = r.model.get_output_embeddings().register_forward_hook(hook)
    try:
        m = r.margins(word, qs, targets)
    finally:
        handle.remove()
    if "h" not in box:
        raise RuntimeError("Output projection hook did not fire")
    box["m"] = torch.tensor([m[k] for k, _ in qs])
    return box


def feature_row(a, neutral, basis):
    """No target label. Sort questions by native calibrated evidence.
    Residue is neutral-subtracted, then split at the final output projection.
    """
    m = a["m"] - neutral["m"]
    order = torch.argsort(m, descending=True, stable=True)
    # Output-only baseline: all candidate margins, selected-token raw logits,
    # full-vocabulary entropy, top log probability and yes/no log mass.
    conf = torch.cat([m[order], a["m"][order], a["z"][order].flatten(),
                      a["aux"][order].flatten()])
    h = (a["h"] - neutral["h"])[order[:2]]
    read, res = split(h, basis)
    return conf, read.flatten(), res.flatten(), int(order[0])


def standardized_kernel(X, train):
    """Training-fold-only standardization for output features."""
    mu = X[train].mean(0)
    sd = X[train].std(0, unbiased=False).clamp_min(1e-3)
    V = (X - mu) / sd
    return unit(V) @ unit(V).T


def vector_kernel(X):
    V = unit(X)
    return V @ V.T


def predict(K, y, train, test):
    # Fixed ridge penalty and training-only mean. No test targets.
    mean = y[train].mean()
    A = K[train][:, train] + 1.0 * torch.eye(len(train))
    return mean + K[test][:, train] @ torch.linalg.solve(A, y[train] - mean)


def out_of_fold(conf, read, res, full, y, groups, seed=17):
    """Leave ONE ENTIRE true family out; labels used for train/eval only.
    This first gate is supervised, not a label-free controller.
    """
    N = len(y)
    out = {k: torch.zeros(N) for k in
           ("output", "output_read", "output_residue", "output_full",
            "output_shuffled_residue")}
    kr, ks, kf = map(vector_kernel, (read, res, full))
    # Shuffle residue association independently of targets; deterministic control.
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(seed))
    kshuffle = ks[perm][:, perm]
    for g in sorted(set(groups)):
        train = torch.tensor([i for i, v in enumerate(groups) if v != g])
        test = torch.tensor([i for i, v in enumerate(groups) if v == g])
        kc = standardized_kernel(conf, train)
        for name, K in (
            ("output", kc),
            ("output_read", (kc + kr) / 2),
            ("output_residue", (kc + ks) / 2),
            ("output_full", (kc + kf) / 2),
            ("output_shuffled_residue", (kc + kshuffle) / 2),
        ):
            out[name][test] = predict(K, y, train, test)
    return out


def evaluate(scores, native, edited, groups, fraction=0.5, seed=19):
    n = len(native)
    gain = edited - native
    k = max(1, int(n * fraction))
    selected = {}
    for name, s in scores.items():
        order = torch.argsort(s, descending=True, stable=True)
        mask = torch.zeros(n)
        mask[order[:k]] = 1
        selected[name] = mask
    def stat(mask):
        return float((native + mask * gain).mean())
    values = {name: stat(mask) for name, mask in selected.items()}
    values.update(never=float(native.mean()), always=float(edited.mean()),
                  random_expected=float(native.mean() + (k/n) * gain.mean()),
                  oracle=stat((torch.arange(n)[:,None] == torch.argsort(
                      gain, descending=True, stable=True)[:k][None,:]).any(1).float()))
    # Paired family bootstrap; masks/rankings fixed, no refit. Descriptive CI.
    rng = torch.Generator().manual_seed(seed)
    families = sorted(set(groups))
    ids = [torch.tensor([i for i, g in enumerate(groups) if g == f]) for f in families]
    delta = (selected["output_residue"] - selected["output"]) * gain
    draws = []
    for _ in range(3000):
        ix = torch.cat([ids[j] for j in torch.randint(len(ids), (len(ids),), generator=rng)])
        draws.append(float(delta[ix].mean()))
    ci = torch.quantile(torch.tensor(draws), torch.tensor([.025, .975])).tolist()
    return dict(n=n, budget=k, accuracy=values, residue_minus_output_ci=ci,
                rescued=int((gain==1).sum()), harmed=int((gain==-1).sum()),
                choices={a: b.bool().tolist() for a,b in selected.items()})


def run(r):
    fam = {k: [w for w in ws if w not in EXCLUDE and r.single(w)]
           for k, ws in BANK.items()}
    fam = {k: v for k, v in fam.items() if len(v) >= 3}
    groups = [k for k, v in fam.items() for _ in v]
    words = [w for v in fam.values() for w in v]
    if len(set(words)) != len(words) or len(fam) < 4:
        raise RuntimeError("Need unique words and at least four families")
    qs = [(k, C.family_q(k)) for k in fam]
    keys = list(fam)
    S = r.states(words)
    l = r.n_layers // 2
    X = torch.stack([S[w][l+1] for w in words])
    if r.centered:
        X = X - X.mean(1, keepdim=True)
    X = X - X.mean(0)
    neutral = measure(r, C.SLOT, qs)
    head = r.model.get_output_embeddings()
    ids = sorted(set(r.yes + r.no))
    W = head.weight.detach()[ids].float().cpu()
    basis = row_basis(W)
    conf, read, res, full, native, edited, records = [], [], [], [], [], [], []
    max_leak, max_reconstruction = 0., 0.
    band = r.band()
    for i, w in enumerate(words):
        a = measure(r, w, qs)
        c, p, q, j = feature_row(a, neutral, basis)
        # Verify logits and nullspace for ACTUAL captured hidden states.
        bias = getattr(head, "bias", None)
        recon = a["h"] @ W.T
        if bias is not None:
            recon += bias.detach()[ids].float().cpu()
        err = float((recon - a["z"]).abs().max())
        max_reconstruction = max(max_reconstruction, err)
        if not torch.allclose(recon, a["z"], atol=2e-3, rtol=2e-4):
            raise RuntimeError(f"Output projection reconstruction failed: {err}")
        leak = float((q.reshape(2,-1) @ W.T).abs().max())
        max_leak = max(max_leak, leak)
        if leak > 2e-3:
            raise RuntimeError(f"Candidate-logit nullspace check failed: {leak}")
        near = neighbors(X, i)
        targets = C.blend(S, [(0.5, S[w])] +
                          [(0.5/len(near), S[words[z]]) for z in near], band)
        m = r.margins(w, qs, targets)
        corrected = torch.tensor([m[k] for k in keys]) - neutral["m"]
        cj = int(corrected.argmax())
        conf.append(c); read.append(p); res.append(q); full.append(p+q)
        native.append(float(keys[j] == groups[i]))
        edited.append(float(keys[cj] == groups[i]))
        records.append(dict(word=w, family=groups[i], neighbors=[words[z] for z in near],
                            native=keys[j], edited=keys[cj],
                            margin=float(torch.topk(a["m"]-neutral["m"],2).values.diff().abs()[0])))
        if (i+1) % 10 == 0:
            print(f"{i+1}/{len(words)}", flush=True)
    native, edited = torch.tensor(native), torch.tensor(edited)
    scores = out_of_fold(*map(torch.stack, (conf, read, res, full)),
                         edited-native, groups)
    result = evaluate(scores, native, edited, groups)
    # Fixed direct uncertainty rule, matched global re-read budget.
    order = sorted(range(len(words)), key=lambda i: (records[i]["margin"],i))
    mask = torch.zeros(len(words)); mask[order[:result["budget"]]] = 1
    result["accuracy"]["lowest_margin"] = float((native+mask*(edited-native)).mean())
    result["choices"]["lowest_margin"] = mask.bool().tolist()
    a = result["accuracy"]
    result["gate"] = bool(result["rescued"] >= 5 and result["harmed"] >= 5
        and a["output_residue"] >= max(a["output"], a["lowest_margin"],
            a["random_expected"], a["always"], a["never"]) + .05
        and result["residue_minus_output_ci"][0] > 0)
    result.update(model=r.name, family_counts={k:len(v) for k,v in fam.items()},
                  band=list(band), neighbor_layer=l, row_rank=len(basis),
                  max_logit_reconstruction_error=max_reconstruction,
                  max_residue_logit_leak=max_leak, records=records,
                  scores={k:v.tolist() for k,v in scores.items()})
    return result


def self_test():
    torch.manual_seed(9)
    W = torch.randn(4,32)
    B = row_basis(W)
    h = torch.randn(8,32)
    p,q = split(h,B)
    assert torch.allclose(p+q,h,atol=1e-6)
    assert (q@W.T).abs().max() < 1e-5
    X = torch.randn(12,32)
    assert 0 not in neighbors(X,0)
    # A synthetic hidden variable determines whether an edit helps or harms.
    groups = [i//12 for i in range(96)]
    y = torch.tensor([1.,-1.]*48)
    native = (y<0).float(); edited = (y>0).float()
    conf = torch.ones(96,4)
    read = torch.randn(96,4)
    res = y[:,None] * torch.ones(96,16)
    scores = out_of_fold(conf,read,res,res,y,groups)
    result = evaluate(scores,native,edited,groups)
    assert result["accuracy"]["output_residue"] == 1.0
    assert result["accuracy"]["oracle"] == 1.0
    assert result["accuracy"]["random_expected"] == .5
    # Changing held-out labels cannot change predictions in its own fold.
    y2 = y.clone(); y2[:12] *= -1
    scores2 = out_of_fold(conf,read,res,res,y2,groups)
    for key in scores:
        assert torch.equal(scores[key][:12],scores2[key][:12])
    # Actual model hook and output reconstruction smoke test on random stub.
    r = C.load_stub("neox")
    qs = [("tree",C.family_q("tree")),("fish",C.family_q("fish"))]
    a = measure(r,"oak",qs); b = measure(r,C.SLOT,qs)
    ids=sorted(set(r.yes+r.no))
    W=r.model.get_output_embeddings().weight.detach()[ids].float()
    assert torch.allclose(a["h"]@W.T,a["z"],atol=1e-4)
    _,_,q,_=feature_row(a,b,row_basis(W))
    assert (q.reshape(2,-1)@W.T).abs().max() < 1e-4
    assert not r.model.get_output_embeddings()._forward_hooks
    print("Iris self-test passed. Synthetic and random-model tests only.")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",default="EleutherAI/pythia-410m")
    p.add_argument("--revision",default=None)
    p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--frame",choices=C.FRAMES,default="shots")
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    if a.self_test:
        return self_test()
    revision=a.revision or ("step143000" if "pythia" in a.model.lower() else None)
    r=C.load(a.model,a.device,torch.float32,revision=revision,frame=a.frame)
    result=run(r)
    out=Path("runs")/("iris_"+a.model.replace("/","_")+"_"+a.frame+
                      ("_"+revision if revision else ""))
    out.mkdir(parents=True,exist_ok=True)
    path=out/"iris.json"
    path.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ("records","choices","scores")},indent=2))
    print("saved",path)


if __name__=="__main__":
    main()
