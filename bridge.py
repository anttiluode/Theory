"""Bridge: can Maatuska's label-free PCA knob repair the frozen model's own answers?

Run: python bridge.py --self-test
     python bridge.py --device cuda
Protocol and frozen gates: BRIDGE.md. No training of the LM. Reuses Iris's words,
frame, neutral calibration and clamp, so the 3-NN row should reproduce Iris.
"""
import argparse
import json
import random
from pathlib import Path

import torch

import common as C
import iris as I

N_VOCAB = 3000
PRIMARY = "pca16_l0.5"
LAMBDA = 0.5


# ---------------------------------------------------------------- word sets
def bank_words(r):
    fam = {k: [w for w in ws if w not in I.EXCLUDE and r.single(w)] for k, ws in I.BANK.items()}
    fam = {k: v for k, v in fam.items() if len(v) >= 3}
    words = [w for v in fam.values() for w in v]
    groups = [k for k, v in fam.items() for _ in v]
    assert len(set(words)) == len(words), "bank words must be unique"
    return fam, words, groups


def vocab_words(r, exclude, n):
    """Unlabeled single-token words from the tokenizer, disjoint from the bank and the shots."""
    shot_words = {w.lower() for w in C.SHOTS.replace(".", " ").replace("?", " ").split()}
    banned = {w.lower() for w in exclude} | shot_words | {C.SLOT}
    if hasattr(r.tok, "get_vocab"):
        items = sorted(r.tok.get_vocab().items(), key=lambda kv: kv[1])
        cands = [t[1:] for t, _ in items if t[:1] in ("Ġ", "▁")]
    else:  # test stub: plain word list
        cands = [t.strip() for t in r.tok.vocab if t.startswith(" ")]
    out, seen = [], set()
    for w in cands:
        if len(w) >= 4 and w.isalpha() and w.islower() and w not in banned and w not in seen:
            seen.add(w)
            if r.single(w):
                out.append(w)
        if len(out) >= n:
            break
    return out


# ---------------------------------------------------------------- label-free bases
def pca_bases(S, vocab, band):
    """Per band layer: vocabulary mean and PCA directions (columns, by variance)."""
    out = {}
    for l in range(band[0], band[1] + 1):
        X = torch.stack([S[w][l + 1] for w in vocab]).double()
        mu = X.mean(0)
        _, _, vh = torch.linalg.svd(X - mu, full_matrices=False)
        out[l] = (mu.float(), vh.T.float().contiguous())
    return out


def random_bases(bases, seed=23):
    g = torch.Generator().manual_seed(seed)
    out = {}
    for l, (mu, V) in bases.items():
        D = V.shape[0]
        q, rr = torch.linalg.qr(torch.randn(D, D, generator=g, dtype=torch.float64))
        out[l] = (mu, (q * torch.sign(torch.diagonal(rr))).float())
    return out


def subspace_edit(S, w, bases, band, k0, d, lam):
    """s + lam * (mu + P(s - mu) - s), P onto components k0 .. k0+d-1."""
    t = {}
    for l in range(band[0], band[1] + 1):
        mu, V = bases[l]
        s = S[w][l + 1]
        B = V[:, k0:k0 + d]
        proj = mu + (s - mu) @ B @ B.T
        t[l] = s + lam * (proj - s)
    return t


def lid_edit(S, w, bases, band, k, lam):
    """s + lam * (-(top-k component of s - mu))."""
    t = {}
    for l in range(band[0], band[1] + 1):
        mu, V = bases[l]
        s = S[w][l + 1]
        B = V[:, :k]
        t[l] = s - lam * ((s - mu) @ B @ B.T)
    return t


def matched_move(S, w, target, seed):
    """A random move with the same length as target - s at every layer."""
    delta = {l: v - S[w][l + 1] for l, v in target.items()}
    rnd = C.matched_random(delta, seed)
    return {l: S[w][l + 1] + rnd[l] for l in target}


# ---------------------------------------------------------------- answers
def answer(r, w, qs, keys, neutral, targets=None):
    m = r.margins(w, qs, targets)
    z = torch.tensor([m[k] for k, _ in qs]) - torch.tensor([neutral[k] for k, _ in qs])
    return keys[int(z.argmax())]


def edits_for(name, S, w, i, words, X, bases, rbases, band, seed):
    if name == "native":
        return None
    if name == "pca16_l0.5":
        return subspace_edit(S, w, bases, band, 0, 16, 0.5)
    if name == "pca16_l1.0":
        return subspace_edit(S, w, bases, band, 0, 16, 1.0)
    if name == "pca64_l0.5":
        return subspace_edit(S, w, bases, band, 0, 64, 0.5)
    if name == "lidpre16_l0.5":
        return subspace_edit(S, w, bases, band, 1, 16, 0.5)
    if name == "lid1_l1.0":
        return lid_edit(S, w, bases, band, 1, 1.0)
    if name == "lid3_l1.0":
        return lid_edit(S, w, bases, band, 3, 1.0)
    if name == "rand16_l0.5":
        return subspace_edit(S, w, rbases, band, 0, 16, 0.5)
    if name == "pca16_l0.5_matched":
        return matched_move(S, w, subspace_edit(S, w, bases, band, 0, 16, 0.5), seed + i)
    if name == "nn3_l0.5":
        near = I.neighbors(X, i)
        return C.blend(S, [(0.5, S[w])] + [(0.5 / 3, S[words[z]]) for z in near], band)
    if name == "nn3_random_l0.5":
        rng = random.Random(seed + 1000 + i)
        near = rng.sample([j for j in range(len(words)) if j != i], 3)
        return C.blend(S, [(0.5, S[w])] + [(0.5 / 3, S[words[z]]) for z in near], band)
    raise KeyError(name)


EDITS = ["native", "pca16_l0.5", "pca16_l1.0", "pca64_l0.5", "lidpre16_l0.5", "lid1_l1.0",
         "lid3_l1.0", "rand16_l0.5", "pca16_l0.5_matched", "nn3_l0.5", "nn3_random_l0.5"]


def paired_ci(a, b, n=4000, seed=0):
    g = torch.Generator().manual_seed(seed)
    d = torch.tensor(a, dtype=torch.float64) - torch.tensor(b, dtype=torch.float64)
    draws = torch.stack([d[torch.randint(len(d), (len(d),), generator=g)].mean() for _ in range(n)])
    return [float(d.mean()), float(torch.quantile(draws, 0.025)), float(torch.quantile(draws, 0.975))]


def run(r, n_vocab=N_VOCAB, seed=31, log=print):
    fam, words, groups = bank_words(r)
    vocab = vocab_words(r, words, n_vocab)
    band = r.band()
    log(f"{len(words)} bank words in {len(fam)} families; {len(vocab)} unlabeled vocabulary "
        f"words; clamp band {band}")
    S = r.states(words + vocab)
    bases = pca_bases(S, vocab, band)
    rbases = random_bases(bases)
    # Iris's neighbour space, exactly as in iris.run
    l = r.n_layers // 2
    X = torch.stack([S[w][l + 1] for w in words])
    if r.centered:
        X = X - X.mean(1, keepdim=True)
    X = X - X.mean(0)

    fam_keys = list(fam)
    fam_qs = [(k, C.family_q(k)) for k in fam_keys]
    leaf_qs_all = [(w, C.leaf_q(w)) for w in words]
    neutral = {}
    allq = fam_qs + leaf_qs_all
    for s0 in range(0, len(allq), 24):  # chunks keep full-vocabulary logits small on a 12 GB GPU
        neutral.update(r.margins(C.SLOT, allq[s0:s0 + 24]))

    acc = {e: {"category": [], "identity": []} for e in EDITS}
    records = []
    for i, (w, g) in enumerate(zip(words, groups)):
        leaf_qs = [(x, C.leaf_q(x)) for x in fam[g]]
        leaf_keys = [x for x, _ in leaf_qs]
        rec = {"word": w, "family": g}
        for e in EDITS:
            t = edits_for(e, S, w, i, words, X, bases, rbases, band, seed)
            cat = answer(r, w, fam_qs, fam_keys, neutral, t)
            ide = answer(r, w, leaf_qs, leaf_keys, neutral, t)
            acc[e]["category"].append(float(cat == g))
            acc[e]["identity"].append(float(ide == w))
            rec[e] = [cat, ide]
        records.append(rec)
        if (i + 1) % 10 == 0:
            log(f"{i + 1}/{len(words)}")

    summary = {e: {k: sum(v) / len(v) for k, v in d.items()} for e, d in acc.items()}
    nat = acc["native"]["category"]
    for e in EDITS:
        c = acc[e]["category"]
        summary[e]["rescued"] = int(sum(1 for a, b in zip(nat, c) if b > a))
        summary[e]["harmed"] = int(sum(1 for a, b in zip(nat, c) if b < a))
    p, rnd, mm = (acc[k]["category"] for k in (PRIMARY, "rand16_l0.5", "pca16_l0.5_matched"))
    s = summary
    ci_rand = paired_ci(p, rnd)
    b1 = bool(s[PRIMARY]["category"] >= s["native"]["category"] + 0.05
              and s[PRIMARY]["category"] >= s["rand16_l0.5"]["category"] + 0.03
              and s[PRIMARY]["category"] >= s["pca16_l0.5_matched"]["category"] + 0.03
              and ci_rand[1] > 0)
    gates = {
        "B1_inductive_repair": {"pass": b1, "primary_minus_rand16_ci": ci_rand,
                                "primary_minus_native_ci": paired_ci(p, nat),
                                "primary_minus_matched_ci": paired_ci(p, mm)},
        "B2_identity_cost": {"identity_change": s[PRIMARY]["identity"] - s["native"]["identity"],
                             "cheap": s[PRIMARY]["identity"] >= s["native"]["identity"] - 0.05},
        "B3_vs_transductive": {"primary_minus_nn3_ci": paired_ci(p, acc["nn3_l0.5"]["category"])},
    }
    return dict(model=r.name, n_words=len(words), n_vocab=len(vocab), band=list(band),
                families={k: len(v) for k, v in fam.items()}, summary=summary, gates=gates,
                records=records)


def report(res):
    s, g = res["summary"], res["gates"]
    lines = [f"# Bridge report — {res['model']}", "",
             f"{res['n_words']} words, {len(res['families'])} families, PCA basis from "
             f"{res['n_vocab']} unlabeled words, clamp band {res['band']}. Rules: BRIDGE.md.", "",
             f"**B1 inductive repair: {'PASS' if g['B1_inductive_repair']['pass'] else 'FAIL'}**  ",
             "primary − rand16: {:+.3f} [{:+.3f}, {:+.3f}]  ".format(*g["B1_inductive_repair"]["primary_minus_rand16_ci"]),
             "primary − native: {:+.3f} [{:+.3f}, {:+.3f}]  ".format(*g["B1_inductive_repair"]["primary_minus_native_ci"]),
             "primary − matched move: {:+.3f} [{:+.3f}, {:+.3f}]".format(*g["B1_inductive_repair"]["primary_minus_matched_ci"]),
             "",
             f"**B2 identity cost:** {g['B2_identity_cost']['identity_change']:+.3f} "
             f"({'cheap' if g['B2_identity_cost']['cheap'] else 'costly'})", "",
             "**B3 primary − 3-NN bank blend:** {:+.3f} [{:+.3f}, {:+.3f}]".format(*g["B3_vs_transductive"]["primary_minus_nn3_ci"]),
             "", "| edit | category | identity | rescued | harmed |", "|---|---:|---:|---:|---:|"]
    for e in EDITS:
        r = s[e]
        lines.append(f"| {e}{' (primary)' if e == PRIMARY else ''} | {r['category']:.3f} | "
                     f"{r['identity']:.3f} | {r['rescued']} | {r['harmed']} |")
    return "\n".join(lines) + "\n"


def self_test():
    extra = [w for ws in I.BANK.values() for w in ws] + list(I.BANK) + \
            [f"vocabword{c}{d}" for c in "abcdefgh" for d in "abcdefgh"]
    extra = [x for x in extra if x.isalpha()] + ["a", "an", "musical", "instrument"]
    r = C.load_stub("neox", extra=extra)
    # identity edit (lam = 0) must leave every answer unchanged
    fam, words, groups = bank_words(r)
    vocab = vocab_words(r, words, 64)
    assert vocab and not set(vocab) & {w.lower() for w in words}, "vocabulary must be disjoint"
    band = r.band()
    S = r.states(words[:3] + vocab)
    bases = pca_bases(S, vocab, band)
    w = words[0]
    t0 = subspace_edit(S, w, bases, band, 0, 16, 0.0)
    qs = [(k, C.family_q(k)) for k in list(fam)[:3]]
    a, b = r.margins(w, qs), r.margins(w, qs, t0)
    assert max(abs(a[k] - b[k]) for k in a) < 1e-4, "lambda=0 edit changed the answer"
    # full prefix (d = D) at lambda=1 is also the identity
    D = bases[band[0]][1].shape[1]
    tD = subspace_edit(S, w, bases, band, 0, D, 1.0)
    c = r.margins(w, qs, tD)
    assert max(abs(a[k] - c[k]) for k in a) < 1e-3, "full-width projection is not the identity"
    # matched move has the same length as the primary move
    tp = subspace_edit(S, w, bases, band, 0, 16, 0.5)
    tm = matched_move(S, w, tp, 5)
    for l in tp:
        n1 = (tp[l] - S[w][l + 1]).norm()
        n2 = (tm[l] - S[w][l + 1]).norm()
        assert abs(float(n1 - n2)) < 1e-3 * max(1.0, float(n1)), "matched move length differs"
    for layer in r.layers:
        assert not layer._forward_hooks and not layer._forward_pre_hooks, "hook left behind"
    res = run(r, n_vocab=64, log=lambda *_: None)
    assert set(res["summary"]) == set(EDITS)
    print("Bridge self-test passed (random stub model; plumbing only).")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="EleutherAI/pythia-410m")
    p.add_argument("--revision", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--frame", choices=C.FRAMES, default="shots")
    p.add_argument("--n-vocab", type=int, default=N_VOCAB)
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.self_test:
        return self_test()
    revision = a.revision or ("step143000" if "pythia" in a.model.lower() else None)
    r = C.load(a.model, a.device, torch.float32, revision=revision, frame=a.frame)
    res = run(r, n_vocab=a.n_vocab)
    out = Path("runs") / ("bridge_" + a.model.replace("/", "_") + "_" + a.frame +
                          ("_" + revision if revision else ""))
    out.mkdir(parents=True, exist_ok=True)
    (out / "bridge.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (out / "REPORT.md").write_text(report(res), encoding="utf-8")
    print(report(res))
    print("saved", out)


if __name__ == "__main__":
    main()
