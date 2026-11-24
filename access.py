"""Access: is the map there before the model can read it?

    python access.py --self-test
    python access.py                    (Pythia-410m, the timeline checkpoints; downloads are cached)

timeline.py showed the tree's geometry in Pythia's word states by step 1000, while the model's
own yes/no answers were still below chance. That gap means one of two things. Either the
states already carry category information a simple reader could use, and the model has not yet
learned to read it. Or the geometry is real but not yet usable on words the reader has not seen.

The external reader here is the simplest one possible, a prototype reader with nothing to tune.
Hold out one word. Build each family's centroid from the remaining words (a held-out word's
own family loses it), subtract the mean of all remaining words, and give the held-out word the
family whose centroid is closest in angle. The same is done for kingdoms. Chance is 1/families
and 1/kingdoms.

Two harder readers sit between that one and the model's own answers:
- unseen-family kingdom: the held-out word's whole family is removed before the kingdom
  centroids are built, so a family the reader never saw must be placed (genuine coarse
  abstraction rather than recognising a familiar family cluster);
- category word: no labelled examples at all. The word goes to the family whose own name
  ("tree", "bird", ...), read as an item in the same frame, lies closest. This asks whether the
  cluster is linked to the word for it, which the native yes/no reader needs as well.

It runs at every depth and every checkpoint, next to the model's own printed-word category
accuracy from timeline.py (the native reader). Prediction, fixed before the run: at step 1000,
the prototype reader is well above chance at the middle layers while the native reader is not.
"""
import argparse
import gc
import json
from pathlib import Path

import torch

import common as C
import timeline as T


def prototype_reader(X, labels):
    """Leave-one-out nearest-centroid accuracy by cosine. X [n, d], labels list of n."""
    n = len(labels)
    ok = 0
    for i in range(n):
        keep = [j for j in range(n) if j != i]
        mu = X[keep].mean(0)
        best, best_s = None, -2.0
        for lab in sorted(set(labels)):
            idx = [j for j in keep if labels[j] == lab]
            if not idx:
                continue
            c = X[idx].mean(0) - mu
            s = float(torch.nn.functional.cosine_similarity(X[i] - mu, c, dim=0))
            if s > best_s:
                best, best_s = lab, s
        ok += best == labels[i]
    return ok / n


def unseen_family_reader(X, fams, kings):
    """Kingdom of a word whose whole family is held out: centroids come from the other families
    only, so the reader must place a family it has never seen."""
    n = len(fams)
    ok = 0
    for i in range(n):
        keep = [j for j in range(n) if fams[j] != fams[i]]
        mu = X[keep].mean(0)
        sims = {}
        for k in set(kings):
            idx = [j for j in keep if kings[j] == k]
            if idx:
                sims[k] = float(torch.nn.functional.cosine_similarity(X[i] - mu, X[idx].mean(0) - mu, dim=0))
        ok += max(sims, key=sims.get) == kings[i]
    return ok / n


def name_reader(X, labels, N, names):
    """Family by the closest category *word* ("tree", "bird", ...), read as an item in the same
    frame; each set is centred on its own mean. No labelled examples at all."""
    mu, nu = X.mean(0), N.mean(0)
    ok = 0
    for i in range(len(labels)):
        s = torch.nn.functional.cosine_similarity((X[i] - mu)[None], N - nu, dim=1)
        ok += names[int(s.argmax())] == labels[i]
    return ok / len(labels)


def one(r, tree):
    words = [w for w, _, _ in tree.leaves]
    S = r.states(words)
    lf = [f for _, f, _ in tree.leaves]
    lk = [k for _, _, k in tree.leaves]
    names = [f for f in tree.fam if r.single(f)]
    NS = r.states(names)
    named = [i for i, f in enumerate(lf) if f in names]
    fam, king, unseen, name = {}, {}, {}, {}
    for l in range(-1, r.n_layers):
        X = torch.stack([S[w][l + 1] for w in words])
        X = X - X.mean(1, keepdim=True)            # what a LayerNorm sees
        fam[l] = prototype_reader(X, lf)
        king[l] = prototype_reader(X, lk)
        unseen[l] = unseen_family_reader(X, lf, lk)
        if len(names) >= 2:
            N = torch.stack([NS[f][l + 1] for f in names])
            N = N - N.mean(1, keepdim=True)
            name[l] = name_reader(X[named], [lf[i] for i in named], N, names)
    return dict(family=fam, kingdom=king, kingdom_unseen_family=unseen, name=name, names=names,
                n=len(words), n_named=len(named), chance_family=1 / len(set(lf)),
                chance_kingdom=1 / len(set(lk)), chance_name=1 / max(1, len(names)))


def run(load_fn, steps, out, native=None, log=print, min_leaves=3):
    out.mkdir(parents=True, exist_ok=True)
    path = out / "access.json"
    rep = json.loads(path.read_text()) if path.exists() else dict(steps={})
    tree = None
    for s in steps:
        if "name" in rep["steps"].get(str(s), {}):
            continue
        r = load_fn(s)
        if tree is None:
            tree = C.Tree(r, min_leaves=min_leaves)
            rep["families"] = {f: ws for f, (_, ws) in tree.fam.items()}
        res = one(r, tree)
        nat = (native or {}).get(str(s))
        res["native_printed_cat"] = nat
        rep["steps"][str(s)] = res
        L = r.n_layers
        mid = res["family"][L // 2]
        best_l = max(res["family"], key=res["family"].get)
        log(f"step {s:>6}: prototype reader, family {res['family'][-1]:.2f} at the embedding, "
            f"{mid:.2f} at layer {L // 2}, best {res['family'][best_l]:.2f} (layer {best_l}); "
            f"kingdom {res['kingdom'][L // 2]:.2f}, unseen-family kingdom {res['kingdom_unseen_family'][L // 2]:.2f}, "
            f"by category word {res['name'].get(L // 2, float('nan')):.2f} ({len(res['names'])} names) | native "
            f"{'n/a' if nat is None else f'{nat:.2f}'} | chance {res['chance_family']:.2f} / "
            f"{res['chance_kingdom']:.2f}")
        path.write_text(json.dumps(rep, default=float))
        del r
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    log(f"saved {path}")
    return rep


def self_test():
    # perfectly separable toy data must read 1.0; shuffled labels must read near chance
    torch.manual_seed(0)
    labels = [i // 4 for i in range(24)]
    X = torch.randn(24, 32) * 0.1 + torch.stack([torch.eye(32)[l] * 3 for l in labels])
    assert prototype_reader(X, labels) == 1.0
    stub = {}
    load_fn = lambda s: stub.setdefault(s, C.load_stub("neox", seed=s))
    rep = run(load_fn, [0, 1], Path("runs/selftest_access"), log=lambda *a: None)
    assert set(rep["steps"]) == {"0", "1"}
    Path("runs/selftest_access/access.json").unlink()
    print("self-test passed (toy data and stub GPT-NeoX); numbers from random models mean nothing")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="EleutherAI/pythia-410m")
    ap.add_argument("--steps", default=",".join(map(str, T.STEPS)))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--min-leaves", type=int, default=3)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    steps = [int(s) for s in a.steps.split(",") if s.strip()]
    tag = a.model.replace("/", "_") + ("" if a.min_leaves == 4 else f"_min{a.min_leaves}")
    tl = Path("runs") / ("timeline_" + tag) / "timeline.json"
    native = None
    if tl.exists():
        native = {s: v["behaviour"]["printed_cat"] for s, v in json.loads(tl.read_text())["steps"].items()}
    load_fn = lambda s: C.load(a.model, a.device, torch.float32, revision=f"step{s}")
    run(load_fn, steps, Path("runs") / ("access_" + tag), native=native, min_leaves=a.min_leaves)


if __name__ == "__main__":
    main()
