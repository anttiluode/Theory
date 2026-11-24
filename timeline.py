"""Timeline: does the tree grow coarse-first inside a model during training?

    python timeline.py --self-test
    python timeline.py                                   (Pythia-410m, 12 checkpoints)
    python timeline.py --model EleutherAI/pythia-1b --steps 0,512,2000,8000,33000,143000

EleutherAI published Pythia's intermediate checkpoints as Hugging Face branches
("step0" ... "step143000"). For each checkpoint this measures:

1. Geometry, no prompting needed. The word states at a few depths are split into the
   tree's Haar levels: how much of their spread is between kingdoms, between families
   inside a kingdom, and between leaves inside a family. Each share is reported as excess
   over 200 random relabelings of the same tree shape.
   Prediction (Saxe, McClelland & Ganguli 2019; coarse modes are learned first): the
   kingdom and family excess rise early; the family excess may then fall back as the
   leaves pull apart.

2. Behaviour, with the plain few-shot frame (Pythia is a base model, so no chat template):
   printed-word category and identity accuracy; then the RakeisuusClaude edit, each leaf
   moved 75% toward its siblings' average, against a random move of the same length.
   Prediction: once printed category accuracy is above 0.6, the sibling move raises
   category evidence more than the matched random move at every later checkpoint.

Checkpoints download once each (about 0.9 GB for 410m, 2 GB for 1b).
"""
import argparse
import gc
import json
import random
import time
from pathlib import Path

import torch

import common as C

STEPS = (0, 128, 512, 1000, 2000, 4000, 8000, 16000, 33000, 66000, 100000, 143000)


def haar_shares(X, labels_f, labels_k):
    """Fractions of the total spread (sum of squared distances to the grand mean) that lie
    between kingdoms, between families inside a kingdom, and between leaves in a family."""
    mu = X.mean(0)
    tot = float(((X - mu) ** 2).sum())
    km = {k: X[[i for i, x in enumerate(labels_k) if x == k]].mean(0) for k in set(labels_k)}
    fm = {f: X[[i for i, x in enumerate(labels_f) if x == f]].mean(0) for f in set(labels_f)}
    top = sum(float(((km[k] - mu) ** 2).sum()) for k in labels_k)
    fam = sum(float(((fm[f] - km[k]) ** 2).sum()) for f, k in zip(labels_f, labels_k))
    leaf = sum(float(((X[i] - fm[f]) ** 2).sum()) for i, f in enumerate(labels_f))
    return dict(top=top / tot, fam=fam / tot, leaf=leaf / tot)


def geometry(S, tree, layers, n_null=200, seed=0):
    words = [w for w, _, _ in tree.leaves]
    lf = [f for _, f, _ in tree.leaves]
    lk = [k for _, _, k in tree.leaves]
    rng = random.Random(seed)
    out = {}
    for l in layers:
        X = torch.stack([S[w][l + 1] for w in words])
        X = X - X.mean(1, keepdim=True)              # what a LayerNorm sees
        obs = haar_shares(X, lf, lk)
        nulls = []
        for _ in range(n_null):
            idx = list(range(len(words)))
            rng.shuffle(idx)
            nulls.append(haar_shares(X[idx], lf, lk))
        res = {}
        for key in obs:
            v = [n[key] for n in nulls]
            m = sum(v) / len(v)
            sd = (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5
            res[key] = dict(obs=obs[key], null=m, excess=obs[key] - m, z=(obs[key] - m) / sd if sd else 0.0)
        out[l] = res
    return out


def behaviour(r, tree, S, band, seed=20261124):
    rows = []
    for i, (w, f, _) in enumerate(tree.leaves):
        qs = tree.questions(f)
        printed = tree.judge(w, r.calibrated(w, qs))
        base = C.blend(S, [(1.0, S[w])], band)
        own = tree.judge(w, r.calibrated(w, qs, base))
        others = [x for x in tree.fam[f][1] if x != w]
        M = C.blend(S, [(0.25, S[w]), (0.75, sum(S[x] for x in others) / len(others))], band)
        R = {l: base[l] + d for l, d in C.matched_random({l: M[l] - base[l] for l in M}, seed + i).items()}
        sib = tree.judge(w, r.calibrated(w, qs, M))
        rnd = tree.judge(w, r.calibrated(w, qs, R))
        rows.append(dict(word=w, printed=printed, own=own, sib=sib, rnd=rnd))
    n = len(rows)
    frac = lambda key, sub: sum(x[sub][key] for x in rows) / n
    return dict(n=n,
                printed_cat=frac("cat_ok", "printed"), printed_leaf=frac("leaf_ok", "printed"),
                own_cat=frac("cat_ok", "own"), own_leaf=frac("leaf_ok", "own"),
                sib_cat=frac("cat_ok", "sib"), sib_leaf=frac("leaf_ok", "sib"),
                d_cat_sib=C.boot([x["sib"]["cat"] - x["own"]["cat"] for x in rows]),
                d_cat_rnd=C.boot([x["rnd"]["cat"] - x["own"]["cat"] for x in rows]),
                sib_minus_rnd=C.boot([x["sib"]["cat"] - x["rnd"]["cat"] for x in rows]),
                cat_up_sib=sum(x["sib"]["cat"] > x["own"]["cat"] for x in rows),
                cat_up_rnd=sum(x["rnd"]["cat"] > x["own"]["cat"] for x in rows), rows=rows)


def one(r, tree, log, band_end=None):
    words = [w for w, _, _ in tree.leaves]
    S = r.states(words)
    L = r.n_layers
    layers = sorted({-1, L // 4, L // 2, (3 * L) // 4, L - 1})
    geo = geometry(S, tree, layers)
    beh = behaviour(r, tree, S, r.band(band_end))
    mid = geo[L // 2]
    log(f"{r.name:34s} mid-layer excess top {mid['top']['excess']:+.3f} fam {mid['fam']['excess']:+.3f} "
        f"| printed cat {beh['printed_cat']:.2f} leaf {beh['printed_leaf']:.2f} "
        f"| 75% to siblings: cat up {beh['cat_up_sib']}/{beh['n']}, random {beh['cat_up_rnd']}/{beh['n']}, "
        f"sib-rnd {beh['sib_minus_rnd']['mean']:+.2f}, identity {beh['sib_leaf']:.2f}")
    return dict(model=r.name, layers=layers, geometry=geo, behaviour=beh)


def run(load_fn, steps, out, log=print, band_end=None):
    out.mkdir(parents=True, exist_ok=True)
    path = out / "timeline.json"
    rep = json.loads(path.read_text()) if path.exists() else dict(steps={})
    tree = None
    for s in steps:
        if str(s) in rep["steps"]:
            log(f"step {s}: already in {path}, skipped")
            continue
        t0 = time.time()
        r = load_fn(s)
        if tree is None:                       # the tokenizer is the same at every checkpoint
            tree = C.Tree(r)
            rep["families"] = {f: ws for f, (_, ws) in tree.fam.items()}
            rep["dropped"] = tree.dropped
            log(f"{len(tree.leaves)} leaves in {len(tree.fam)} families; dropped: {', '.join(tree.dropped) or 'none'}")
        res = one(r, tree, log, band_end)
        res["seconds"] = round(time.time() - t0, 1)
        rep["steps"][str(s)] = res
        path.write_text(json.dumps(rep, default=float))
        del r
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    log(f"saved {path}")
    return rep


def self_test():
    stub = {}

    def load_fn(step):                         # different random weights stand in for checkpoints
        return stub.setdefault(step, C.load_stub("neox", seed=step))
    C.plumbing_test(load_fn(0))
    rep = run(load_fn, [0, 1], Path("runs/selftest_timeline"), log=lambda *a: None)
    assert set(rep["steps"]) == {"0", "1"}
    g = rep["steps"]["0"]["geometry"]
    for l, res in g.items():
        tot = res["top"]["obs"] + res["fam"]["obs"] + res["leaf"]["obs"]
        assert abs(tot - 1) < 1e-4, "the three Haar shares must add up to 1"
    Path("runs/selftest_timeline/timeline.json").unlink()
    print("self-test passed (stub GPT-NeoX); numbers from random models mean nothing")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="EleutherAI/pythia-410m")
    ap.add_argument("--steps", default=",".join(map(str, STEPS)))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--band-end", type=int, default=None)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    steps = [int(s) for s in a.steps.split(",") if s.strip()]
    load_fn = lambda s: C.load(a.model, a.device, torch.float32, revision=f"step{s}")
    run(load_fn, steps, Path("runs") / ("timeline_" + a.model.replace("/", "_")), band_end=a.band_end)


if __name__ == "__main__":
    main()
