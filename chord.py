"""Chord: is the sibling-average result just the geometry of normalization?

    python chord.py --self-test
    python chord.py                         (Qwen3-1.7B, the RakeisuusClaude checkpoint)
    python chord.py --model EleutherAI/pythia-1b

Three parts, all on every single-token leaf (nothing is fitted, so there is no train/test split):

A  The pending control. Move each leaf 75% toward the average of its siblings, and
   separately move it the SAME distance in a random direction (matched at every layer
   to the sibling move, not to the category-word move as in RakeisuusClaude).

B  The chord prediction. The model normalizes before every read. Two siblings a, b of
   similar length lie near a sphere; their blend (1-t)a + tb cuts inside it, so it is
   shorter, and the norm stretches it back. Anything a and b share is stretched with it,
   by g(t) = ((1-t)|a| + t|b|) / |(1-t)a + tb|, which at t = 0.5 and equal lengths is
   1 / cos(phi/2), phi the angle between them. Prediction: the category overshoot of a
   blend over the straight line between its ends grows with g. Across all sibling pairs,
   measured overshoot is correlated with g (Spearman, permutation p).

C  Real versus synthetic sibling. Replace b by s = family mean + (a random direction with
   b's distance from the mean). s shares only the family average with a, and nothing of
   b's private part. If the overshoot is pure normalization geometry, synthetic siblings
   overshoot as much as real ones at the same g. If real siblings overshoot more, their
   private parts carry something that adds up.
"""
import argparse
import itertools
import json
import time
from pathlib import Path

import torch

import common as C

T_BLEND = (0.25, 0.5, 0.75)


def chord_gain(S, a, b, t, band, centered):
    gs = []
    for l in range(band[0], band[1] + 1):
        x, y = S[a][l + 1], S[b][l + 1]
        if centered:
            x, y = x - x.mean(), y - y.mean()
        gs.append(((1 - t) * x.norm() + t * y.norm()) / ((1 - t) * x + t * y).norm().clamp_min(1e-9))
    return float(torch.stack(gs).mean())


def run(r, out, log=print, band_end=None, seed=20261124):
    t0 = time.time()
    tree = C.Tree(r)
    band = r.band(band_end)
    words = [w for w, _, _ in tree.leaves]
    S = r.states(words)
    log(f"{r.name}: {r.n_layers} layers, band {band}, {len(words)} leaves in {len(tree.fam)} families; "
        f"dropped (not one token): {', '.join(tree.dropped) or 'none'}")
    rep = dict(model=r.name, band=band, centered_norm=r.centered, families={f: ws for f, (_, ws) in tree.fam.items()},
               dropped=tree.dropped)
    own = {}
    for w, f, _ in tree.leaves:
        own[w] = tree.judge(w, r.calibrated(w, tree.questions(f), C.blend(S, [(1.0, S[w])], band)))

    # ---- A: sibling move vs equal-length random move
    rowsA = []
    for i, (w, f, _) in enumerate(tree.leaves):
        others = [x for x in tree.fam[f][1] if x != w]
        mean_o = sum(S[x] for x in others) / len(others)
        M = C.blend(S, [(0.25, S[w]), (0.75, mean_o)], band)
        base = C.blend(S, [(1.0, S[w])], band)
        delta = {l: M[l] - base[l] for l in M}
        R = {l: base[l] + d for l, d in C.matched_random(delta, seed + i).items()}
        jm = tree.judge(w, r.calibrated(w, tree.questions(f), M))
        jr = tree.judge(w, r.calibrated(w, tree.questions(f), R))
        rowsA.append(dict(word=w, family=f, own=own[w], sib=jm, rnd=jr,
                          d_cat_sib=jm["cat"] - own[w]["cat"], d_cat_rnd=jr["cat"] - own[w]["cat"]))
    A = dict(rows=rowsA,
             cat_up_sib=sum(x["d_cat_sib"] > 0 for x in rowsA), cat_up_rnd=sum(x["d_cat_rnd"] > 0 for x in rowsA),
             d_cat_sib=C.boot([x["d_cat_sib"] for x in rowsA]), d_cat_rnd=C.boot([x["d_cat_rnd"] for x in rowsA]),
             sib_minus_rnd=C.boot([x["d_cat_sib"] - x["d_cat_rnd"] for x in rowsA]),
             identity_own=sum(x["own"]["leaf_ok"] for x in rowsA), identity_sib=sum(x["sib"]["leaf_ok"] for x in rowsA),
             identity_rnd=sum(x["rnd"]["leaf_ok"] for x in rowsA), n=len(rowsA))
    rep["A"] = A
    f2 = lambda b: f"{b['mean']:+.2f} [{b['lo']:+.2f}, {b['hi']:+.2f}]"
    log(f"\nA  75% toward siblings vs the same distance at random (matched per layer to the sibling move), n={A['n']}")
    log(f"   category evidence rose: siblings {A['cat_up_sib']}/{A['n']}, random {A['cat_up_rnd']}/{A['n']}")
    log(f"   mean change: siblings {f2(A['d_cat_sib'])}, random {f2(A['d_cat_rnd'])}, "
        f"difference {f2(A['sib_minus_rnd'])}")
    log(f"   exact identity: own {A['identity_own']}, siblings {A['identity_sib']}, random {A['identity_rnd']}")

    # ---- B and C: all sibling pairs
    rowsB = []
    gen = torch.Generator().manual_seed(seed)
    for f, (k, ws) in tree.fam.items():
        mean_f = sum(S[x] for x in ws) / len(ws)
        for a, b in itertools.combinations(ws, 2):
            qs = tree.questions(f)
            row = dict(family=f, a=a, b=b, ends=dict(a=own[a]["cat"], b=own[b]["cat"]), blend={}, gain={})
            for t in T_BLEND:
                tg = C.blend(S, [(1 - t, S[a]), (t, S[b])], band)
                c = r.calibrated(a, qs, tg)
                lin = (1 - t) * own[a]["cat"] + t * own[b]["cat"]
                row["blend"][t] = dict(cat=c[f"cat:{f}"], over_line=c[f"cat:{f}"] - lin,
                                       over_both=c[f"cat:{f}"] - max(own[a]["cat"], own[b]["cat"]),
                                       rejects_both=c[f"leaf:{a}"] < 0 and c[f"leaf:{b}"] < 0)
                row["gain"][t] = chord_gain(S, a, b, t, band, r.centered)
            # C: a synthetic sibling with b's distance from the family mean, random direction
            s = torch.stack([mean_f[i] + (S[b][i] - mean_f[i]).norm() *
                             (lambda u: u / u.norm())(torch.randn(mean_f.shape[1], generator=gen))
                             for i in range(mean_f.shape[0])])
            S["_syn"] = s
            cs_end = r.calibrated(a, qs, C.blend(S, [(1.0, s)], band))[f"cat:{f}"]
            cs_mid = r.calibrated(a, qs, C.blend(S, [(0.5, S[a]), (0.5, s)], band))[f"cat:{f}"]
            row["synthetic"] = dict(end=cs_end, mid=cs_mid, over_line=cs_mid - 0.5 * (own[a]["cat"] + cs_end),
                                    gain=chord_gain(S, a, "_syn", 0.5, band, r.centered))
            del S["_syn"]
            rowsB.append(row)
    g = [x["gain"][0.5] for x in rowsB]
    ov = [x["blend"][0.5]["over_line"] for x in rowsB]
    B = dict(rows=rowsB, n_pairs=len(rowsB),
             over_line_mid=C.boot(ov),
             over_both_mid=sum(x["blend"][0.5]["over_both"] > 0 for x in rowsB),
             rejects_both_mid=sum(x["blend"][0.5]["rejects_both"] for x in rowsB),
             spearman_gain_vs_overshoot=C.spearman(g, ov), perm_p=C.perm_p(g, ov),
             gain_mid=C.boot(g))
    syn_ov = [x["synthetic"]["over_line"] for x in rowsB]
    syn_g = [x["synthetic"]["gain"] for x in rowsB]
    Cc = dict(real_over_line=C.boot(ov), synthetic_over_line=C.boot(syn_ov),
              real_minus_synthetic=C.boot([a - b for a, b in zip(ov, syn_ov)]),
              real_gain=C.boot(g), synthetic_gain=C.boot(syn_g),
              spearman_gain_vs_overshoot_all=C.spearman(g + syn_g, ov + syn_ov),
              perm_p_all=C.perm_p(g + syn_g, ov + syn_ov))
    rep["B"], rep["C"] = B, Cc
    log(f"\nB  50/50 blends of sibling pairs, n={B['n_pairs']}")
    log(f"   category above the straight line between the ends: {f2(B['over_line_mid'])} nats")
    log(f"   above BOTH ends: {B['over_both_mid']}/{B['n_pairs']}   rejects both names: "
        f"{B['rejects_both_mid']}/{B['n_pairs']}")
    log(f"   chord gain g(0.5): {f2(B['gain_mid'])}   Spearman(g, overshoot) = "
        f"{B['spearman_gain_vs_overshoot']:+.2f}, permutation p = {B['perm_p']:.3f}")
    log(f"\nC  real sibling vs synthetic sibling (family mean + random private part of the same size)")
    log(f"   overshoot: real {f2(Cc['real_over_line'])}, synthetic {f2(Cc['synthetic_over_line'])}, "
        f"real - synthetic {f2(Cc['real_minus_synthetic'])}")
    log(f"   chord gain: real {f2(Cc['real_gain'])}, synthetic {f2(Cc['synthetic_gain'])}; "
        f"Spearman(g, overshoot) over both = {Cc['spearman_gain_vs_overshoot_all']:+.2f}, "
        f"p = {Cc['perm_p_all']:.3f}")
    rep["seconds"] = round(time.time() - t0, 1)
    out.mkdir(parents=True, exist_ok=True)
    (out / "chord.json").write_text(json.dumps(rep, indent=1, default=float))
    log(f"\nsaved {out / 'chord.json'} ({rep['seconds']} s)")
    return rep


def self_test():
    for arch in ("qwen", "neox"):
        r = C.load_stub(arch)
        C.plumbing_test(r)
        rep = run(r, Path("runs/selftest_" + arch), log=lambda *a: None)
        assert rep["A"]["n"] > 0 and rep["B"]["n_pairs"] > 0
        # the chord gain is 1 at the ends and >= 1 in between (triangle inequality)
        S = r.states(["oak", "pine"])
        assert abs(chord_gain(S, "oak", "pine", 0.0, r.band(), r.centered) - 1) < 1e-5
        assert chord_gain(S, "oak", "pine", 0.5, r.band(), r.centered) >= 1 - 1e-6
    print("self-test passed (stub Qwen2 and GPT-NeoX); numbers from random models mean nothing")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--band-end", type=int, default=None)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    r = C.load(a.model, a.device, torch.float32, a.revision)
    run(r, Path("runs") / (r.name.replace("/", "_").replace("@", "_")), band_end=a.band_end)


if __name__ == "__main__":
    main()
