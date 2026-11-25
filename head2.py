"""Head experiment 002: the knobs of the closed-loop head, measured (free mode only).

Run: python head2.py --self-test
     python head2.py --device cuda --norms strength.SWOW-EN.R123.20180827.csv
Protocol and frozen gates: HEAD2.md. Reuses head.py's Head (frames, states, clamp, voice,
Qwen's proposal) unchanged; the walk, the conditions and the analysis are new.
"""
import argparse
import json
import random
from pathlib import Path

import torch

import common as C
import head as H0
import ocular as O

KS = [1, 2, 4, 8, 16, 32]
ALPHAS = [0.5, 0.85]
GRID = [f"k{k}_a{a}" for a in ALPHAS for k in KS]
OTHER = [f"flip_a{a}" for a in ALPHAS] + [f"randk_a{a}" for a in ALPHAS] + ["discrete", "native"]
CONDS = GRID + OTHER
T, F = H0.T, H0.F
N_CAL, N_STARTS, N_SAVED_STEPS = 20, 60, (3, 6, 9)
SEED_CAL, SEED_START = 7, 11
DEMO = ["river", "bread", "winter", "iron", "music"]
BONF = 0.05 / len(GRID)


def parse(cond):
    if cond == "discrete":
        return "fixed", 1, 1.0
    rule, a = cond.rsplit("_a", 1)
    if rule.startswith("k"):
        return "fixed", int(rule[1:]), float(a)
    return rule, 4, float(a)


# ---------------------------------------------------------------- the walk (closed loop, no residue)
def step(H, thought, k, alpha, mask_idx):
    """One closed-loop move. Returns (new thought, proposal log-probs, chosen indices)."""
    lp = H.associate(thought)
    base = lp
    for use_mask in (True, False):
        sim = base.clone()
        m = set(mask_idx) if use_mask else {H.index[w] for w in thought if w in H.index}
        if m:
            sim[list(m)] = -float("inf")
        if torch.isfinite(sim).any():
            break
    kk = min(k, int(torch.isfinite(sim).sum()))
    top = torch.topk(sim, kk).indices.tolist()
    tw = torch.softmax(lp[top], -1).tolist()
    new = {w: (1 - alpha) * c for w, c in thought.items() if (1 - alpha) * c >= H0.PRUNE}
    tgt = {}
    for j, wt in zip(top, tw):
        tgt[H.pool[j]] = tgt.get(H.pool[j], 0.0) + alpha * wt
    for w, c in tgt.items():
        new[w] = new.get(w, 0.0) + c
    rest = sorted((w for w in new if w not in tgt), key=lambda w: -new[w])
    keep = set(tgt) | set(rest[:max(0, H0.MAXW - len(tgt))])
    z = sum(new[w] for w in keep)
    return {w: new[w] / z for w in keep}, lp, top


def walk(H, start, cond, band=None, steps=T, seed=0, save_at=()):
    rule, k, alpha = parse(cond)
    rng = random.Random(seed)
    thought = {start: 1.0}
    fatigue = {}
    q_prev = H.vec(thought)
    q0 = q_prev.clone()
    trail, ks, surprises, saved, passes = [], [], [], {}, 0
    for t in range(steps):
        if rule == "randk":
            k = rng.choice(KS)
        mask = {i for i, e in fatigue.items() if e > t} | {H.index[w] for w in thought if w in H.index}
        thought, _, top = step(H, thought, k, alpha, mask)
        passes += 1
        for j in top:
            fatigue[j] = t + 1 + F
        q = H.vec(thought)
        s = float(1 - q @ q_prev)
        surprises.append(s)
        ks.append(k)
        word, n = H.speak(thought)
        passes += n
        trail.append(word)
        if t + 1 in save_at:
            saved[t + 1] = dict(thought)
        if rule == "flip" and band is not None:        # corrected sign (HEAD2.md)
            stuck = len(trail) > 1 and trail[-1] == trail[-2]
            if s < band[0] or stuck:
                k = max(1, k // 2)
            elif s > band[1]:
                k = min(KS[-1], 2 * k)
        q_prev = q
    return dict(trail=trail, k=ks, surprise=surprises, passes=passes, saved=saved,
                start_trace=float(q_prev @ q0))


# ---------------------------------------------------------------- scoring
def collapse(seq):
    out = []
    for w in seq:
        if not out or out[-1] != w:
            out.append(w)
    return out


def rate(seq, norms, collapsed=True):
    s = collapse(seq) if collapsed else list(seq)
    hits = valid = 0
    for a, b in zip(s, s[1:]):
        if a in norms:
            valid += 1
            hits += b in norms[a]
    return (hits / valid) if valid else None


def stats(trail, start, norms):
    seq = [start] + trail
    reps = sum(1 for a, b in zip(seq, seq[1:]) if a == b)
    counts = {}
    for w in trail:
        counts[w] = counts.get(w, 0) + 1
    back = sum(1 for i, w in enumerate(trail) if w in seq[max(0, i - 1):i + 1])
    strength = []
    for a, b in zip(collapse(seq), collapse(seq)[1:]):
        if a in norms:
            strength.append(norms[a].get(b, 0.0))
    return dict(collapsed=rate(seq, norms), raw=rate(seq, norms, False),
                first_half=rate(seq[:7], norms), second_half=rate(seq[6:], norms),
                strength=(sum(strength) / len(strength)) if strength else None,
                repeat=reps / len(trail), distinct=len(counts) / len(trail),
                longest=max(counts.values()), backtrack=back / len(trail))


def paired(a, b, q=0.05, n=4000, seed=0):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return [None, None, None]
    g = torch.Generator().manual_seed(seed)
    d = torch.tensor([x - y for x, y in pairs], dtype=torch.float64)
    draws = torch.stack([d[torch.randint(len(d), (len(d),), generator=g)].mean() for _ in range(n)])
    return [float(d.mean()), float(torch.quantile(draws, q / 2)), float(torch.quantile(draws, 1 - q / 2))]


def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def convergence(trails):
    words = [w for t in trails for w in t]
    in_trails = {}
    for t in trails:
        for w in set(t):
            in_trails[w] = in_trails.get(w, 0) + 1
    highway = sum(1 for w in words if in_trails[w] >= 5)
    return dict(highway_share=highway / max(1, len(words)), unique_share=len(set(words)) / max(1, len(words)),
                top=sorted(in_trails.items(), key=lambda x: -x[1])[:5])


# ---------------------------------------------------------------- the run
def calibrate(H, cal, alpha):
    s = []
    for w in cal:
        s += walk(H, w, f"k4_a{alpha}")["surprise"]
    s = sorted(s)
    return [s[int(0.25 * len(s))], s[min(len(s) - 1, int(0.75 * len(s)))]]


@torch.no_grad()
def control_surface(H, thoughts):
    out = {c: dict(move=[], entropy=[]) for c in GRID}
    for th in thoughts:
        q0 = H.vec(th)
        own = {H.index[w] for w in th if w in H.index}
        for c in GRID:
            _, k, a = parse(c)
            new, _, _ = step(H, th, k, a, own)
            out[c]["move"].append(float(1 - H.vec(new) @ q0))
            lp = H.associate(new)
            out[c]["entropy"].append(float(-(lp.exp() * lp).sum()))
    return {c: dict(move=mean(v["move"]), entropy=mean(v["entropy"])) for c, v in out.items()}


def run(r, norms, norms_name, n_pool=H0.N_POOL, n_starts=N_STARTS, n_cal=N_CAL, steps=T, log=print):
    pool = O.generic_pool(r, [], n_pool)
    H = H0.Head(r, pool, log)
    clean = [w for w in H.pool if w.islower() and w in norms]
    log(f"pool {len(H.pool)}; clean cue words {len(clean)}")
    cal = random.Random(SEED_CAL).sample(clean, n_cal)
    rest = [w for w in clean if w not in set(cal)]
    starts = random.Random(SEED_START).sample(rest, min(n_starts, len(rest)))
    demo = [w for w in DEMO if w in H.index and w in norms and w not in starts and w not in cal]
    bands = {a: calibrate(H, cal, a) for a in ALPHAS}
    for a, b in bands.items():
        log(f"flip band at alpha {a}: [{b[0]:.4f}, {b[1]:.4f}]")
    res = {c: [] for c in CONDS}
    saved = []
    for i, w in enumerate(starts + demo):
        for c in CONDS:
            if c == "native":
                out = H0.native_chain(H, w, steps)
            else:
                rule, _, a = parse(c)
                out = walk(H, w, c, band=bands.get(a) if rule == "flip" else None, steps=steps,
                           seed=1000 + i, save_at=N_SAVED_STEPS if c == "k4_a0.5" and w in starts else ())
                if out["saved"]:
                    saved += list(out["saved"].values())
                out.pop("saved")
            out.update(start=w, demo=w in demo, **stats(out["trail"], w, norms))
            res[c].append(out)
        if (i + 1) % 5 == 0:
            log(f"trails {i + 1}/{len(starts) + len(demo)}")
    thoughts = saved[:180] + [{w: 1.0} for w in starts[:20]]
    log(f"control surface on {len(thoughts)} saved thoughts")
    surface = control_surface(H, thoughts)
    out = dict(model=r.name, n_pool=len(H.pool), norms=norms_name, steps=steps, calibration=cal,
               starts=starts, demo=demo, bands={str(a): b for a, b in bands.items()}, trails=res,
               surface=surface, n_surface=len(thoughts))
    out["gates"] = gates(out)
    return out


def gates(res):
    tr = res["trails"]
    scored = [i for i, x in enumerate(tr[GRID[0]]) if not x["demo"]]
    col = lambda c, key="collapsed": [tr[c][i][key] for i in scored]
    g = {}
    # G1: per-start choice on the first half, scored on the second half
    first = {c: col(c, "first_half") for c in GRID}
    second = {c: col(c, "second_half") for c in GRID}
    glob = max(GRID, key=lambda c: (mean(first[c]) if mean(first[c]) is not None else -1))
    per, gl = [], []
    for j in range(len(scored)):
        cands = [c for c in GRID if first[c][j] is not None]
        if not cands:
            per.append(None)
            gl.append(None)
            continue
        best = max(cands, key=lambda c: (first[c][j], -GRID.index(c)))
        per.append(second[best][j])
        gl.append(second[glob][j])
    g1 = paired(per, gl)
    in_sample = [max((col(c)[j] for c in GRID if col(c)[j] is not None), default=None) for j in range(len(scored))]
    best_cell = max(GRID, key=lambda c: mean(col(c)) or -1)
    g["G1_headroom"] = dict(pass_=bool(g1[1] is not None and g1[1] > 0), per_start_minus_global=g1,
                            global_cell_first_half=glob,
                            in_sample_oracle=mean(in_sample), best_cell=best_cell, best_cell_mean=mean(col(best_cell)))
    # G2: corrected regulator at the alpha whose k4 cell is higher
    a = max(ALPHAS, key=lambda x: mean(col(f"k4_a{x}")) or -1)
    d_k4 = paired(col(f"flip_a{a}"), col(f"k4_a{a}"))
    d_rk = paired(col(f"flip_a{a}"), col(f"randk_a{a}"))
    g["G2_corrected_regulator"] = dict(pass_=bool(d_k4[1] is not None and d_k4[1] > 0 and d_rk[1] > 0), alpha=a,
                                       flip_minus_k4=d_k4, flip_minus_randk=d_rk,
                                       flip_minus_best_cell=paired(col(f"flip_a{a}"), col(best_cell), q=BONF))
    d3 = paired(col(best_cell), col("discrete"), q=BONF)
    g["G3_blend_vs_words"] = dict(pass_=bool(d3[1] is not None and d3[1] > 0), best_cell=best_cell,
                                  best_cell_minus_discrete=d3,
                                  native_minus_best_cell=paired(col("native"), col(best_cell), q=BONF))
    return g


# ---------------------------------------------------------------- report
def report(res):
    f = lambda v: "–" if v is None else f"{v:.3f}"
    ci = lambda v: "–" if v is None or v[0] is None else "{:+.3f} [{:+.3f}, {:+.3f}]".format(*v)
    g, tr = res["gates"], res["trails"]
    P = lambda k: "PASS" if g[k]["pass_"] else "FAIL"
    scored = lambda c: [x for x in tr[c] if not x["demo"]]
    m = lambda c, key: mean([x[key] for x in scored(c)])
    L = [f"# Head experiment 002 report — {res['model']}", "",
         f"Pool {res['n_pool']}; {len(scored(GRID[0]))} scored starts; {res['steps']} steps; residue off; "
         f"norms {res['norms']}. Flip bands: " +
         "; ".join(f"α {a} [{b[0]:.4f}, {b[1]:.4f}]" for a, b in res["bands"].items()) + ". Rules: HEAD2.md.", "",
         f"**G1 headroom (per-start choice beats one global setting, held out): {P('G1_headroom')}**  "
         f"{ci(g['G1_headroom']['per_start_minus_global'])}; global cell chosen on first halves: "
         f"`{g['G1_headroom']['global_cell_first_half']}`; best cell overall `{g['G1_headroom']['best_cell']}` "
         f"{f(g['G1_headroom']['best_cell_mean'])}; in-sample per-start best (biased, not gated) "
         f"{f(g['G1_headroom']['in_sample_oracle'])}", "",
         f"**G2 corrected regulator (α {g['G2_corrected_regulator']['alpha']}): {P('G2_corrected_regulator')}**  "
         f"flip − k4 {ci(g['G2_corrected_regulator']['flip_minus_k4'])}; flip − randk "
         f"{ci(g['G2_corrected_regulator']['flip_minus_randk'])}; flip − best cell (Bonferroni) "
         f"{ci(g['G2_corrected_regulator']['flip_minus_best_cell'])}", "",
         f"**G3 blending states beats choosing words: {P('G3_blend_vs_words')}**  "
         f"{g['G3_blend_vs_words']['best_cell']} − discrete (Bonferroni) "
         f"{ci(g['G3_blend_vs_words']['best_cell_minus_discrete'])}; native − best cell "
         f"{ci(g['G3_blend_vs_words']['native_minus_best_cell'])}", "",
         "## The grid (collapsed associate rate; raw rate; repeat fraction; distinct words)", "",
         "| k | " + " | ".join(f"α {a}" for a in ALPHAS) + " |", "|---:|" + "---|" * len(ALPHAS)]
    for k in KS:
        L.append(f"| {k} | " + " | ".join(
            f"{f(m(f'k{k}_a{a}', 'collapsed'))}; {f(m(f'k{k}_a{a}', 'raw'))}; {f(m(f'k{k}_a{a}', 'repeat'))}; "
            f"{f(m(f'k{k}_a{a}', 'distinct'))}" for a in ALPHAS) + " |")
    L += ["", "## All conditions", "",
          "| condition | collapsed | raw | strength | repeat | distinct | longest | back-tracks | highway share | unique share | start trace | passes |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in CONDS:
        cv = convergence([x["trail"] for x in scored(c)])
        L.append(f"| {c} | {f(m(c, 'collapsed'))} | {f(m(c, 'raw'))} | {f(m(c, 'strength'))} | {f(m(c, 'repeat'))} | "
                 f"{f(m(c, 'distinct'))} | {f(m(c, 'longest'))} | {f(m(c, 'backtrack'))} | {f(cv['highway_share'])} | "
                 f"{f(cv['unique_share'])} | {f(m(c, 'start_trace')) if c != 'native' else '–'} | {f(m(c, 'passes'))} |")
    L += ["", "Words shared by the most trails: " + "; ".join(
        f"{c}: " + ", ".join(f"{w} {n}" for w, n in convergence([x['trail'] for x in scored(c)])['top'][:3])
        for c in ("k1_a0.85", "k4_a0.5", "discrete", "native")), "",
          f"## Control surface ({res['n_surface']} identical saved thoughts, one step each)", "",
          "| k | " + " | ".join(f"move α {a} | next-proposal entropy α {a}" for a in ALPHAS) + " |",
          "|---:|" + "---:|" * (2 * len(ALPHAS))]
    for k in KS:
        L.append(f"| {k} | " + " | ".join(f"{f(res['surface'][f'k{k}_a{a}']['move'])} | "
                                          f"{f(res['surface'][f'k{k}_a{a}']['entropy'])}" for a in ALPHAS) + " |")
    L += ["", "## Trails", ""]
    show = [i for i, x in enumerate(tr[GRID[0]]) if x["demo"]] + list(range(4))
    for i in dict.fromkeys(show):
        L.append(f"**{tr[GRID[0]][i]['start']}**  ")
        for c in ("k1_a0.85", "k4_a0.5", "k32_a0.85", "flip_a0.85", "discrete", "native"):
            ks = f"  (k: {' '.join(map(str, tr[c][i]['k']))})" if c.startswith("flip") else ""
            L.append(f"- {c}: " + " → ".join(tr[c][i]["trail"]) + ks)
        L.append("")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- self-test
def self_test():
    texts = [H0.NAME_SHOTS, H0.NAME_Q, H0.ASSOC_SHOTS, H0.ASSOC_FRAME, H0.ASSOC_Q,
             "oak tree wood river water bread iron music winter One word Word Association"]
    texts += [f"pool{c}{d}" for c in "abcdefgh" for d in "abcdefgh"]
    r = C.load_stub("qwen", extra=texts)
    pool = O.generic_pool(r, [], 120)
    Hh = H0.Head(r, pool, log=lambda *_: None)
    words = [w for w in Hh.pool if w.islower()]
    norms = {w: {words[(i + 1) % len(words)]: 0.2, "tree": 0.1} for i, w in enumerate(words)}
    assert parse("k8_a0.85") == ("fixed", 8, 0.85) and parse("discrete") == ("fixed", 1, 1.0)
    assert parse("flip_a0.5") == ("flip", 4, 0.5)
    assert collapse(["a", "a", "b", "b", "a"]) == ["a", "b", "a"]
    n2 = {"a": {"b": 1.0}, "b": {"c": 1.0}}
    assert rate(["a", "a", "b", "c"], n2) == 1.0 and abs(rate(["a", "a", "b", "c"], n2, False) - 2 / 3) < 1e-9
    w = words[2]
    d = walk(Hh, w, "discrete", steps=3)
    assert len(d["trail"]) == 3 and d["k"] == [1, 1, 1]
    f = walk(Hh, w, "flip_a0.5", band=[5.0, 6.0], steps=4)       # always below the band -> zoom in
    assert f["k"] == [4, 2, 1, 1], f["k"]
    f = walk(Hh, w, "flip_a0.5", band=[-2.0, -1.0], steps=4)     # always above -> zoom out (unless stuck)
    assert f["k"][1] in (8, 2)
    rk = walk(Hh, w, "randk_a0.85", steps=5, seed=3)
    assert all(k in KS for k in rk["k"])
    res = run(r, norms, "selftest", n_pool=120, n_starts=4, n_cal=3, steps=7, log=lambda *_: None)
    assert set(res["trails"]) == set(CONDS) and len(res["surface"]) == len(GRID)
    assert res["n_surface"] > 0
    json.dumps(res)
    txt = report(res)
    assert "G1 headroom" in txt and "G3 blending" in txt
    for layer in r.layers:
        assert not layer._forward_hooks and not layer._forward_pre_hooks, "hook left behind"
    print("Head2 self-test passed (random Qwen stub; plumbing only).")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    p.add_argument("--norms", default=None)
    p.add_argument("--n-pool", type=int, default=H0.N_POOL)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--report-from", default=None)
    a = p.parse_args()
    if a.self_test:
        return self_test()
    if a.report_from:
        src = Path(a.report_from)
        res = json.loads(src.read_text(encoding="utf-8"))
        (src.parent / "REPORT.md").write_text(report(res), encoding="utf-8")
        return print(report(res))
    if not a.norms:
        raise SystemExit("--norms is required (HEAD2.md)")
    norms = H0.load_norms(a.norms)
    print(f"loading {a.model} as {a.dtype} ...", flush=True)
    r = C.load(a.model, a.device, getattr(torch, a.dtype), frame="shots")
    print("loaded", flush=True)
    res = run(r, norms, Path(a.norms).name, n_pool=a.n_pool, log=lambda *x: print(*x, flush=True))
    res["dtype"] = a.dtype
    out = Path("runs") / ("head2_" + a.model.replace("/", "_"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "head2.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (out / "REPORT.md").write_text(report(res), encoding="utf-8")
    print(report(res))
    print("saved", out)


if __name__ == "__main__":
    main()
