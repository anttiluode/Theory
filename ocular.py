"""Ocular: choose neighbours through a narrow PCA band, carry their full states.

Run: python ocular.py --self-test
     python ocular.py --device cuda
Protocol and frozen gates: OCULAR.md. No training of the LM. Uses Iris's words, frame,
neutral calibration, clamp and blend; only neighbour selection and the neighbour pool change.
"""
import argparse
import json
import random
from pathlib import Path

import torch

import bridge as B
import common as C
import iris as I

N_POOL = 8000
PRIMARY = "band4_vocab"
SPACES = {"full": None, "pre16": (0, 16), "band4": (4, 16), "band8": (8, 16), "rand16": "rand"}
POOLS = ("bank", "vocab")
CONDITIONS = (["native", "iris"] + [f"{s}_{p}" for p in POOLS for s in SPACES] + ["random3_vocab"])


@torch.no_grad()
def margins_lean(r, word, questions, targets=None):
    """Same numbers as common.Reader.margins, but the output head is applied only at each
    prompt's last position. The full-sequence logits (batch x tokens x 152k for Qwen) ran the
    12 GB GPU out of memory. Uses the base model (which includes the final norm) and the
    model's own output embedding; clamp hooks are Theory's own (Reader._hooks)."""
    encs = [r.prompt(word, q) for _, q in questions]
    pos = encs[0][1]
    assert pos is not None and all(p == pos for _, p in encs)
    n = max(len(ids) for ids, _ in encs)
    x = torch.full((len(encs), n), r.pad, dtype=torch.long)
    mask = torch.zeros((len(encs), n), dtype=torch.long)
    for b, (ids, _) in enumerate(encs):
        x[b, :len(ids)] = torch.tensor(ids)
        mask[b, :len(ids)] = 1
    last = (mask.sum(1) - 1).to(r.device)
    hooks = r._hooks(pos, targets)
    try:
        h = r.model.base_model(input_ids=x.to(r.device), attention_mask=mask.to(r.device)).last_hidden_state
    finally:
        for hk in hooks:
            hk.remove()
    hl = h[torch.arange(len(encs), device=h.device), last]
    lp = torch.log_softmax(r.model.get_output_embeddings()(hl).float(), -1)
    m = torch.logsumexp(lp[:, r.yes], -1) - torch.logsumexp(lp[:, r.no], -1)
    return {k: float(v) for (k, _), v in zip(questions, m.cpu())}


def answer(r, w, qs, keys, neutral, targets=None):
    m = margins_lean(r, w, qs, targets)
    z = torch.tensor([m[k] for k, _ in qs]) - torch.tensor([neutral[k] for k, _ in qs])
    return keys[int(z.argmax())]


def generic_pool(r, exclude, n):
    """Single-token words, lowercase or title-case, 3+ letters, disjoint from bank and shots."""
    shot = {w.lower() for w in C.SHOTS.replace(".", " ").replace("?", " ").split()}
    banned = {w.lower() for w in exclude} | shot | {C.SLOT}
    if hasattr(r.tok, "get_vocab"):
        items = sorted(r.tok.get_vocab().items(), key=lambda kv: kv[1])
        cands = [t[1:] for t, _ in items if t[:1] in ("Ġ", "▁")]
    else:  # test stub
        cands = [t.strip() for t in r.tok.vocab if t.startswith(" ")]
    out, seen = [], set()
    for w in cands:
        if len(w) < 3 or not w.isalpha() or not (w.islower() or w.istitle()):
            continue
        if w.lower() in banned or w in seen:
            continue
        seen.add(w)
        if r.single(w):
            out.append(w)
        if len(out) >= n:
            break
    return out


def selection_spaces(Xpool, layer_dim, seed=41):
    """Label-free projections fitted on the generic pool only."""
    mu = Xpool.double().mean(0)
    # PCA through the D x D covariance (eigh) instead of an N x D SVD: same directions,
    # far less memory (the SVD ran out of RAM on Windows for Qwen: 8000 x 2048)
    Z = Xpool.double() - mu
    _, evecs = torch.linalg.eigh(Z.T @ Z)
    V = evecs.flip(1)
    del Z
    g = torch.Generator().manual_seed(seed)
    q, rr = torch.linalg.qr(torch.randn(layer_dim, layer_dim, generator=g, dtype=torch.float64))
    R = q * torch.sign(torch.diagonal(rr))
    proj = {}
    for name, spec in SPACES.items():
        if spec is None:
            proj[name] = None
        elif spec == "rand":
            proj[name] = R[:, :16]
        else:
            k0, d = spec
            proj[name] = V[:, k0:k0 + d]
    return mu, proj


def embed(X, mu, P, centered):
    Z = X.double()
    if centered:  # measure angles the way LayerNorm sees them, as Iris does
        Z = Z - Z.mean(1, keepdim=True)
    Z = Z - mu
    if P is not None:
        Z = Z @ P
    return C_unit(Z)


def C_unit(Z):
    return Z / Z.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def top3(q, pool, exclude=None):
    sim = pool @ q
    if exclude is not None:
        sim[exclude] = -float("inf")
    return torch.argsort(sim, descending=True, stable=True)[:3].tolist()


def coherence(Xfull, idx):
    """Mean pairwise cosine among chosen neighbours in the full centred space (common yardstick)."""
    Z = Xfull[idx]
    return float(((Z @ Z.T).sum() - 3) / 6)


def run(r, n_pool=N_POOL, seed=43, log=print):
    fam, words, groups = B.bank_words(r)
    pool = generic_pool(r, words, n_pool)
    band = r.band()
    sel = r.n_layers // 2
    log(f"{len(words)} bank words in {len(fam)} families; generic pool {len(pool)} words; "
        f"selection layer {sel}; clamp band {band}")
    # Keep only the rows the run uses (token embedding .. max(band end, selection layer)),
    # captured in chunks, so all 8000 pool states fit in RAM for 2048-wide models.
    keep = max(band[1], sel) + 2
    S, allw = {}, words + pool
    for s0 in range(0, len(allw), 500):
        for w, t in r.states(allw[s0:s0 + 500]).items():
            S[w] = t[:keep].clone()
        log(f"states {min(s0 + 500, len(allw))}/{len(allw)}")
    if str(r.device).startswith("cuda"):
        torch.cuda.empty_cache()
    Xb = torch.stack([S[w][sel + 1] for w in words])
    Xv = torch.stack([S[w][sel + 1] for w in pool])
    mu, proj = selection_spaces(Xv, Xb.shape[1])
    if r.centered:
        mu_c = (Xv.double() - Xv.double().mean(1, keepdim=True)).mean(0)
    else:
        mu_c = mu
    emb_b = {s: embed(Xb, mu_c, P, r.centered) for s, P in proj.items()}
    emb_v = {s: embed(Xv, mu_c, P, r.centered) for s, P in proj.items()}
    full_b, full_v = emb_b["full"], emb_v["full"]
    # Iris's exact neighbour space (bank-centred), for the reference row
    Xi = Xb.clone()
    if r.centered:
        Xi = Xi - Xi.mean(1, keepdim=True)
    Xi = Xi - Xi.mean(0)

    fam_keys = list(fam)
    fam_qs = [(k, C.family_q(k)) for k in fam_keys]
    allq = fam_qs + [(w, C.leaf_q(w)) for w in words]
    neutral = {}
    for s0 in range(0, len(allq), 24):
        neutral.update(margins_lean(r, C.SLOT, allq[s0:s0 + 24]))

    acc = {c: {"category": [], "identity": [], "coherence": [], "family_hits": []} for c in CONDITIONS}
    records = []
    for i, (w, g) in enumerate(zip(words, groups)):
        leaf_qs = [(x, C.leaf_q(x)) for x in fam[g]]
        leaf_keys = [x for x, _ in leaf_qs]
        rec = {"word": w, "family": g}
        for c in CONDITIONS:
            names, coh, hits = None, None, None
            if c == "native":
                targets = None
            else:
                if c == "iris":
                    idx = I.neighbors(Xi, i)
                    names = [words[j] for j in idx]
                    coh = coherence(full_b, idx)
                elif c == "random3_vocab":
                    idx = random.Random(seed + i).sample(range(len(pool)), 3)
                    names = [pool[j] for j in idx]
                    coh = coherence(full_v, idx)
                else:
                    space, which = c.rsplit("_", 1)
                    if which == "bank":
                        idx = top3(emb_b[space][i], emb_b[space].clone(), exclude=i)
                        names = [words[j] for j in idx]
                        coh = coherence(full_b, idx)
                    else:
                        idx = top3(emb_b[space][i], emb_v[space])
                        names = [pool[j] for j in idx]
                        coh = coherence(full_v, idx)
                if c == "iris" or c.endswith("_bank"):
                    hits = sum(groups[words.index(n)] == g for n in names) / 3
                targets = C.blend(S, [(0.5, S[w])] + [(0.5 / 3, S[n]) for n in names], band)
            cat = answer(r, w, fam_qs, fam_keys, neutral, targets)
            ide = answer(r, w, leaf_qs, leaf_keys, neutral, targets)
            acc[c]["category"].append(float(cat == g))
            acc[c]["identity"].append(float(ide == w))
            if coh is not None:
                acc[c]["coherence"].append(coh)
            if hits is not None:
                acc[c]["family_hits"].append(hits)
            rec[c] = {"answer": [cat, ide], "neighbors": names}
        records.append(rec)
        if (i + 1) % 10 == 0:
            log(f"{i + 1}/{len(words)}")

    def mean(v):
        return sum(v) / len(v) if v else None
    nat = acc["native"]["category"]
    summary = {}
    for c in CONDITIONS:
        a = acc[c]
        summary[c] = {"category": mean(a["category"]), "identity": mean(a["identity"]),
                      "coherence": mean(a["coherence"]), "family_hits": mean(a["family_hits"]),
                      "rescued": sum(1 for x, y in zip(nat, a["category"]) if y > x),
                      "harmed": sum(1 for x, y in zip(nat, a["category"]) if y < x)}
    p = acc[PRIMARY]["category"]
    s = summary
    ci_rand = B.paired_ci(p, acc["rand16_vocab"]["category"])
    o1 = bool(s[PRIMARY]["category"] >= s["native"]["category"] + 0.05
              and s[PRIMARY]["category"] >= s["rand16_vocab"]["category"] + 0.03
              and s[PRIMARY]["category"] >= s["random3_vocab"]["category"] + 0.03
              and ci_rand[1] > 0)
    ci_full = B.paired_ci(p, acc["full_vocab"]["category"])
    # Q1 (OCULAR.md amendment 1, preregistered before the Qwen run): inductive full-width repair
    fv = acc["full_vocab"]["category"]
    ci_fv_nat = B.paired_ci(fv, nat)
    q1 = bool(s["full_vocab"]["category"] >= s["native"]["category"] + 0.03
              and s["full_vocab"]["category"] >= s["random3_vocab"]["category"] + 0.03
              and ci_fv_nat[1] > 0)
    gates = {
        "O1_inductive_repair": {"pass": o1, "primary_minus_rand16_vocab_ci": ci_rand,
                                "primary_minus_native_ci": B.paired_ci(p, nat),
                                "primary_minus_random3_ci": B.paired_ci(p, acc["random3_vocab"]["category"])},
        "O2_band_beats_full_inductive": {"pass": bool(ci_full[1] > 0), "primary_minus_full_vocab_ci": ci_full},
        "O3_transductive": {"band4_bank_minus_iris_ci": B.paired_ci(acc["band4_bank"]["category"],
                                                                     acc["iris"]["category"])},
        "Q1_full_width_inductive": {"pass": q1, "full_vocab_minus_native_ci": ci_fv_nat,
                                    "full_vocab_minus_random3_ci": B.paired_ci(fv, acc["random3_vocab"]["category"]),
                                    "identity_change_ci": B.paired_ci(acc["full_vocab"]["identity"],
                                                                      acc["native"]["identity"])},
    }
    return dict(model=r.name, n_words=len(words), n_pool=len(pool), selection_layer=sel,
                band=list(band), families={k: len(v) for k, v in fam.items()},
                pool_sample=pool[:60], summary=summary, gates=gates, records=records)


def report(res):
    s, g = res["summary"], res["gates"]
    f = lambda v: "–" if v is None else f"{v:.3f}"
    L = [f"# Ocular report — {res['model']}", "",
         f"{res['n_words']} words, {len(res['families'])} families; generic pool {res['n_pool']} words; "
         f"selection layer {res['selection_layer']}; clamp band {res['band']}; dtype {res.get('dtype', 'float32')}. Rules: OCULAR.md.", "",
         f"**O1 inductive ocular repair: {'PASS' if g['O1_inductive_repair']['pass'] else 'FAIL'}**  ",
         "band4_vocab − rand16_vocab: {:+.3f} [{:+.3f}, {:+.3f}]  ".format(*g["O1_inductive_repair"]["primary_minus_rand16_vocab_ci"]),
         "band4_vocab − native: {:+.3f} [{:+.3f}, {:+.3f}]  ".format(*g["O1_inductive_repair"]["primary_minus_native_ci"]),
         "band4_vocab − random3_vocab: {:+.3f} [{:+.3f}, {:+.3f}]".format(*g["O1_inductive_repair"]["primary_minus_random3_ci"]),
         "",
         f"**O2 band beats full width at choosing (inductive): {'PASS' if g['O2_band_beats_full_inductive']['pass'] else 'FAIL'}**"
         "  band4_vocab − full_vocab: {:+.3f} [{:+.3f}, {:+.3f}]".format(*g["O2_band_beats_full_inductive"]["primary_minus_full_vocab_ci"]),
         "",
         "**O3 band4_bank − iris:** {:+.3f} [{:+.3f}, {:+.3f}]".format(*g["O3_transductive"]["band4_bank_minus_iris_ci"]),
         "",
         f"**Q1 full-width inductive repair (amendment 1): {'PASS' if g['Q1_full_width_inductive']['pass'] else 'FAIL'}**  ",
         "full_vocab − native: {:+.3f} [{:+.3f}, {:+.3f}]  ".format(*g["Q1_full_width_inductive"]["full_vocab_minus_native_ci"]),
         "full_vocab − random3_vocab: {:+.3f} [{:+.3f}, {:+.3f}]  ".format(*g["Q1_full_width_inductive"]["full_vocab_minus_random3_ci"]),
         "identity change: {:+.3f} [{:+.3f}, {:+.3f}]".format(*g["Q1_full_width_inductive"]["identity_change_ci"]),
         "", "| condition | category | identity | rescued | harmed | neighbour coherence | true-family neighbours |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    for c in CONDITIONS:
        r = s[c]
        L.append(f"| {c}{' (primary)' if c == PRIMARY else ''} | {f(r['category'])} | {f(r['identity'])} | "
                 f"{r['rescued']} | {r['harmed']} | {f(r['coherence'])} | {f(r['family_hits'])} |")
    L += ["", "Neighbours chosen for a few words (vocab pool):", "",
          "| word | full | pre16 | band4 | band8 | rand16 |", "|---|---|---|---|---|---|"]
    for rec in res["records"][::12]:
        L.append(f"| {rec['word']} | " + " | ".join(", ".join(rec[f'{sp}_vocab']['neighbors'])
                                                      for sp in SPACES) + " |")
    return "\n".join(L) + "\n"


def self_test():
    extra = [w for ws in I.BANK.values() for w in ws] + list(I.BANK) + \
            [f"pool{c}{d}" for c in "abcdefgh" for d in "abcdefgh"] + ["Pooltitle", "fox", "Oslo"]
    extra = [x for x in extra if x.isalpha()] + ["a", "an", "musical", "instrument"]
    r = C.load_stub("neox", extra=extra)
    fam, words, groups = B.bank_words(r)
    pool = generic_pool(r, words, 200)
    assert "Pooltitle" in pool and "fox" in pool and "Oslo" in pool, "title-case and 3-letter words must be allowed"
    assert not {w.lower() for w in pool} & {w.lower() for w in words}, "pool must exclude bank words"
    for stub in (r, C.load_stub("qwen", extra=extra)):
        _, fw, _ = B.bank_words(stub)
        Sx = stub.states(fw[:4])
        qs = [(k, C.family_q(k)) for k in list(I.BANK)[:5]]
        tg = C.blend(Sx, [(0.5, Sx[fw[0]]), (0.5, Sx[fw[1]])], stub.band())
        for t in (None, tg):
            a1, a2 = stub.margins(fw[0], qs, t), margins_lean(stub, fw[0], qs, t)
            assert max(abs(a1[k] - a2[k]) for k in a1) < 1e-4, "lean margins differ from Theory's"
    res = run(r, n_pool=80, log=lambda *_: None)
    assert set(res["summary"]) == set(CONDITIONS)
    # the iris row must use exactly Iris's neighbour function
    S = r.states(words)
    sel = r.n_layers // 2
    X = torch.stack([S[w][sel + 1] for w in words])
    if r.centered:
        X = X - X.mean(1, keepdim=True)
    X = X - X.mean(0)
    for i in (0, 5, 17):
        want = [words[j] for j in I.neighbors(X, i)]
        assert res["records"][i]["iris"]["neighbors"] == want, "iris row does not match Iris"
    for i, rec in enumerate(res["records"]):
        assert rec["word"] not in rec["band4_bank"]["neighbors"], "a word chose itself"
    for layer in r.layers:
        assert not layer._forward_hooks and not layer._forward_pre_hooks, "hook left behind"
    report(res)
    rq = C.load_stub("qwen", extra=extra)
    assert not rq.centered, "Qwen stub should use RMSNorm geometry"
    resq = run(rq, n_pool=80, log=lambda *_: None)
    assert "Q1_full_width_inductive" in resq["gates"]
    report(resq)
    print("Ocular self-test passed (random NeoX and Qwen stub models; plumbing only).")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="EleutherAI/pythia-410m")
    p.add_argument("--revision", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--frame", choices=C.FRAMES, default="shots")
    p.add_argument("--n-pool", type=int, default=N_POOL)
    p.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32",
                   help="float32 is the protocol; bfloat16 halves memory (a recorded deviation)")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--report-from", default=None, help="rewrite REPORT.md from a saved ocular.json")
    a = p.parse_args()
    if a.self_test:
        return self_test()
    if a.report_from:
        src = Path(a.report_from)
        res = json.loads(src.read_text(encoding="utf-8"))
        (src.parent / "REPORT.md").write_text(report(res), encoding="utf-8")
        return print(report(res))
    revision = a.revision or ("step143000" if "pythia" in a.model.lower() else None)
    print(f"loading {a.model} as {a.dtype} ...", flush=True)
    r = C.load(a.model, a.device, getattr(torch, a.dtype), revision=revision, frame=a.frame)
    print("loaded", flush=True)
    res = run(r, n_pool=a.n_pool)
    res["dtype"] = a.dtype
    out = Path("runs") / ("ocular_" + a.model.replace("/", "_") + "_" + a.frame +
                          ("_" + revision if revision else ""))
    out.mkdir(parents=True, exist_ok=True)
    (out / "ocular.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (out / "REPORT.md").write_text(report(res), encoding="utf-8")
    print(report(res))
    print("saved", out)


if __name__ == "__main__":
    main()
