"""Head experiment 003: three cues kept apart. Remote associates by intersecting Qwen's own
association distributions, one per cue, with a base-rate correction. No loop, no states.

Run: python head3.py --self-test
     python head3.py --device cuda --norms strength.SWOW-EN.R123.20180827.csv
Protocol and frozen gates: HEAD3.md.
"""
import argparse
import json
import random
import re
from pathlib import Path

import torch

import common as C
import head as H0

FRAME = ("Item: lemon. Association: sour\nItem: needle. Association: thread\n"
         "Item: doctor. Association: nurse\n")
SHOT_WORDS = {"lemon", "sour", "needle", "thread", "doctor", "nurse"}
N_BASE, SEED_BASE = 200, 7
LAMBDAS = {"prod_raw": 0.0, "prod_l1": 1.0, "prod_pmi": 2.0}
CONDS = ["native", "prod_raw", "prod_l1", "prod_pmi", "mix"]


class Voice:
    """The few pieces of head.Head that greedy one-word answers need (no pool, no states)."""

    def __init__(self, r):
        self.r, self.passes = r, 0
        tok = r.tok
        V = r.model.get_output_embeddings().weight.shape[0]
        n_tok = len(tok) if hasattr(tok, "__len__") else len(tok.vocab)
        if isinstance(tok, C.StubTokenizer):
            text = [tok.decode([i]) for i in range(n_tok)]
        else:
            text = tok.batch_decode([[i] for i in range(n_tok)])
        self.text = text + [""] * (V - n_tok)
        self.start_mask = torch.tensor([bool(re.fullmatch(r" [A-Za-z]+", t)) for t in self.text])
        self.cont_mask = torch.tensor([bool(re.fullmatch(r"[a-z]+", t)) for t in self.text])

    ids = H0.Head.ids
    ban_ids = H0.Head.ban_ids
    say = H0.Head.say

    @torch.no_grad()
    def next_logprobs(self, text):
        r = self.r
        h = r.model.base_model(input_ids=torch.tensor([self.ids(text)], device=r.device)).last_hidden_state[0, -1]
        self.passes += 1
        return torch.log_softmax(r.model.get_output_embeddings()(h).float().cpu(), -1)


def excluded(word, cues):
    for c in cues:
        c = c.lower()
        if word == c or word == c + "s" or c == word + "s" or word.startswith(c) or c.startswith(word):
            return True
    return False


def scores(P, logp0, cond):
    """P: [3, n] log p(w | cue_j), renormalised over candidates. Returns a score per candidate."""
    if cond == "mix":
        return torch.logsumexp(P, 0) - logp0
    return P.sum(0) - LAMBDAS[cond] * logp0


def rank_of(score, j):
    return int((score > score[j]).sum()) + 1


def spearman(a, b):
    def rk(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            for t in range(i, j + 1):
                r[o[t]] = (i + j) / 2
            i = j + 1
        return r
    x, y = rk(a), rk(b)
    mx, my = sum(x) / len(x), sum(y) / len(y)
    num = sum((p - mx) * (q - my) for p, q in zip(x, y))
    den = (sum((p - mx) ** 2 for p in x) * sum((q - my) ** 2 for q in y)) ** 0.5
    return num / den if den else None


def run(r, norms, norms_name, items, human, log=print):
    v = Voice(r)
    vocab_words = set(norms) | {w for d in norms.values() for w in d}
    cand_ids, cand_words = [], []
    for i, t in enumerate(v.text):
        if re.fullmatch(r" [a-z]{3,}", t) and t[1:] in vocab_words:
            cand_ids.append(i)
            cand_words.append(t[1:])
    cid = torch.tensor(cand_ids)
    index = {w: j for j, w in enumerate(cand_words)}
    log(f"{len(cand_words)} candidate words")

    def dist(word):
        lp = v.next_logprobs(FRAME + f"Item: {word}. Association:")[cid]
        return lp - torch.logsumexp(lp, 0)

    item_words = {w for it in items for x in it for w in x.split("/")}
    base_pool = sorted(w for w in norms if re.fullmatch(r"[a-z]{3,}", w) and w in index
                       and w not in item_words and w not in SHOT_WORDS)
    base_words = random.Random(SEED_BASE).sample(base_pool, min(N_BASE, len(base_pool)))
    acc = None
    for w in base_words:
        p = dist(w).exp()
        acc = p if acc is None else acc + p
    logp0 = (acc / len(base_words)).log()
    log(f"base rate from {len(base_words)} words")
    recs = []
    for n, (a, b, c, sol) in enumerate(items):
        cues = [a, b, c]
        P = torch.stack([dist(x) for x in cues])
        mask = torch.tensor([excluded(w, cues) for w in cand_words])
        sols = [s for s in sol.split("/") if s in index]
        rec = dict(cues=cues, solution=sol, candidate=bool(sols))
        for cond in CONDS[1:]:
            sc = scores(P, logp0, cond).masked_fill(mask, -float("inf"))
            top = torch.topk(sc, 10).indices.tolist()
            ans = cand_words[top[0]]
            rk = min(rank_of(sc, index[s]) for s in sols) if sols else None
            rec[cond] = dict(answer=ans, correct=H0.correct(ans, sol), top10=[cand_words[j] for j in top],
                             hit10=any(H0.correct(cand_words[j], sol) for j in top), rank=rk)
        if sols:
            rec["cue_ranks"] = [min(rank_of(P[j].masked_fill(mask, -float("inf")), index[s]) for s in sols)
                                for j in range(3)]
        word, k = v.say(v.ids(H0.rat_prompt(a, b, c)), ban=v.ban_ids(cues))
        rec["native"] = dict(answer=word, correct=H0.correct(word, sol), passes=k)
        rec["human_15s"] = human.get((a, b, c))
        recs.append(rec)
        if (n + 1) % 20 == 0:
            log(f"items {n + 1}/{len(items)}")
    res = dict(model=r.name, norms=norms_name, n_candidates=len(cand_words), base_words=base_words,
               records=recs, passes=v.passes)
    res["gates"] = gates(res)
    return res


def gates(res):
    recs = res["records"]
    acc = {c: [float(x[c]["correct"]) for x in recs] for c in CONDS}
    paired = lambda a, b: H0.B.paired_ci(acc[a], acc[b])
    g1, g2 = paired("prod_pmi", "native"), paired("prod_pmi", "mix")
    with_rank = [x for x in recs if x["prod_pmi"]["rank"] is not None and x["human_15s"] is not None]
    rho = spearman([x["human_15s"] for x in with_rank], [-x["prod_pmi"]["rank"] for x in with_rank]) if len(with_rank) > 2 else None
    return dict(G1_intersection_beats_prompt=dict(pass_=bool(g1[1] > 0), prod_pmi_minus_native=g1),
                G2_intersection_beats_union=dict(pass_=bool(g2[1] > 0), prod_pmi_minus_mix=g2),
                human_vs_model_spearman=rho, n_ranked=len(with_rank))


def report(res):
    recs, g = res["records"], res["gates"]
    n = len(recs)
    P = lambda k: "PASS" if g[k]["pass_"] else "FAIL"
    ci = lambda v: "{:+.3f} [{:+.3f}, {:+.3f}]".format(*v)
    med = lambda v: sorted(v)[len(v) // 2] if v else None
    cand = sum(x["candidate"] for x in recs)
    L = [f"# Head experiment 003 report — {res['model']}", "",
         f"{n} items; {res['n_candidates']} candidate words (single tokens in the SWOW vocabulary); solution is a "
         f"candidate for {cand}/{n}; base rate from {len(res['base_words'])} words; {res['passes']} forward passes. "
         f"Norms {res['norms']}. Rules: HEAD3.md.", "",
         f"**G1 intersection beats the prompt: {P('G1_intersection_beats_prompt')}**  prod_pmi − native "
         f"{ci(g['G1_intersection_beats_prompt']['prod_pmi_minus_native'])}", "",
         f"**G2 intersection beats union: {P('G2_intersection_beats_union')}**  prod_pmi − mix "
         f"{ci(g['G2_intersection_beats_union']['prod_pmi_minus_mix'])}", "",
         "| condition | correct | accuracy | solution in top 10 | median rank of solution (when a candidate) |",
         "|---|---:|---:|---:|---:|"]
    for c in CONDS:
        k = sum(x[c]["correct"] for x in recs)
        if c == "native":
            L.append(f"| native | {k}/{n} | {k / n:.3f} | – | – |")
        else:
            h = sum(x[c]["hit10"] for x in recs)
            L.append(f"| {c} | {k}/{n} | {k / n:.3f} | {h}/{n} | {med([x[c]['rank'] for x in recs if x[c]['rank']])} |")
    cr = [x["cue_ranks"] for x in recs if "cue_ranks" in x]
    if cr:
        best = sorted(min(v) for v in cr)
        worst = sorted(max(v) for v in cr)
        L += ["", f"Per cue (λ = 0, one cue alone), rank of the solution: median of the best cue {med(best)}, "
              f"of the worst cue {med(worst)}; solution in the best cue's top 10 for "
              f"{sum(b <= 10 for b in best)}/{len(cr)} items, in all three cues' top 100 for "
              f"{sum(w <= 100 for w in worst)}/{len(cr)}."]
    rho = g["human_vs_model_spearman"]
    L += ["", f"Human 15-second solve rate vs the solution's rank under prod_pmi: Spearman "
          f"{'–' if rho is None else f'{rho:+.3f}'} over {g['n_ranked']} items (positive = the model finds easy "
          "what people find easy).", "",
          "| cues | solution | native | prod_raw | prod_pmi | mix | prod_pmi top 10 |", "|---|---|---|---|---|---|---|"]
    b = lambda x, c: f"**{x[c]['answer']}**" if x[c]["correct"] else x[c]["answer"]
    for x in recs:
        L.append(f"| {', '.join(x['cues'])} | {x['solution']} | {b(x, 'native')} | {b(x, 'prod_raw')} | "
                 f"{b(x, 'prod_pmi')} | {b(x, 'mix')} | {', '.join(x['prod_pmi']['top10'])} |")
    return "\n".join(L) + "\n"


def load_human(path):
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines()[1:]:
        p = line.split("\t")
        if len(p) >= 5:
            try:
                out[tuple(x.strip().lower() for x in p[:3])] = float(p[4])
            except ValueError:
                pass
    return out


def self_test():
    # the scoring math
    P = torch.log(torch.tensor([[0.5, 0.3, 0.2], [0.1, 0.6, 0.3], [0.2, 0.5, 0.3]]))
    logp0 = torch.log(torch.tensor([0.6, 0.2, 0.2]))
    s = scores(P, logp0, "prod_pmi")
    want = P.sum(0) - 2 * logp0
    assert torch.allclose(s, want) and int(s.argmax()) == 1
    assert int(scores(P, logp0, "prod_raw").argmax()) == 1
    m = scores(P, logp0, "mix")
    assert torch.allclose(m, torch.log(P.exp().sum(0)) - logp0)
    assert rank_of(torch.tensor([3.0, 1.0, 2.0]), 2) == 2
    assert excluded("skates", ["skate"]) and excluded("skateboard", ["skate"]) and excluded("cot", ["cottage"])
    assert not excluded("ice", ["cream", "skate", "water"])
    assert abs(spearman([1, 2, 3], [1, 2, 3]) - 1) < 1e-9
    rat = H0.load_rat(Path(__file__).with_name("data") / "rat144.tsv")
    assert len(rat) == 144
    words = {w for it in rat for x in it for w in x.split("/")}
    assert not SHOT_WORDS & words, "few-shot words overlap the items"
    assert len(load_human(Path(__file__).with_name("data") / "rat144.tsv")) == 144
    # the run, on a stub
    stub = ("cream skate water ice cottage swiss cake cheese oak tree wood river sea bread butter jam iron "
            "steel music song winter snow")
    texts = [FRAME, H0.RAT_INTRO, H0.rat_prompt("x", "y", "z"), stub] + [f"base{c}{d}" for c in "abcdef" for d in "abcdef"]
    r = C.load_stub("qwen", extra=texts)
    sw = stub.split() + [f"base{c}{d}" for c in "abcdef" for d in "abcdef"]
    norms = {w: {"tree": 0.1} for w in sw}
    items = [("cream", "skate", "water", "ice"), ("cottage", "swiss", "cake", "cheese"), ("oak", "river", "iron", "zzz")]
    res = run(r, norms, "selftest", items, {}, log=lambda *_: None)
    assert len(res["records"]) == 3 and res["passes"] >= 3 * 3
    for x in res["records"]:
        for c in CONDS[1:]:
            assert not excluded(x[c]["answer"], x["cues"]), "an excluded candidate won"
    assert res["records"][2]["prod_pmi"]["rank"] is None
    json.dumps(res)
    assert "G1 intersection" in report(res)
    print("Head3 self-test passed (scoring math; random Qwen stub for plumbing).")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    p.add_argument("--norms", default=None)
    p.add_argument("--rat", default=str(Path(__file__).with_name("data") / "rat144.tsv"))
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
        raise SystemExit("--norms is required (HEAD3.md)")
    norms = H0.load_norms(a.norms)
    items = H0.load_rat(a.rat)
    print(f"loading {a.model} as {a.dtype} ...", flush=True)
    r = C.load(a.model, a.device, getattr(torch, a.dtype), frame="shots")
    print("loaded", flush=True)
    res = run(r, norms, Path(a.norms).name, items, load_human(a.rat), log=lambda *x: print(*x, flush=True))
    res["dtype"] = a.dtype
    out = Path("runs") / ("head3_" + a.model.replace("/", "_"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "head3.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (out / "REPORT.md").write_text(report(res), encoding="utf-8")
    print(report(res))
    print("saved", out)


if __name__ == "__main__":
    main()
