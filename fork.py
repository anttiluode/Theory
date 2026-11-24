"""Fork: when Qwen reads two words, does it build the sibling average itself?

    python fork.py --self-test
    python fork.py                     (Qwen3-1.7B, chat frame: where the overshoot lives)
    python fork.py --frame shots

RakeisuusClaude and chord.py manufacture the average: they write a blended state into one
word position. Here nothing is clamped. The model reads "Items: oak, pine" in its normal
forward pass and its own attention does whatever pooling it does. Inputs of equal length:

    oak, oak     repeated identity
    pine, pine   another repeated identity
    oak, pine    shared category, different identities (and pine, oak for order)
    oak, salmon  different identities and categories

Questions, about the same input: "Are both of the items trees?" (category) and
"Is one of the items a pine?" (identity), for every member of the family.

1  Category overshoot at equal length: does "oak, pine" read as more tree than both
   "oak, oak" and "pine, pine"? This alone does NOT locate the bend (attention, the MLPs and
   later blocks could all do it); it says whether the model's own fork shows the effect.
2  Identity kept: in the same "oak, pine" context, is "pine" still fully available when asked?
   Retention = (id_pine(oak,pine) - id_pine(oak,oak)) / (id_pine(pine,pine) - id_pine(oak,oak)).
   The manufactured average lost identity; a query-dependent fork would not.
3  Query dependence: swap pine for another sibling (elm). Under the category question the
   answer should barely move; under "Is one of the items a pine?" it should move a lot.
4  Attention, descriptive only: from the answer position, the attention mass each head puts on
   the two item positions, and the share on the second item, under the category question and
   under the identity question about the second item. An even share is a balanced fork.
"""
import argparse
import itertools
import json
import statistics as st
import time
from pathlib import Path

import torch

import common as C

PLURAL = {"tree": "trees", "vegetable": "vegetables", "bird": "birds", "fish": "fish", "insect": "insects",
          "tool": "tools", "vehicle": "vehicles", "instrument": "musical instruments"}
CHAT2 = "Items: {a}, {b}\n\n{q} Answer yes or no."
SHOTS2 = ("Items: rose, tulip. Are both of them flowers? yes\n"
          "Items: lion, chair. Are both of them animals? no\n"
          "Items: cup, plate. Is one of them a cup? yes\n"
          "Items: wolf, wolf. Is one of them a tiger? no\n"
          "Items: lion, lion. Are both of them animals? yes\n"
          "Items: cup, lion. Is one of them a lion? yes\n")
SLOT = "thing"


def cat_q(frame, f):
    p = PLURAL.get(f, f + "s")
    return f"Are both of the items above {p}?" if frame == "chat" else f"Are both of them {p}?"


def id_q(frame, w):
    return (f"Is one of the items above {C.article(w)} {w}?" if frame == "chat"
            else f"Is one of them {C.article(w)} {w}?")


class Fork:
    def __init__(self, r):
        self.r, self.frame = r, r.frame
        self._cal = {}

    def prompt(self, a, b, q):
        """ids and the positions of a and b, or None if either is not exactly one token."""
        if self.frame == "chat":
            msgs = [{"role": "user", "content": CHAT2.format(a=a, b=b, q=q)}]
            try:
                text = self.r.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                      enable_thinking=False)
            except TypeError:
                text = self.r.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            text += C.CHAT_PREFIX
            start = text.rindex(f"Items: {a}, {b}\n")
        else:
            text = SHOTS2 + f"Items: {a}, {b}. {q}"
            start = text.rindex(f"Items: {a}, {b}.")
        sa = start + len("Items: ")
        sb = sa + len(a) + len(", ")
        enc = self.r.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        off = enc["offset_mapping"]

        def one(s, e):
            hit = [i for i, (x, y) in enumerate(off) if x < e and y > s]
            return hit[0] if len(hit) == 1 and off[hit[0]][1] <= e else None
        return enc["input_ids"], one(sa, sa + len(a)), one(sb, sb + len(b))

    @torch.no_grad()
    def margins(self, a, b, qs, attn=False):
        """{key: yes-minus-no} for (key, question) pairs about "Items: a, b"; with attn=True also
        {key: tensor [layers, heads, 2]}, the answer position's attention on a and on b."""
        r = self.r
        encs = [self.prompt(a, b, q) for _, q in qs]
        pa, pb = encs[0][1], encs[0][2]
        assert pa is not None and pb is not None and all(e[1] == pa and e[2] == pb for e in encs)
        n = max(len(e[0]) for e in encs)
        x = torch.full((len(encs), n), r.pad, dtype=torch.long)
        mask = torch.zeros((len(encs), n), dtype=torch.long)
        for i, e in enumerate(encs):
            x[i, :len(e[0])] = torch.tensor(e[0])
            mask[i, :len(e[0])] = 1
        last = mask.sum(1) - 1
        out = r.model(input_ids=x.to(r.device), attention_mask=mask.to(r.device), output_attentions=attn)
        rows = torch.arange(len(encs), device=out.logits.device)
        lp = torch.log_softmax(out.logits[rows, last.to(out.logits.device)].float(), -1)
        m = torch.logsumexp(lp[:, r.yes], -1) - torch.logsumexp(lp[:, r.no], -1)
        ev = {k: float(v) for (k, _), v in zip(qs, m.cpu())}
        if not attn:
            return ev
        att = getattr(out, "attentions", None)
        if not att or att[0] is None:
            return ev, None
        A = {}
        for i, (k, _) in enumerate(qs):
            li = int(last[i])
            A[k] = torch.stack([torch.stack([layer[i, :, li, pa], layer[i, :, li, pb]], -1).float().cpu()
                                for layer in att])
        return ev, A

    def calibrated(self, a, b, qs, attn=False):
        need = [(k, q) for k, q in qs if q not in self._cal]
        if need:
            for (k, q), v in zip(need, self.margins(SLOT, SLOT, need).values()):
                self._cal[q] = v
        got = self.margins(a, b, qs, attn)
        ev, A = got if attn else (got, None)
        ev = {k: v - self._cal[q] for (k, q), v in zip(qs, ev.values())}
        return (ev, A) if attn else ev


def cross_partner(tree, f, i):
    k = tree.fam[f][0]
    fams = [g for g in tree.fam if tree.fam[g][0] != k]
    g = fams[(list(tree.fam).index(f) + i) % len(fams)]
    return tree.fam[g][1][i % len(tree.fam[g][1])]


def run(r, out, log=print):
    t0 = time.time()
    fk = Fork(r)
    tree = C.Tree(r)
    # every leaf must also be one token in the pair frame
    fam = {}
    for f, (k, ws) in tree.fam.items():
        ok = [w for w in ws if None not in fk.prompt(w, w, cat_q(r.frame, f))[1:]]
        if len(ok) >= 3:
            fam[f] = ok
    log(f"{r.name} ({r.frame} frame): {sum(map(len, fam.values()))} leaves in {len(fam)} families")
    E, ATT = {}, {}
    for f, ws in fam.items():
        qs = [("cat", cat_q(r.frame, f))] + [(f"id:{y}", id_q(r.frame, y)) for y in ws]
        for i, a in enumerate(ws):
            partners = list(ws) + [cross_partner(tree, f, i)]
            for b in partners:
                if None in fk.prompt(a, b, qs[0][1])[1:]:
                    continue
                want_attn = b != a and b in ws
                got = fk.calibrated(a, b, qs, attn=want_attn)
                if want_attn:
                    E[(a, b)], A = got
                    if A is not None:
                        ATT[(a, b)] = dict(cat=A["cat"], id_b=A[f"id:{b}"])
                else:
                    E[(a, b)] = got
    # ---- 1 and 2: pairs
    rows = []
    for f, ws in fam.items():
        for a, b in itertools.combinations(ws, 2):
            aa, bb, ab, ba = E[(a, a)], E[(b, b)], E[(a, b)], E[(b, a)]
            mid = 0.5 * (ab["cat"] + ba["cat"])
            ret = lambda y, full, ctx, absent: (ctx[f"id:{y}"] - absent[f"id:{y}"]) / \
                (full[f"id:{y}"] - absent[f"id:{y}"]) if abs(full[f"id:{y}"] - absent[f"id:{y}"]) > 1e-6 else None
            rows.append(dict(family=f, a=a, b=b, cat_aa=aa["cat"], cat_bb=bb["cat"], cat_ab=ab["cat"],
                             cat_ba=ba["cat"], over_line=mid - 0.5 * (aa["cat"] + bb["cat"]),
                             over_both=mid - max(aa["cat"], bb["cat"]),
                             id_b_in_ab=ab[f"id:{b}"], id_b_in_bb=bb[f"id:{b}"], id_b_in_aa=aa[f"id:{b}"],
                             id_a_in_ab=ab[f"id:{a}"], id_a_in_aa=aa[f"id:{a}"], id_a_in_bb=bb[f"id:{a}"],
                             retention_b=ret(b, bb, ab, aa), retention_a=ret(a, aa, ab, bb),
                             order_gap=ab["cat"] - ba["cat"]))
    cross = []
    for f, ws in fam.items():
        for i, a in enumerate(ws):
            c = cross_partner(tree, f, i)
            if (a, c) in E:
                cross.append(dict(a=a, c=c, cat_ac=E[(a, c)]["cat"], cat_aa=E[(a, a)]["cat"],
                                  id_a_in_ac=E[(a, c)][f"id:{a}"]))
    rets = [x for r_ in rows for x in (r_["retention_a"], r_["retention_b"]) if x is not None]
    one = dict(n_pairs=len(rows), over_line=C.boot([x["over_line"] for x in rows]),
               over_both=sum(x["over_both"] > 0 for x in rows),
               order_gap=C.boot([abs(x["order_gap"]) for x in rows]),
               cross_below=sum(x["cat_ac"] < x["cat_aa"] for x in cross), n_cross=len(cross),
               id_yes_in_pair=sum((x["id_b_in_ab"] > 0) + (x["id_a_in_ab"] > 0) for x in rows),
               id_no_when_absent=sum((x["id_b_in_aa"] < 0) + (x["id_a_in_bb"] < 0) for x in rows),
               retention=dict(median=st.median(rets) if rets else None, n=len(rets),
                              boot=C.boot(rets)))
    # ---- 3: query dependence, swapping the second item for another sibling
    swaps = []
    for f, ws in fam.items():
        for a in ws:
            for b, b2 in itertools.permutations([w for w in ws if w != a], 2):
                swaps.append(dict(a=a, b=b, b2=b2, d_cat=E[(a, b)]["cat"] - E[(a, b2)]["cat"],
                                  d_id=E[(a, b)][f"id:{b}"] - E[(a, b2)][f"id:{b}"]))
    dcat = [abs(x["d_cat"]) for x in swaps]
    did = [abs(x["d_id"]) for x in swaps]
    dcross = [abs(E[(x["a"], x["c"])]["cat"] - st.mean(E[(x["a"], y)]["cat"] for y in fam[f] if y != x["a"]))
              for f in fam for x in cross if x["a"] in fam[f]]
    three = dict(n=len(swaps), median_abs_d_cat=st.median(dcat), median_abs_d_id=st.median(did),
                 median_abs_d_cat_cross=st.median(dcross) if dcross else None,
                 id_moves_more=sum(i > c for i, c in zip(did, dcat)))
    # ---- 4: attention
    four = None
    if ATT:
        L = next(iter(ATT.values()))["cat"].shape[0]
        per = {}
        for kind in ("cat", "id_b"):
            mass = torch.stack([v[kind].sum(-1) for v in ATT.values()])          # [pairs, L, H]
            share = torch.stack([v[kind][..., 1] / v[kind].sum(-1).clamp_min(1e-9) for v in ATT.values()])
            w = mass / mass.sum(-1, keepdim=True).clamp_min(1e-9)
            per[kind] = dict(mass=mass.mean((0, 2)).tolist(),                   # mean over heads
                             share_b=(w * share).sum(-1).mean(0).tolist())      # mass-weighted over heads
        four = dict(layers=L, per_layer=per)
    rep = dict(model=r.name, frame=r.frame, families=fam, pairs=rows, cross=cross, swaps=swaps,
               part1_2=one, part3=three, part4=four, seconds=round(time.time() - t0, 1))
    f2 = lambda b: f"{b['mean']:+.2f} [{b['lo']:+.2f}, {b['hi']:+.2f}]"
    log(f"\n1  category at equal length, n={one['n_pairs']} sibling pairs (both orders averaged)")
    log(f"   'a, b' above the line between 'a, a' and 'b, b': {f2(one['over_line'])} nats; "
        f"above both: {one['over_both']}/{one['n_pairs']}")
    log(f"   order |cat(a,b) - cat(b,a)|: {f2(one['order_gap'])}; 'a, other-kingdom' below 'a, a': "
        f"{one['cross_below']}/{one['n_cross']}")
    log(f"\n2  identity kept in the pair: says yes to a present item {one['id_yes_in_pair']}/{2 * one['n_pairs']}, "
        f"no to an absent one {one['id_no_when_absent']}/{2 * one['n_pairs']}")
    rb = one["retention"]["boot"]
    log(f"   retention (1 = as available as when repeated): median {one['retention']['median']:.2f}, "
        f"mean {f2(rb)}")
    log(f"\n3  swap the second sibling for another, n={three['n']}: median |change| under the category "
        f"question {three['median_abs_d_cat']:.2f}, under 'is one of them <the swapped one>' "
        f"{three['median_abs_d_id']:.2f}; identity moved more in {three['id_moves_more']}/{three['n']}")
    if three["median_abs_d_cat_cross"] is not None:
        log(f"   for scale: swapping in an other-kingdom item moves the category answer "
            f"{three['median_abs_d_cat_cross']:.2f}")
    if four:
        log("\n4  attention from the answer position (descriptive), every 4th layer")
        log("   layer | mass on items: cat  id | share on 2nd item: cat  id")
        for l in range(0, four["layers"], max(1, four["layers"] // 7)):
            p = four["per_layer"]
            log(f"   {l:5d} | {p['cat']['mass'][l]:14.3f} {p['id_b']['mass'][l]:4.3f} | "
                f"{p['cat']['share_b'][l]:17.2f} {p['id_b']['share_b'][l]:4.2f}")
    else:
        log("\n4  attention weights not available from this model build (load with eager attention)")
    out.mkdir(parents=True, exist_ok=True)
    (out / "fork.json").write_text(json.dumps(rep, indent=1, default=float))
    log(f"\nsaved {out / 'fork.json'} ({rep['seconds']} s)")
    return rep


def self_test():
    extra = [CHAT2, SHOTS2, "Items: , Are both of the items above them Is one tulip plate tiger",
             " ".join(PLURAL.values())]
    for arch, frame in (("qwen", "chat"), ("qwen", "shots"), ("neox", "shots")):
        r = C.load_stub(arch, frame=frame, extra=extra)
        fk = Fork(r)
        ids, pa, pb = fk.prompt("oak", "pine", cat_q(frame, "tree"))
        assert pa is not None and pb is not None and pb > pa
        rep = run(r, Path(f"runs/selftest_fork_{arch}_{frame}"), log=lambda *a: None)
        assert rep["part1_2"]["n_pairs"] > 0 and rep["part3"]["n"] > 0
        assert rep["part4"] is not None, "attention weights missing"
        for v in rep["part4"]["per_layer"]["cat"]["share_b"]:
            assert 0 <= v <= 1
    print("self-test passed (stub Qwen2 and GPT-NeoX, both frames); numbers from random models mean nothing")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--frame", choices=C.FRAMES, default="chat")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    r = C.load(a.model, a.device, torch.float32, a.revision, a.frame, attn="eager")
    name = r.name.replace("/", "_").replace("@", "_") + ("_chat" if a.frame == "chat" else "")
    run(r, Path("runs") / name)


if __name__ == "__main__":
    main()
