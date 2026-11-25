"""Head: a thought held as real middle-layer states of a frozen model, moving by itself.

Run: python head.py --self-test
     python head.py --device cuda --norms path/to/strength.SWOW-EN.R123.csv
Protocol and frozen gates: HEAD.md. No training. Reuses Theory's clamp (Reader._hooks), blend,
band and Ocular's generic pool; the loop, the voice and the two tests are new.
"""
import argparse
import json
import math
import random
import re
from pathlib import Path

import torch

import bridge as B
import common as C
import ocular as O

# ---------------------------------------------------------------- frozen constants (HEAD.md)
NAME_SHOTS = ("Item: rose. One word: rose\nItem: lion. One word: lion\n"
              "Item: chair. One word: chair\nItem: cup. One word: cup\n")
NAME_Q = "One word:"
# Amendment 1 (HEAD.md): the frame the closed-loop engine thinks in. States are captured in it
# separately, so a clamped thought is always made of real states of this very frame.
ASSOC_FRAME = ("Item: salt. Association: pepper\nItem: needle. Association: thread\n"
               "Item: king. Association: queen\n")
ASSOC_Q = "Association:"
FRAMES_H = {"name": (NAME_SHOTS, NAME_Q), "assoc": (ASSOC_FRAME, ASSOC_Q)}
ASSOC_SHOTS = ("Word: salt. Association: pepper\nWord: needle. Association: thread\n"
               "Word: king. Association: queen\n")
RAT_INTRO = ("Each puzzle gives three words. Find one word that forms a compound word or a common "
             "phrase with each of them.\n")
RAT_SHOT_ITEMS = [("skunk", "kings", "boiled", "cabbage",
                   "skunk cabbage, cabbages and kings, boiled cabbage."),
                  ("widow", "bite", "monkey", "spider",
                   "black widow spider, spider bite, spider monkey."),
                  ("elephant", "lapse", "vivid", "memory",
                   "elephant memory, memory lapse, vivid memory.")]
N_POOL = 8000
T = 12
ALPHA, F, BETA, RHO, K0, KMAX, PRUNE, MAXW = 0.5, 6, 0.3, 0.2, 4, 32, 0.02, 48
BETA_L = 2.0      # amendment 1: residue tilt for the loop engine, nats per unit cosine
N_CAL, N_STARTS, N_P0 = 20, 60, 200
SEED_CAL, SEED_START, SEED_P0, SEED_RAND = 7, 11, 5, 13
DEMO = ["oak", "river", "bread", "winter", "iron", "music", "sauna"]
POLICIES = {
    "head": dict(engine="static", adapt=True, k0=K0, alpha=ALPHA, fat=F, beta=BETA, rho=RHO),
    "fixed": dict(engine="static", adapt=False, k0=K0, alpha=ALPHA, fat=F, beta=BETA, rho=RHO),
    "nn": dict(engine="static", adapt=False, k0=1, alpha=1.0, fat=0, beta=0.0, rho=0.0),
    # amendment 1: Qwen's own upper layers choose the next blend
    "loop": dict(engine="loop", adapt=True, k0=K0, alpha=ALPHA, fat=F, beta=BETA_L, rho=RHO),
    "loop_fixed": dict(engine="loop", adapt=False, k0=K0, alpha=ALPHA, fat=F, beta=BETA_L, rho=RHO),
    "loop_nores": dict(engine="loop", adapt=True, k0=K0, alpha=ALPHA, fat=F, beta=0.0, rho=0.0),
}
FREE_CONDS = ["head", "fixed", "nn", "loop", "loop_fixed", "loop_nores", "random", "native"]
GOAL_WALKS = [("head", "head"), ("walk_fixed", "fixed"), ("loop", "loop"), ("loop_fixed", "loop_fixed"),
              ("loop_nores", "loop_nores")]
GOAL_CONDS = ["head", "walk_fixed", "loop", "loop_fixed", "loop_nores", "fixed_avg", "nearest_min",
              "native", "reason", "reason_loop"]


def engine_band(bands, policy):
    return bands["loop" if POLICIES[policy]["engine"] == "loop" else "static"]


class _Stop(Exception):
    pass


def rat_prompt(a, b, c, think=False):
    lines = [RAT_INTRO]
    for x, y, z, s, why in RAT_SHOT_ITEMS:
        lines.append(f"Words: {x}, {y}, {z}.\n" + (f"Thinking: {why}\n" if think else "") + f"Answer: {s}\n")
    lines.append(f"Words: {a}, {b}, {c}.\n" + ("Thinking:" if think else "Answer:"))
    return "".join(lines)


def norm_word(w):
    return re.sub(r"[^a-z]", "", w.lower())


def correct(ans, solution):
    a = norm_word(ans)
    if not a:
        return False
    for s in solution.split("/"):
        s = norm_word(s)
        if a in (s, s + "s", s + "es") or s in (a + "s", a + "es"):
            return True
    return False


def load_rat(path):
    items = []
    for line in Path(path).read_text(encoding="utf-8").splitlines()[1:]:
        p = line.rstrip("\n").split("\t")
        if len(p) >= 4:
            items.append(tuple(x.strip().lower() for x in p[:4]))
    return items


def load_norms(path):
    """cue -> {response: strength}. Reads SWOW strength files (cue, response, R123.Strength),
    USF Cue_Target_Pairs (CUE, TARGET, FSG) or any table with cue/response(or target)/strength."""
    text = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    head = text[0]
    sep = "\t" if "\t" in head else ","
    cols = [h.strip().strip('"').lower() for h in head.split(sep)]

    def find(names):
        for n in names:
            for i, c in enumerate(cols):
                if c == n:
                    return i
        return None
    ci = find(["cue"])
    ri = find(["response", "target"])
    si = find(["r123.strength", "r1.strength", "fsg", "strength", "weight"])
    if ci is None or ri is None or si is None:
        raise SystemExit(f"cannot find cue/response/strength columns in {path}: {cols[:8]}")
    out = {}
    for line in text[1:]:
        p = [x.strip().strip('"') for x in line.split(sep)]
        if len(p) <= max(ci, ri, si):
            continue
        try:
            s = float(p[si])
        except ValueError:
            continue
        a, b = p[ci].lower(), p[ri].lower()
        if a and b and s > 0:
            out.setdefault(a, {})[b] = max(s, out.get(a, {}).get(b, 0.0))
    return out


# ---------------------------------------------------------------- the frame, the memory, the voice
class Head:
    def __init__(self, r, pool, log=print):
        self.r, self.log = r, log
        self.band = r.band()
        self.sel = r.n_layers // 2
        self.keep = max(self.band[1], self.sel) + 2
        tok = r.tok
        V = r.model.get_output_embeddings().weight.shape[0]
        n_tok = len(tok) if hasattr(tok, "__len__") else len(tok.vocab)
        if hasattr(tok, "batch_decode") and not isinstance(tok, C.StubTokenizer):
            text = tok.batch_decode([[i] for i in range(n_tok)])
        else:
            text = [tok.decode([i]) for i in range(n_tok)]
        self.text = text + [""] * (V - n_tok)
        self.start_mask = torch.tensor([bool(re.fullmatch(r" [A-Za-z]+", t)) for t in self.text])
        self.cont_mask = torch.tensor([bool(re.fullmatch(r"[a-z]+", t)) for t in self.text])
        self.S, self.SA = {}, {}      # naming-frame states, association-frame states
        pool = [w for w in pool if self.name_prompt(w)[1] is not None
                and self.name_prompt(w, "assoc")[1] is not None]
        self.capture(pool)
        self.capture(pool, "assoc")
        self.pool = [w for w in pool if w in self.S and w in self.SA]
        self.pool_tok = torch.tensor([self.name_prompt(w)[0][self.name_prompt(w)[1]] for w in self.pool])
        self.index = {w: i for i, w in enumerate(self.pool)}
        self.lower_index = {}
        for i, w in enumerate(self.pool):
            self.lower_index.setdefault(w.lower(), i)
        X = torch.stack([self.S[w][self.sel + 1] for w in self.pool]).double()
        if r.centered:
            X = X - X.mean(1, keepdim=True)
        self.mu = X.mean(0)
        self.Xn = O.C_unit(X - self.mu).float()
        self.passes = 0

    # ---- prompts
    def name_prompt(self, word, frame="name"):
        shots, q = FRAMES_H[frame]
        text = shots + f"Item: {word}. {q}"
        ws = text.rindex("Item: " + word + ".") + len("Item: ")
        we = ws + len(word)
        enc = self.r.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        hit = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if a < we and b > ws]
        ok = len(hit) == 1 and enc["offset_mapping"][hit[0]][1] <= we
        return enc["input_ids"], (hit[0] if ok else None)

    def ids(self, text):
        return self.r.tok(text, add_special_tokens=False)["input_ids"]

    # ---- states, captured in the naming frame, batched, stopped after the last needed layer
    @torch.no_grad()
    def capture(self, words, frame="name", bs=32):
        dest = self.S if frame == "name" else self.SA
        r, need = self.r, [w for w in words if w not in dest]
        groups = {}
        for w in need:
            ids, pos = self.name_prompt(w, frame)
            if pos is not None:
                groups.setdefault((len(ids), pos), []).append((w, ids))
        last = self.keep - 2
        done = 0
        for (n, pos), items in groups.items():
            for s0 in range(0, len(items), bs):
                chunk = items[s0:s0 + bs]
                store, hooks = {}, []

                def pre(mod, args, kwargs):
                    h = args[0] if args else kwargs["hidden_states"]
                    store[-1] = h[:, pos].detach().float().cpu()
                hooks.append(r.layers[0].register_forward_pre_hook(pre, with_kwargs=True))
                for l in range(last + 1):
                    def post(m, i, o, l=l):
                        store[l] = C._hs(o)[:, pos].detach().float().cpu()
                        if l == last:
                            raise _Stop
                    hooks.append(r.layers[l].register_forward_hook(post))
                try:
                    r.model(input_ids=torch.tensor([x for _, x in chunk], device=r.device))
                except _Stop:
                    pass
                finally:
                    for h in hooks:
                        h.remove()
                for b, (w, _) in enumerate(chunk):
                    dest[w] = torch.stack([store[l][b] for l in range(-1, last + 1)])
                done += len(chunk)
                if done % 1000 < bs:
                    self.log(f"states ({frame} frame) {done}/{len(need)}")
        if str(r.device).startswith("cuda"):
            torch.cuda.empty_cache()

    def has_state(self, w):
        if w in self.S and w in self.SA:
            return True
        if self.name_prompt(w)[1] is None or self.name_prompt(w, "assoc")[1] is None:
            return False
        self.capture([w])
        self.capture([w], "assoc")
        return w in self.S and w in self.SA

    # ---- geometry
    def vec(self, thought):
        raw = sum(c * self.S[w][self.sel + 1].double() for w, c in thought.items())
        if self.r.centered:
            raw = raw - raw.mean()
        return O.C_unit((raw - self.mu).unsqueeze(0))[0].float()

    def targets(self, thought, frame="name"):
        S = self.S if frame == "name" else self.SA
        assert thought, "empty thought"
        return C.blend(S, [(c, S[w]) for w, c in thought.items()], self.band)

    @torch.no_grad()
    def associate(self, thought):
        """Amendment 1: clamp the thought into `Item: thing. Association:` and let Qwen's upper
        layers run. Returns log-probabilities over the pool words (renormalised over the pool)."""
        r = self.r
        ids, pos = self.name_prompt(C.SLOT, "assoc")
        hooks = r._hooks(pos, self.targets(thought, "assoc"))
        try:
            h = r.model.base_model(input_ids=torch.tensor([ids], device=r.device)).last_hidden_state[0, -1]
        finally:
            for hk in hooks:
                hk.remove()
        self.passes += 1
        lg = r.model.get_output_embeddings()(h).float().cpu()
        return torch.log_softmax(lg[self.pool_tok], -1)

    # ---- the voice
    def ban_ids(self, words):
        out = set()
        for w in words:
            for form in (w, w.capitalize(), w.upper()):
                ids = self.ids(" " + form)
                if ids:
                    out.add(ids[0])
        return sorted(out)

    @torch.no_grad()
    def say(self, ids, pos=None, targets=None, ban=(), max_tok=3):
        """Greedy one word. Returns (word, forward passes)."""
        r, ids, pieces, n = self.r, list(ids), [], 0
        for t in range(max_tok):
            hooks = r._hooks(pos, targets) if targets else []
            try:
                h = r.model.base_model(input_ids=torch.tensor([ids], device=r.device)).last_hidden_state[0, -1]
            finally:
                for hk in hooks:
                    hk.remove()
            n += 1
            lg = r.model.get_output_embeddings()(h).float().cpu()
            if t == 0:
                lg = lg.masked_fill(~self.start_mask, -float("inf"))
                if ban:
                    lg[list(ban)] = -float("inf")
            j = int(lg.argmax())
            if t > 0 and not self.cont_mask[j]:
                break
            pieces.append(self.text[j])
            ids.append(j)
        self.passes += n
        return "".join(pieces).strip().lower(), n

    def speak(self, thought, ban=()):
        ids, pos = self.name_prompt(C.SLOT)
        return self.say(ids, pos, self.targets(thought), ban)

    @torch.no_grad()
    def generate(self, ids, budget):
        """Greedy continuation, stopping at a newline, up to `budget` forward passes. Each token
        is a full forward pass (no KV cache: simple and identical across transformers versions;
        the count is the same either way). Returns (generated ids, passes)."""
        r, ids, gen, n = self.r, list(ids), [], 0
        while n < budget:
            h = r.model.base_model(input_ids=torch.tensor([ids], device=r.device)).last_hidden_state[0, -1]
            j = int(r.model.get_output_embeddings()(h).float().argmax())
            n += 1
            gen.append(j)
            ids.append(j)
            if "\n" in self.text[j]:
                break
        self.passes += n
        return gen, n

    # ---- the loop
    def walk(self, start, policy, band=None, steps=T, exclude=(), ban=(), anchor_fixed=False,
             speak=True):
        p = POLICIES[policy] if isinstance(policy, str) else policy
        thought = dict(start)
        k = p["k0"]
        fatigue = {}
        excl = {self.index[w] for w in exclude if w in self.index}
        q_prev = self.vec(thought)
        resid = q_prev.clone()
        rho = 0.0 if anchor_fixed else p["rho"]
        trail, ks, surprises, passes = [], [], [], 0
        loop = p.get("engine") == "loop"
        for t in range(steps):
            if loop:                     # Qwen's upper layers propose; amendment 1
                lp = self.associate(thought)
                passes += 1
                sim = lp.clone()
            else:                        # the static map proposes
                sim = self.Xn @ q_prev
            if p["beta"]:
                sim = sim + p["beta"] * (self.Xn @ O.C_unit(resid.unsqueeze(0))[0])
            base = sim
            # mask fatigued words and the thought's own words; if nothing is left (tiny pools only),
            # fatigue is lifted first, then the thought's own words
            for layers in ((True, True), (False, True), (False, False)):
                mask = set(excl)
                if layers[0]:
                    mask |= {i for i, e in fatigue.items() if e > t}
                if layers[1]:
                    mask |= {self.index[w] for w in thought if w in self.index}
                sim = base.clone()
                if mask:
                    sim[list(mask)] = -float("inf")
                if torch.isfinite(sim).any():
                    break
            kk = min(k, int(torch.isfinite(sim).sum()))
            top = torch.topk(sim, kk).indices.tolist()
            if loop:                     # Qwen's own probabilities shape the blend
                tw = torch.softmax(lp[top], -1).tolist()
            else:
                tw = [1.0 / kk] * kk
            # carry-over below PRUNE is dropped; this step's targets are always kept (fix, HEAD.md
            # amendment 1: at k = 32, alpha / k = 0.016 < PRUNE emptied the thought)
            new = {w: (1 - p["alpha"]) * c for w, c in thought.items() if (1 - p["alpha"]) * c >= PRUNE}
            tgt = {}
            for j, wt in zip(top, tw):
                tgt[self.pool[j]] = tgt.get(self.pool[j], 0.0) + p["alpha"] * wt
            for w, c in tgt.items():
                new[w] = new.get(w, 0.0) + c
            rest = sorted((w for w in new if w not in tgt), key=lambda w: -new[w])
            keep = set(tgt) | set(rest[:max(0, MAXW - len(tgt))])
            z = sum(new[w] for w in keep)
            thought = {w: new[w] / z for w in keep}
            if p["fat"]:
                for j in top:
                    fatigue[j] = t + 1 + p["fat"]
            q = self.vec(thought)
            s = float(1 - q @ q_prev)
            surprises.append(s)
            ks.append(k)
            word = None
            if speak:
                word, n = self.speak(thought, ban)
                passes += n
                trail.append(word)
            if p["adapt"] and band is not None:
                stuck = speak and len(trail) > 1 and trail[-1] == trail[-2]
                if s < band[0] or stuck:
                    k = min(KMAX, 2 * k)
                elif s > band[1]:
                    k = max(1, k // 2)
            if rho:
                resid = (1 - rho) * resid + rho * q
            q_prev = q
        return dict(trail=trail, k=ks, surprise=surprises, passes=passes)


# ---------------------------------------------------------------- tests
def p0(H, n=None):
    n = N_P0 if n is None else n
    rng = random.Random(SEED_P0)
    words = rng.sample(H.pool, min(n, len(H.pool)))
    copy = blend = 0
    for w in words:
        ids, _ = H.name_prompt(w)
        copy += H.say(ids)[0] == w.lower()
        o = rng.choice(H.pool)
        while o == w:
            o = rng.choice(H.pool)
        said = H.speak({w: 0.5, o: 0.5})[0]
        blend += said in (w.lower(), o.lower())
    return dict(copy=copy / len(words), blend=blend / len(words), n=len(words),
                passed=bool(copy / len(words) >= 0.9 and blend / len(words) >= 0.5))


def calibrate(H, starts, policy="fixed"):
    s = []
    for w in starts:
        s += H.walk({w: 1.0}, policy, speak=False)["surprise"]
    s = sorted(s)
    q = lambda f: s[min(len(s) - 1, int(f * len(s)))]
    return [q(0.25), q(0.75)]


def trail_stats(trail, start, norms):
    seq = [start.lower()] + trail
    hits, strength, valid = 0, 0.0, 0
    if norms is not None:
        for a, b in zip(seq, seq[1:]):
            if a in norms:
                valid += 1
                hits += b in norms[a]
                strength += norms[a].get(b, 0.0)
    counts = {}
    for w in trail:
        counts[w] = counts.get(w, 0) + 1
    back = sum(1 for i, w in enumerate(trail) if w in seq[max(0, i - 1):i + 1])  # == one of 2 before
    return dict(assoc=(hits / valid) if valid else None, strength=(strength / valid) if valid else None,
                valid=valid, distinct=len(counts) / max(1, len(trail)), longest=max(counts.values(), default=0),
                backtrack=back / max(1, len(trail)))


def native_chain(H, start, steps=T):
    trail, prev, passes = [], start, 0
    for t in range(steps):
        recent = ([start] + trail)[-F:]
        word, n = H.say(H.ids(ASSOC_SHOTS + f"Word: {prev}. Association:"), ban=H.ban_ids(recent))
        passes += n
        trail.append(word)
        prev = word or prev
    return dict(trail=trail, passes=passes)


def run_free(H, bands, norms, n_starts=N_STARTS, cal=(), steps=T, log=print):
    lower = [w for w in H.pool if w.islower() and w not in set(cal)]
    if norms is not None:
        cand = [w for w in lower if w in norms]
    else:
        cand = [w for w in lower if len(w) >= 4]
    starts = random.Random(SEED_START).sample(cand, min(n_starts, len(cand)))
    demo = [w for w in DEMO if w in H.index and w not in starts]
    rng = random.Random(SEED_RAND)
    out = {c: [] for c in FREE_CONDS}
    for i, w in enumerate(starts + demo):
        for c in FREE_CONDS:
            if c in POLICIES:
                res = H.walk({w: 1.0}, c, band=engine_band(bands, c), steps=steps)
            elif c == "random":
                trail, passes = [], 0
                for _ in range(steps):
                    word, n = H.speak({rng.choice(H.pool): 1.0})
                    trail.append(word)
                    passes += n
                res = dict(trail=trail, passes=passes)
            else:
                res = native_chain(H, w, steps)
            res.update(start=w, demo=w in demo, **trail_stats(res["trail"], w, norms))
            out[c].append(res)
        if (i + 1) % 10 == 0:
            log(f"free {i + 1}/{len(starts) + len(demo)}")
    return out


def run_goal(H, items, bands, steps=T, log=print):
    kept, dropped = [], []
    for it in items:
        (kept if all(H.has_state(c) for c in it[:3]) else dropped).append(it)
    pool_lower = {w.lower() for w in H.pool}
    recs = []
    for n_i, (a, b, c, sol) in enumerate(kept):
        cues = [a, b, c]
        ban = H.ban_ids(cues)
        cue_v = torch.stack([H.vec({x: 1.0}) for x in cues])
        excl = [w for w in H.pool if w.lower() in set(cues)]
        rec = dict(cues=cues, solution=sol, in_pool=any(s in pool_lower for s in sol.split("/")))

        def pick(trail):
            """The spoken word closest to all three cues (max of min cosine)."""
            best, best_g, extra = None, -float("inf"), 0
            for wd in dict.fromkeys(trail):
                if not wd or wd in cues:
                    continue
                key = wd if wd in H.S else (H.pool[H.lower_index[wd]] if wd in H.lower_index else wd)
                if key not in H.S:
                    if not H.has_state(key):
                        continue
                if key not in H.index:
                    extra += 1          # one forward pass for a spoken word outside the pool
                g = float((cue_v @ H.vec({key: 1.0})).min())
                if g > best_g:
                    best, best_g = wd, g
            return (best if best is not None else (trail[-1] if trail else "")), extra

        start = {x: 1 / 3 for x in cues}
        for name, pol in GOAL_WALKS:
            res = H.walk(start, pol, band=engine_band(bands, pol), steps=steps, exclude=excl, ban=ban,
                         anchor_fixed=True)
            ans, extra = pick(res["trail"])
            rec[name] = dict(answer=ans, trail=res["trail"], k=res["k"], passes=res["passes"] + extra,
                             hit_trail=any(correct(x, sol) for x in res["trail"]),
                             first_correct=bool(res["trail"]) and correct(res["trail"][0], sol))
        word, n = H.speak(start, ban)
        rec["fixed_avg"] = dict(answer=word, passes=n)
        sim = (H.Xn @ cue_v.T).min(1).values
        sim[[H.index[w] for w in excl]] = -float("inf")
        rec["nearest_min"] = dict(answer=H.pool[int(sim.argmax())].lower(), passes=0)
        word, n = H.say(H.ids(rat_prompt(a, b, c)), ban=ban)
        rec["native"] = dict(answer=word, passes=n)
        budget = max(8, rec["head"]["passes"] - 3)
        p_ids = H.ids(rat_prompt(a, b, c, think=True))
        gen, n1 = H.generate(p_ids, budget)
        thinking = H.r.tok.decode(gen)
        word, n2 = H.say(p_ids + gen + H.ids("\nAnswer:" if "\n" not in thinking else "Answer:"), ban=ban)
        rec["reason"] = dict(answer=word, passes=n1 + n2, budget=budget, thinking=thinking.strip())
        budget = max(8, rec["loop"]["passes"] - 3)         # amendment 1: matched to the loop engine
        gen, n1 = H.generate(p_ids, budget)
        thinking = H.r.tok.decode(gen)
        word, n2 = H.say(p_ids + gen + H.ids("\nAnswer:" if "\n" not in thinking else "Answer:"), ban=ban)
        rec["reason_loop"] = dict(answer=word, passes=n1 + n2, budget=budget, thinking=thinking.strip())
        for cnd in GOAL_CONDS:
            rec[cnd]["correct"] = correct(rec[cnd]["answer"], sol)
        recs.append(rec)
        if (n_i + 1) % 10 == 0:
            log(f"goal {n_i + 1}/{len(kept)}")
    return dict(records=recs, dropped=[list(x) for x in dropped])


def gates(res):
    g = {}
    pz = res["P0"]
    g["P0_voice"] = dict(pass_=pz["passed"], copy=pz["copy"], blend=pz["blend"])
    free = res["free"]
    scored = [i for i, rec in enumerate(free["head"]) if not rec["demo"]]

    def series(c, key):
        return [free[c][i][key] for i in scored]
    if res["norms"] and scored:
        def paired(c1, c2):
            pairs = [(x, y) for x, y in zip(series(c1, "assoc"), series(c2, "assoc"))
                     if x is not None and y is not None]
            return B.paired_ci([x for x, _ in pairs], [y for _, y in pairs]) if pairs else [None] * 3
        f0 = paired("head", "random")
        f1 = paired("head", "fixed")
        dist_h, dist_f = series("head", "distinct"), series("fixed", "distinct")
        g["F0_sanity"] = dict(pass_=bool(f0[1] is not None and f0[1] > 0), head_minus_random=f0)
        lf = paired("loop", "loop_fixed")
        g["LF_loop_regulator_free"] = dict(pass_=bool(lf[1] is not None and lf[1] > 0),
                                           loop_minus_loop_fixed=lf, loop_minus_head=paired("loop", "head"),
                                           loop_minus_loop_nores=paired("loop", "loop_nores"),
                                           loop_minus_native=paired("loop", "native"))
        g["F1_regulator_free"] = dict(pass_=bool(f1[1] is not None and f1[1] > 0
                                                 and sum(dist_h) >= sum(dist_f)),
                                      head_minus_fixed=f1, head_minus_nn=paired("head", "nn"),
                                      head_minus_native=paired("head", "native"),
                                      distinct_head=sum(dist_h) / len(dist_h),
                                      distinct_fixed=sum(dist_f) / len(dist_f))
    recs = res["goal"]["records"]
    if recs:
        acc = {c: [float(r_[c]["correct"]) for r_ in recs] for c in GOAL_CONDS}
        h1 = {c: B.paired_ci(acc["head"], acc[c]) for c in ("walk_fixed", "fixed_avg", "nearest_min")}
        h2 = {c: B.paired_ci(acc["head"], acc[c]) for c in ("native", "reason")}
        g["H1_walk_beats_lookup"] = dict(pass_=all(v[1] > 0 for v in h1.values()),
                                         **{f"head_minus_{k}": v for k, v in h1.items()})
        g["H2_walk_beats_tokens"] = dict(pass_=all(v[1] > 0 for v in h2.values()),
                                         **{f"head_minus_{k}": v for k, v in h2.items()})
        d = lambda c: B.paired_ci(acc["loop"], acc[c])
        g["L1_loop_beats_static"] = dict(pass_=d("head")[1] > 0, loop_minus_head=d("head"))
        l2 = {c: d(c) for c in ("fixed_avg", "nearest_min")}
        g["L2_loop_beats_lookup"] = dict(pass_=all(v[1] > 0 for v in l2.values()),
                                         **{f"loop_minus_{k}": v for k, v in l2.items()})
        g["L3_breathing_goal"] = dict(pass_=d("loop_fixed")[1] > 0, loop_minus_loop_fixed=d("loop_fixed"),
                                      loop_minus_loop_nores=d("loop_nores"))
        l4 = {c: d(c) for c in ("native", "reason_loop")}
        g["L4_loop_beats_tokens"] = dict(pass_=all(v[1] > 0 for v in l4.values()),
                                         **{f"loop_minus_{k}": v for k, v in l4.items()})
    return g


def run(r, norms=None, norms_name=None, items=(), n_pool=N_POOL, n_starts=N_STARTS, steps=T, log=print):
    pool = O.generic_pool(r, [], n_pool)
    H = Head(r, pool, log)
    log(f"pool {len(H.pool)} words; selection layer {H.sel}; clamp band {H.band}")
    voice = p0(H)
    log(f"P0 voice: copy {voice['copy']:.3f}, blend {voice['blend']:.3f}")
    cal = random.Random(SEED_CAL).sample([w for w in H.pool if w.islower()], N_CAL)
    bands = {"static": calibrate(H, cal, "fixed"), "loop": calibrate(H, cal, "loop_fixed")}
    for e, b in bands.items():
        log(f"regulator band ({e}) lo {b[0]:.4f}, hi {b[1]:.4f}")
    free = run_free(H, bands, norms, n_starts, cal, steps, log)
    goal = run_goal(H, items, bands, steps, log)
    res = dict(model=r.name, n_pool=len(H.pool), selection_layer=H.sel, band=list(H.band),
               regulator_band=bands["static"], regulator_bands=bands, calibration_starts=cal,
               norms=norms_name, steps=steps,
               P0=voice, free=free, goal=goal,
               constants=dict(alpha=ALPHA, fatigue=F, beta=BETA, rho=RHO, k0=K0, kmax=KMAX, prune=PRUNE,
                              max_words=MAXW, beta_loop=BETA_L))
    res["gates"] = gates(res)
    return res


# ---------------------------------------------------------------- report
def report(res):
    f = lambda v: "–" if v is None else f"{v:.3f}"
    ci = lambda v: "–" if v is None or v[0] is None else "{:+.3f} [{:+.3f}, {:+.3f}]".format(*v)
    g = res["gates"]
    P = lambda k: "PASS" if g[k]["pass_"] else "FAIL"
    L = [f"# Head report — {res['model']}", "",
         f"Pool {res['n_pool']} words; selection layer {res['selection_layer']}; clamp band {res['band']}; "
         f"dtype {res.get('dtype', 'float32')}; {res['steps']} steps; regulator bands (fixed-walk quartiles on "
         f"{len(res['calibration_starts'])} calibration words): " +
         "; ".join(f"{e} [{b[0]:.4f}, {b[1]:.4f}]" for e, b in
                   res.get("regulator_bands", {"static": res["regulator_band"]}).items()) +
         f". Norms: {res['norms'] or 'none'}. Rules: HEAD.md (with amendment 1).",
         "", "The pool states (naming and association frames) are a one-time memory shared by every walk, "
         "fixed_avg and nearest_min; they are not charged per item. A loop step costs one forward pass "
         "(Qwen proposes) plus the voice.", "",
         f"**P0 voice: {P('P0_voice')}**  copy {g['P0_voice']['copy']:.3f} (≥ 0.90), "
         f"blend → one of the two {g['P0_voice']['blend']:.3f} (≥ 0.50)"
         + ("" if g["P0_voice"]["pass_"] else " — *voice unreliable: trails may not say what the thought holds*"), ""]
    if "F0_sanity" in g:
        L += [f"**F0 sanity: {P('F0_sanity')}**  head − random associate rate: {ci(g['F0_sanity']['head_minus_random'])}", "",
              f"**F1 regulator earns its keep (free): {P('F1_regulator_free')}**  head − fixed: "
              f"{ci(g['F1_regulator_free']['head_minus_fixed'])}; distinct words head "
              f"{g['F1_regulator_free']['distinct_head']:.3f} vs fixed {g['F1_regulator_free']['distinct_fixed']:.3f}  ",
              f"head − nn: {ci(g['F1_regulator_free']['head_minus_nn'])}; head − native (reference): "
              f"{ci(g['F1_regulator_free']['head_minus_native'])}", ""]
        if "LF_loop_regulator_free" in g:
            lf = g["LF_loop_regulator_free"]
            L += [f"**LF loop regulator earns its keep (free): {P('LF_loop_regulator_free')}**  loop − loop_fixed: "
                  f"{ci(lf['loop_minus_loop_fixed'])}  ",
                  f"loop − head {ci(lf['loop_minus_head'])}; loop − loop_nores {ci(lf['loop_minus_loop_nores'])}; "
                  f"loop − native (reference) {ci(lf['loop_minus_native'])}", ""]
    else:
        L += ["F0, F1, LF: not scored (no association norms given).", ""]
    if "H1_walk_beats_lookup" in g:
        h1, h2 = g["H1_walk_beats_lookup"], g["H2_walk_beats_tokens"]
        L += [f"**H1 walk beats the lookup (goal): {P('H1_walk_beats_lookup')}**  ",
              f"head − walk_fixed {ci(h1['head_minus_walk_fixed'])}  ",
              f"head − fixed_avg {ci(h1['head_minus_fixed_avg'])}  ",
              f"head − nearest_min {ci(h1['head_minus_nearest_min'])}", "",
              f"**H2 walk beats the model's own tokens: {P('H2_walk_beats_tokens')}**  ",
              f"head − native {ci(h2['head_minus_native'])}  ",
              f"head − reason (equal passes) {ci(h2['head_minus_reason'])}", ""]
    if "L1_loop_beats_static" in g:
        L += ["### Amendment 1: the closed-loop engine", "",
              f"**L1 Qwen-driven moves beat the static map: {P('L1_loop_beats_static')}**  loop − head "
              f"{ci(g['L1_loop_beats_static']['loop_minus_head'])}", "",
              f"**L2 loop beats the lookup: {P('L2_loop_beats_lookup')}**  loop − fixed_avg "
              f"{ci(g['L2_loop_beats_lookup']['loop_minus_fixed_avg'])}; loop − nearest_min "
              f"{ci(g['L2_loop_beats_lookup']['loop_minus_nearest_min'])}", "",
              f"**L3 breathing earns its keep (goal): {P('L3_breathing_goal')}**  loop − loop_fixed "
              f"{ci(g['L3_breathing_goal']['loop_minus_loop_fixed'])}; loop − loop_nores (reported) "
              f"{ci(g['L3_breathing_goal']['loop_minus_loop_nores'])}", "",
              f"**L4 loop beats the model's own tokens: {P('L4_loop_beats_tokens')}**  loop − native "
              f"{ci(g['L4_loop_beats_tokens']['loop_minus_native'])}; loop − reason_loop (equal passes) "
              f"{ci(g['L4_loop_beats_tokens']['loop_minus_reason_loop'])}", ""]
    free = res["free"]
    L += ["## Free mode", "", "| condition | associate rate | strength | distinct words | longest repeat | back-tracks | passes/trail |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for c in FREE_CONDS:
        rs = [x for x in free[c] if not x["demo"]]
        m = lambda key: (sum(x[key] for x in rs if x[key] is not None) /
                         max(1, sum(1 for x in rs if x[key] is not None))) if rs and any(x[key] is not None for x in rs) else None
        L.append(f"| {c} | {f(m('assoc'))} | {f(m('strength'))} | {f(m('distinct'))} | {f(m('longest'))} | "
                 f"{f(m('backtrack'))} | {f(m('passes'))} |")
    L += ["", "Trails (demo words first, then the first scored starts):", ""]
    show = [i for i, x in enumerate(free["head"]) if x["demo"]] + list(range(min(6, len(free["head"]))))
    for i in dict.fromkeys(show):
        L.append(f"**{free['head'][i]['start']}**  ")
        for c in FREE_CONDS:
            extra = f"  (k: {' '.join(map(str, free[c][i]['k']))})" if c in ("head", "loop") else ""
            L.append(f"- {c}: " + " → ".join(free[c][i]["trail"]) + extra)
        L.append("")
    recs = res["goal"]["records"]
    if recs:
        n = len(recs)
        L += ["## Goal mode: Remote Associates", "",
              f"{n} items kept (all three cues single tokens), {len(res['goal']['dropped'])} dropped; "
              f"solution in pool for {sum(r_['in_pool'] for r_ in recs)}/{n}.", "",
              "| condition | correct | accuracy | mean passes |", "|---|---:|---:|---:|"]
        for c in GOAL_CONDS:
            k = sum(r_[c]["correct"] for r_ in recs)
            L.append(f"| {c} | {k}/{n} | {k / n:.3f} | {sum(r_[c]['passes'] for r_ in recs) / n:.1f} |")
        L += ["", "Solution spoken anywhere in the trail / as the first spoken word (descriptive): " +
              "; ".join(f"{c} {sum(r_[c]['hit_trail'] for r_ in recs)}/{sum(r_[c]['first_correct'] for r_ in recs)}"
                        for c, _ in GOAL_WALKS) + f" (of {n}).", "",
              "| cues | solution | " + " | ".join(GOAL_CONDS) + " |", "|---|---|" + "---|" * len(GOAL_CONDS)]
        for r_ in recs[:20]:
            L.append(f"| {', '.join(r_['cues'])} | {r_['solution']} | " +
                     " | ".join(("**" + r_[c]["answer"] + "**") if r_[c]["correct"] else r_[c]["answer"]
                                for c in GOAL_CONDS) + " |")
        L += ["", "Trails for the first 5 items (head, then loop):", ""]
        for r_ in recs[:5]:
            for c in ("head", "loop"):
                L.append(f"- {c}: {', '.join(r_['cues'])} → " + " → ".join(r_[c]["trail"]) +
                         f"  (k: {' '.join(map(str, r_[c]['k']))})")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- self-test
def self_test():
    stub_words = ([f"pool{c}{d}" for c in "abcdefgh" for d in "abcdefgh"] +
                  ["oak", "tree", "wood", "river", "water", "bread", "cottage", "swiss", "cake", "cheese",
                   "cream", "skate", "ice", "Oslo", "One", "word", "Word", "Association", "Answer",
                   "Thinking", "Words", "Each", "puzzle", "gives"])
    texts = [NAME_SHOTS, NAME_Q, ASSOC_SHOTS, ASSOC_FRAME, ASSOC_Q, RAT_INTRO, rat_prompt("x", "y", "z", True),
             rat_prompt("x", "y", "z")] + stub_words
    for arch in ("neox", "qwen"):
        r = C.load_stub(arch, extra=texts)
        pool = O.generic_pool(r, [], 120)
        H = Head(r, pool, log=lambda *_: None)
        assert len(H.pool) > 40, "stub pool too small"
        # 1. the naming-frame states are exact: clamping a word's own state at full depth into the
        #    slot gives the same logits as reading the word itself
        w = H.pool[3]
        ids_w, pos_w = H.name_prompt(w)
        ids_s, pos_s = H.name_prompt(C.SLOT)
        assert pos_w == pos_s and len(ids_w) == len(ids_s)
        tg = H.targets({w: 1.0})
        with torch.no_grad():
            a = r.model.base_model(input_ids=torch.tensor([ids_w])).last_hidden_state[0, -1]
            hooks = r._hooks(pos_s, tg)
            try:
                b = r.model.base_model(input_ids=torch.tensor([ids_s])).last_hidden_state[0, -1]
            finally:
                for hk in hooks:
                    hk.remove()
        assert (a - b).abs().max() < 1e-4, "clamping a word's own state into the slot is not exact"
        ia, pa = H.name_prompt(w, "assoc")
        isl, psl = H.name_prompt(C.SLOT, "assoc")
        with torch.no_grad():
            a2 = r.model.base_model(input_ids=torch.tensor([ia])).last_hidden_state[0, -1]
            hooks = r._hooks(psl, H.targets({w: 1.0}, "assoc"))
            try:
                b2 = r.model.base_model(input_ids=torch.tensor([isl])).last_hidden_state[0, -1]
            finally:
                for hk in hooks:
                    hk.remove()
        assert (a2 - b2).abs().max() < 1e-4, "association-frame clamp is not exact"
        lp = H.associate({w: 1.0})
        assert lp.shape[0] == len(H.pool) and abs(float(lp.exp().sum()) - 1) < 1e-4
        assert int(H.pool_tok[3]) == H.name_prompt(w)[0][H.name_prompt(w)[1]]
        # batched capture equals one-at-a-time capture
        one = H.S[w].clone()
        del H.S[w]
        H.capture([w], bs=1)
        assert (one - H.S[w]).abs().max() < 1e-4, "batched capture differs"
        # 2. the voice respects bans and counts passes
        p_before = H.passes
        word, n = H.speak({w: 1.0}, ban=H.ban_ids([w.lower()]))
        assert word != w.lower() and 1 <= n <= 3 and H.passes == p_before + n
        # 3. a thought only ever holds pool words, weights sum to 1
        res = H.walk({w: 1.0}, "head", band=[5.0, 6.0], steps=5, speak=False)
        assert res["k"] == [4, 8, 16, 32, 32], "regulator should widen when change is below the band"
        res = H.walk({w: 1.0}, "head", band=[-2.0, -1.0], steps=4, speak=False)
        assert res["k"] == [4, 2, 1, 1], "regulator should narrow when change is above the band"
        res = H.walk({w: 1.0}, "head", band=[5.0, 6.0], steps=7)   # reaches k = 32 while speaking
        assert len(res["trail"]) == 7 and res["k"][-1] == KMAX       # (the crash of the first real run)
        res = H.walk({w: 1.0}, "loop", band=[5.0, 6.0], steps=7)
        assert len(res["trail"]) == 7 and res["k"][-1] == KMAX and res["passes"] >= 14
        res = H.walk({w: 1.0}, "head", band=[5.0, 6.0], steps=3)
        assert len(res["trail"]) == 3 and res["passes"] >= 3
        nn = H.walk({w: 1.0}, "nn", steps=6, speak=False)
        assert len(nn["surprise"]) == 6
        for layer in r.layers:
            assert not layer._forward_hooks and not layer._forward_pre_hooks, "hook left behind"
    # 4. scoring helpers
    assert correct("cheese", "cheese") and correct("Cheeses", "cheese") and correct("court", "school/court")
    assert not correct("", "cheese") and not correct("chess", "cheese")
    st = trail_stats(["tree", "wood", "tree"], "oak", {"oak": {"tree": 0.8}, "tree": {"wood": 0.1}})
    assert st["assoc"] == 1.0 and st["valid"] == 2 and st["longest"] == 2  # wood is not a cue
    # 5. the whole run, both modes, on the Qwen stub
    tmp = Path("runs") / "selftest_head"
    tmp.mkdir(parents=True, exist_ok=True)
    nf = tmp / "norms.tsv"
    lines = ["cue\tresponse\tR123.Strength"]
    for x in H.pool[:60]:
        if x.islower():
            lines += [f"{x}\t{H.pool[(H.index[x] + 1) % len(H.pool)].lower()}\t0.2", f"{x}\ttree\t0.1"]
    nf.write_text("\n".join(lines) + "\n", encoding="utf-8")
    norms = load_norms(nf)
    rf = tmp / "rat.tsv"
    rf.write_text("cue1\tcue2\tcue3\tsolution\ncottage\tswiss\tcake\tcheese\ncream\tskate\twater\tice\n"
                  "oak\tab-cd\triver\twood\n", encoding="utf-8")
    items = load_rat(rf)
    gl = globals()
    old = (gl["N_CAL"], gl["N_P0"])
    gl["N_CAL"], gl["N_P0"] = 3, 10
    try:
        res = run(r, norms, "selftest", items, n_pool=120, n_starts=4, steps=4, log=lambda *_: None)
    finally:
        gl["N_CAL"], gl["N_P0"] = old
    assert len(res["goal"]["records"]) == 2 and len(res["goal"]["dropped"]) == 1
    for rec in res["goal"]["records"]:
        for c in GOAL_CONDS:
            assert rec[c]["answer"] not in rec["cues"], "a banned cue was spoken"
        assert rec["reason"]["passes"] <= max(8, rec["head"]["passes"] - 3) + 3
    json.dumps(res)
    txt = report(res)
    assert "H1 walk beats the lookup" in txt and "F1 regulator" in txt and "L4 loop" in txt and "LF loop" in txt
    for rec in res["goal"]["records"]:
        assert rec["reason_loop"]["passes"] <= max(8, rec["loop"]["passes"] - 3) + 3
    rat = load_rat(Path(__file__).with_name("data") / "rat144.tsv")
    assert len(rat) == 144 and rat[0] == ("cottage", "swiss", "cake", "cheese")
    shot_words = {w for it in RAT_SHOT_ITEMS for w in it[:4]}
    assert not shot_words & {w for it in rat for x in it for w in x.split("/")}, "few-shot words overlap the test"
    print("Head self-test passed (random NeoX and Qwen stub models; plumbing only).")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--revision", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32",
                   help="float32 is the protocol; bfloat16 halves memory (a recorded deviation)")
    p.add_argument("--norms", default=None, help="association norms (SWOW strength file or USF pairs)")
    p.add_argument("--rat", default=str(Path(__file__).with_name("data") / "rat144.tsv"))
    p.add_argument("--n-pool", type=int, default=N_POOL)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--report-from", default=None, help="rewrite REPORT.md from a saved head.json")
    a = p.parse_args()
    if a.self_test:
        return self_test()
    if a.report_from:
        src = Path(a.report_from)
        res = json.loads(src.read_text(encoding="utf-8"))
        (src.parent / "REPORT.md").write_text(report(res), encoding="utf-8")
        return print(report(res))
    norms = load_norms(a.norms) if a.norms else None
    items = load_rat(a.rat)
    print(f"loading {a.model} as {a.dtype} ...", flush=True)
    r = C.load(a.model, a.device, getattr(torch, a.dtype), revision=a.revision, frame="shots")
    print("loaded", flush=True)
    res = run(r, norms, Path(a.norms).name if a.norms else None, items, n_pool=a.n_pool,
              log=lambda *x: print(*x, flush=True))
    res["dtype"] = a.dtype
    out = Path("runs") / ("head_" + a.model.replace("/", "_") + ("_" + a.revision if a.revision else ""))
    out.mkdir(parents=True, exist_ok=True)
    (out / "head.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (out / "REPORT.md").write_text(report(res), encoding="utf-8")
    print(report(res))
    print("saved", out)


if __name__ == "__main__":
    main()
