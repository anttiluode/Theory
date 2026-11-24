"""Shared machinery: the concept tree, a plain few-shot yes/no reader that works on
base and instruct models alike, state capture at the item word, and the clamp.

The clamp overwrites the item word's position from the token embedding through the
last band layer, so nothing of the original word survives around it (the
RakeisuusClaude bypass fix). The self-test checks that.
"""
import math
import random
import re

import torch

# family -> (kingdom, leaf candidates). Leaves that are not one token are dropped per model.
FAMILIES = {
    "tree": ("plant", ["oak", "maple", "pine", "elm", "cedar", "birch", "willow", "spruce"]),
    "vegetable": ("plant", ["carrot", "potato", "onion", "cabbage", "lettuce", "cucumber", "garlic", "turnip"]),
    "bird": ("animal", ["eagle", "owl", "crow", "pigeon", "duck", "sparrow", "parrot", "swan"]),
    "fish": ("animal", ["salmon", "shark", "trout", "tuna", "cod", "carp", "herring", "eel"]),
    "insect": ("animal", ["ant", "bee", "beetle", "butterfly", "mosquito", "moth", "wasp", "termite"]),
    "tool": ("made", ["hammer", "wrench", "drill", "shovel", "axe", "rake", "chisel", "spade"]),
    "vehicle": ("made", ["car", "bus", "truck", "bicycle", "train", "tractor", "van", "taxi"]),
    "instrument": ("made", ["guitar", "piano", "violin", "drum", "flute", "trumpet", "harp", "cello"]),
}
FAMILY_PHRASE = {"instrument": "a musical instrument"}
KINGDOM_PHRASE = {"plant": "a plant", "animal": "an animal", "made": "made by people"}
MAX_LEAVES, MIN_LEAVES = 6, 4
SLOT = "thing"                       # the neutral item the combined state is written into
# Few-shot frame, plain text, no chat template, so base checkpoints can answer too.
# None of these example items belongs to the tree above.
SHOTS = ("Item: rose. Is it a flower? yes\n"
         "Item: lion. Is it a fish? no\n"
         "Item: chair. Is it a tool? no\n"
         "Item: wolf. Is it an animal? yes\n"
         "Item: cup. Is it a cup? yes\n"
         "Item: lion. Is it a tiger? no\n")
# The RakeisuusClaude frame, for instruct models only: chat template, zero-shot.
CHAT_USER = "Item: {w}\n\n{q} Answer yes or no."
CHAT_PREFIX = "Answer:"
FRAMES = ("shots", "chat")


def article(x):
    return "an" if x[:1] in "aeiou" else "a"


def family_q(f):
    return f"Is it {FAMILY_PHRASE.get(f, article(f) + ' ' + f)}?"


def kingdom_q(k):
    p = KINGDOM_PHRASE[k]
    return f"Is it {p}?" if not p.startswith("made") else f"Is it {p}?"


def leaf_q(w):
    return f"Is it {article(w)} {w}?"


def _hs(out):
    return out[0] if isinstance(out, (tuple, list)) else out


def _replace(out, new):
    return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new


def find_layers(model):
    for path in ("model.layers", "gpt_neox.layers", "transformer.h", "model.decoder.layers"):
        obj = model
        try:
            for p in path.split("."):
                obj = getattr(obj, p)
            if len(obj):
                return obj
        except AttributeError:
            continue
    raise SystemExit("unsupported decoder layout (Qwen, Llama, GPT-NeoX/Pythia and GPT-2 style work)")


class Reader:
    def __init__(self, model, tok, device, name, frame="shots"):
        if frame not in FRAMES:
            raise SystemExit(f"unknown frame {frame!r}; use one of {FRAMES}")
        if frame == "chat" and not getattr(tok, "chat_template", None):
            raise SystemExit("the chat frame needs a model with a chat template (base models: use --frame shots)")
        self.model, self.tok, self.device, self.name, self.frame = model.eval(), tok, device, name, frame
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.layers = find_layers(model)
        self.n_layers = len(self.layers)
        pad = getattr(tok, "pad_token_id", None)
        self.pad = pad if pad is not None else (getattr(tok, "eos_token_id", None) or 0)
        self.yes, self.no = self._first_ids(["yes", "Yes"]), self._first_ids(["no", "No"])
        if frame == "chat":                    # as in RakeisuusClaude: with and without the space
            self.yes = sorted(set(self.yes) | set(self._bare_ids(["yes", "Yes"])))
            self.no = sorted(set(self.no) | set(self._bare_ids(["no", "No"])))
        if not self.yes or not self.no or set(self.yes) & set(self.no):
            raise SystemExit("STOP: yes/no are not clean single tokens")
        # RMSNorm reads the raw vector's direction; LayerNorm reads it after removing the
        # mean over coordinates. Angles are measured the way the model's own norm sees them.
        names = {type(m).__name__ for m in model.modules()}
        self.centered = not any("RMSNorm" in n for n in names)
        self._cal = {}

    def _first_ids(self, words):
        out = set()
        for w in words:
            ids = self.tok(" " + w, add_special_tokens=False)["input_ids"]
            if len(ids) == 1:
                out.add(ids[0])
        return sorted(out)

    def _bare_ids(self, words):
        out = []
        for w in words:
            ids = self.tok(w, add_special_tokens=False)["input_ids"]
            if len(ids) == 1:
                out.append(ids[0])
        return out

    def band(self, end=None):
        return (-1, end if end is not None else min(self.n_layers - 1, round(12 * self.n_layers / 28)))

    def prompt(self, word, q):
        """Token ids and the item word's position, or None if the word is not one token."""
        if self.frame == "chat":
            user = CHAT_USER.format(w=word, q=q.replace("Is it", "Is the item above", 1))
            msgs = [{"role": "user", "content": user}]
            try:
                text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                    enable_thinking=False)
            except TypeError:
                text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            text += CHAT_PREFIX
            ws = text.rindex("Item: " + word + "\n") + len("Item: ")
        else:
            text = SHOTS + f"Item: {word}. {q}"
            ws = text.rindex("Item: " + word + ".") + len("Item: ")
        we = ws + len(word)
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        hit = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if a < we and b > ws]
        ok = len(hit) == 1 and enc["offset_mapping"][hit[0]][1] <= we
        return enc["input_ids"], (hit[0] if ok else None)

    def single(self, word):
        return self.prompt(word, "Is it?")[1] is not None

    # ---- states
    @torch.no_grad()
    def states(self, words):
        """{word: tensor [L+1, d]}; row 0 is the token embedding, row l+1 the output of layer l."""
        out = {}
        for w in words:
            ids, pos = self.prompt(w, "Is it?")
            store, hooks = {}, []
            for l in range(self.n_layers):
                hooks.append(self.layers[l].register_forward_hook(
                    lambda m, i, o, l=l: store.__setitem__(l, _hs(o)[0, pos].detach().float().cpu())))

            def pre(mod, args, kwargs):
                h = args[0] if args else kwargs["hidden_states"]
                store[-1] = h[0, pos].detach().float().cpu()
            hooks.append(self.layers[0].register_forward_pre_hook(pre, with_kwargs=True))
            try:
                self.model(input_ids=torch.tensor([ids], device=self.device))
            finally:
                for h in hooks:
                    h.remove()
            out[w] = torch.stack([store[l] for l in range(-1, self.n_layers)])
        return out

    # ---- clamped yes/no evidence
    def _hooks(self, pos, targets):
        hooks = []
        for l, v in (targets or {}).items():
            if l == -1:
                def pre(mod, args, kwargs, v=v):
                    h = (args[0] if args else kwargs["hidden_states"]).clone()
                    h[:, pos] = v.to(h.device, h.dtype)
                    if args:
                        return (h,) + tuple(args[1:]), kwargs
                    kwargs["hidden_states"] = h
                    return args, kwargs
                hooks.append(self.layers[0].register_forward_pre_hook(pre, with_kwargs=True))
            else:
                def post(mod, inp, out, v=v):
                    h = _hs(out).clone()
                    h[:, pos] = v.to(h.device, h.dtype)
                    return _replace(out, h)
                hooks.append(self.layers[l].register_forward_hook(post))
        return hooks

    @torch.no_grad()
    def margins(self, word, questions, targets=None):
        """{key: yes-minus-no log evidence} for (key, question) about "Item: word"."""
        encs = [self.prompt(word, q) for _, q in questions]
        pos = encs[0][1]
        assert pos is not None and all(p == pos for _, p in encs)
        n = max(len(ids) for ids, _ in encs)
        x = torch.full((len(encs), n), self.pad, dtype=torch.long)
        mask = torch.zeros((len(encs), n), dtype=torch.long)
        for b, (ids, _) in enumerate(encs):
            x[b, :len(ids)] = torch.tensor(ids)
            mask[b, :len(ids)] = 1
        last = (mask.sum(1) - 1).to(self.device)
        hooks = self._hooks(pos, targets)
        try:
            logits = self.model(input_ids=x.to(self.device), attention_mask=mask.to(self.device)).logits
        finally:
            for h in hooks:
                h.remove()
        lp = torch.log_softmax(logits[torch.arange(len(encs), device=logits.device), last].float(), -1)
        m = torch.logsumexp(lp[:, self.yes], -1) - torch.logsumexp(lp[:, self.no], -1)
        return {k: float(v) for (k, _), v in zip(questions, m.cpu())}

    def calibrated(self, word, questions, targets=None):
        """Evidence minus the evidence for the printed empty slot 'Item: thing'."""
        need = [(k, q) for k, q in questions if k not in self._cal]
        if need:
            self._cal.update(self.margins(SLOT, need))
        m = self.margins(word, questions, targets)
        return {k: v - self._cal[k] for k, v in m.items()}


# ================================================================ the tree for one model
class Tree:
    def __init__(self, reader, max_leaves=MAX_LEAVES, min_leaves=MIN_LEAVES):
        self.fam, self.dropped = {}, []
        for f, (k, cands) in FAMILIES.items():
            kept = []
            for w in cands:
                if len(kept) == max_leaves:
                    break
                (kept if reader.single(w) else self.dropped).append(w)
            if len(kept) >= min_leaves:
                self.fam[f] = (k, kept)
        self.kingdoms = sorted({k for k, _ in self.fam.values()})
        self.leaves = [(w, f, k) for f, (k, ws) in self.fam.items() for w in ws]
        self.family_of = {w: f for w, f, _ in self.leaves}

    def questions(self, family):
        qs = [(f"top:{k}", kingdom_q(k)) for k in self.kingdoms]
        qs += [(f"cat:{f}", family_q(f)) for f in self.fam]
        qs += [(f"leaf:{w}", leaf_q(w)) for w in self.fam[family][1]]
        return qs

    def judge(self, word, m):
        f = self.family_of[word]
        k = self.fam[f][0]
        pick = lambda pre: max((key for key in m if key.startswith(pre)), key=m.get)
        return dict(cat_ok=pick("cat:") == f"cat:{f}", top_ok=pick("top:") == f"top:{k}",
                    leaf_ok=pick("leaf:") == f"leaf:{word}", cat=m[f"cat:{f}"], top=m[f"top:{k}"],
                    leaf=m[f"leaf:{word}"],
                    rejects_own=m[f"leaf:{word}"] < 0)


def blend(S, pairs, band):
    """sum of coef * state at each band layer. pairs: [(coef, tensor[L+1, d])]."""
    return {l: sum(c * s[l + 1] for c, s in pairs) for l in range(band[0], band[1] + 1)}


def matched_random(delta, seed):
    """A random move with the same length as `delta` at every layer (fixed seed)."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    for l, d in delta.items():
        r = torch.randn(d.shape, generator=g)
        out[l] = r * (d.norm() / r.norm().clamp_min(1e-12))
    return out


def boot(xs, n=4000, seed=0):
    xs = list(xs)
    if not xs:
        return dict(mean=None, lo=None, hi=None, n=0)
    rng = random.Random(seed)
    ms = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(n))
    return dict(mean=sum(xs) / len(xs), lo=ms[int(0.025 * n)], hi=ms[int(0.975 * n) - 1], n=len(xs))


def spearman(a, b):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for i, j in enumerate(o):
            r[j] = float(i)
        return r
    ra, rb = rank(a), rank(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else 0.0


def perm_p(a, b, n=2000, seed=0):
    """Two-sided permutation p-value for a Spearman correlation."""
    rng = random.Random(seed)
    r0 = abs(spearman(a, b))
    bb = list(b)
    hits = 0
    for _ in range(n):
        rng.shuffle(bb)
        hits += abs(spearman(a, bb)) >= r0
    return (hits + 1) / (n + 1)


# ================================================================ loading
def load(name, device, dtype=torch.float32, revision=None, frame="shots"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, revision=revision)
    kw = dict(revision=revision) if revision else {}
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, **kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, **kw)
    return Reader(model.to(device), tok, device, name + (f"@{revision}" if revision else ""), frame)


class StubTokenizer:
    """Word-level tokenizer with offsets (tests only)."""
    PAT = re.compile(r" ?[A-Za-z']+| ?[^\sA-Za-z]|\n")
    pad_token_id, eos_token_id = 0, 1

    def __init__(self, texts):
        self.vocab, seen = ["<pad>", "<eos>", "<unk>"], {"<pad>", "<eos>", "<unk>"}
        for t in texts:
            for p in self.PAT.findall(t):
                for q in (p, " " + p.strip(), p.strip()):
                    if q and q not in seen:
                        seen.add(q)
                        self.vocab.append(q)
        self.idx = {w: i for i, w in enumerate(self.vocab)}

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        ms = list(self.PAT.finditer(text))
        out = {"input_ids": [self.idx.get(m.group(), 2) for m in ms]}
        if return_offsets_mapping:
            out["offset_mapping"] = [m.span() for m in ms]
        return out

    def decode(self, ids):
        return "".join(self.vocab[i] for i in ids)

    chat_template = "stub"

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
        return "".join(f"<{m['role']}>\n{m['content']}\n" for m in msgs) + "<assistant>\n"


def load_stub(arch="qwen", layers=6, d=64, seed=0, frame="shots"):
    words = [w for _, ws in FAMILIES.values() for w in ws]
    texts = [SHOTS, SLOT, CHAT_USER, CHAT_PREFIX, "<user> <assistant> the item above",
             "Item: . Is it? yes no Yes No a an made by people musical instrument"] + \
        words + list(FAMILIES) + ["plant", "animal"]
    tok = StubTokenizer(texts)
    torch.manual_seed(seed)
    if arch == "qwen":
        from transformers import Qwen2Config, Qwen2ForCausalLM
        cfg = Qwen2Config(vocab_size=len(tok.vocab), hidden_size=d, intermediate_size=2 * d,
                          num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
                          max_position_embeddings=512, tie_word_embeddings=True)
        model = Qwen2ForCausalLM(cfg)
    else:
        from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
        cfg = GPTNeoXConfig(vocab_size=len(tok.vocab), hidden_size=d, intermediate_size=2 * d,
                            num_hidden_layers=layers, num_attention_heads=4, max_position_embeddings=512)
        model = GPTNeoXForCausalLM(cfg)
    return Reader(model, tok, "cpu", f"stub-{arch}", frame)


def plumbing_test(r):
    """Identity clamp changes nothing; a full-depth clamp from the embedding leaves no trace
    of the original word; every hook is removed afterwards."""
    tree = Tree(r)
    (w, f, _), (o, _, _) = tree.leaves[0], tree.leaves[-1]
    qs = tree.questions(f)
    S = r.states([w, o])
    full = (-1, r.n_layers - 1)
    a = r.margins(w, qs)
    b = r.margins(w, qs, blend(S, [(1.0, S[w])], full))
    assert max(abs(a[k] - b[k]) for k in a) < 1e-4, "identity clamp changed the answer"
    own = r.margins(o, qs)
    via = r.margins(w, qs, blend(S, [(1.0, S[o])], full))
    assert max(abs(own[k] - via[k]) for k in own) < 1e-4, "the original word leaked around the clamp"
    for l in r.layers:
        assert not l._forward_hooks and not l._forward_pre_hooks, "a hook was left behind"
    return tree
