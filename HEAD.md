# Head — a thought that moves by itself inside a frozen model

Protocol recorded before any real-model run, 25 November 2026 (Claude, Opus 5.5).
Gates, thresholds, constants and the primary comparisons are frozen here. Changes after seeing
results go in a dated amendment.

## Why

Today's runs separated four jobs that the "head" idea had mixed together:

- **Reading works at full width.** Neighbours chosen with the whole middle-layer state are good
  (oak → wood, tree; Nile → river, Egypt). A narrow PCA window chooses worse (Ocular O2).
- **Only real states can be written back.** A projected shadow does nothing (Bridge B1); blends of
  real word states change what the model answers (Iris, Ocular).
- **Averages of many neighbours move a word toward its kind** (the sibling-average/prototype
  effect, RakeisuusClaude). One neighbour moves it sideways. That is a grain knob: zoom out, zoom in.
- **Fixed edits help only an under-read model.** Qwen3-1.7B answers 96% of the taxonomy questions
  natively, so there was nothing left to repair (Ocular amendment 1). A test of a strong model needs a
  task it fails.

The head puts these four parts into one loop outside a frozen model, with the regulator of the 2025
PerceptionLab ECG loop (homeostatic coupler: hold change inside a band, widen when stuck, narrow when
too changeable). It has nothing trained. The question is whether that loop does anything the model's
own tokens, or a fixed lookup, do not.

## The loop (fixed)

Model `Qwen/Qwen3-1.7B`, float32, plain text frames (no chat template), greedy decoding everywhere.

- **Frame.** `NAME_SHOTS` = four copy examples (`Item: rose. One word: rose`, lion, chair, cup), then
  `Item: <w>. One word:`. All states are captured at the item word *in this frame*, so a clamped pure
  state is exactly the model's own run on that word.
- **Memory.** The 8000-word generic pool of `ocular.py` (`generic_pool`: single tokens, lowercase or
  title-case, 3+ letters, few-shot words and the slot word excluded). States are kept from the token
  embedding through layer max(band end, selection layer).
- **A thought** is a set of real pool words with weights summing to 1. It is never anything else.
  Writing it = the weighted sum of those words' real states, clamped into the slot `thing` at every
  layer of `band()` = (−1, 12), as in Ocular.
- **Reading (selection).** Query = the thought's state at the selection layer n_layers // 2 = 14,
  centred on the pool mean, cosine against all pool words at full width. Score = cos(query, word) +
  β · cos(residue, word). Masked: fatigued words, the thought's own words, excluded words.
- **Grain.** k = how many top-scoring words the thought moves toward. Move: thought ← (1 − α) · thought
  + α · (uniform over the top k). k = 1 is zoom in (one neighbour's real state), large k is zoom out
  (the average of many: the prototype). k ∈ {1, 2, 4, 8, 16, 32}. Weights below 0.02 are dropped,
  at most 48 words kept, then renormalised.
- **Fatigue (inhibition of return).** Every word used as a target is masked for F = 6 steps.
- **Residue (the hanging drop).** A running average of past queries, residue ← 0.8 · residue +
  0.2 · query (ρ = 0.2), β = 0.3. In goal mode the residue is the cue average and does not drift (ρ = 0).
- **Regulator.** Surprise s = 1 − cos(query now, query before the move). If s < lo, or the spoken
  word repeats the previous one, k doubles (zoom out). If s > hi, k halves (zoom in). Otherwise k stays.
  Starting k = 4. **lo and hi are the 25th and 75th percentiles of s for the fixed-grain walk on 20
  calibration start words**, drawn from the pool with seed 7 and never used as test starts. This is
  label-free and fixed before any test trail runs.
- **Voice.** After each move, the thought is clamped into `Item: thing. One word:` and Qwen says one
  word (greedy, first token must begin a word, up to 3 tokens, stop at the first non-letter
  continuation). That word is the trail. α = 0.5, T = 12 steps.

## Conditions

### Free mode (no goal)

Starts: 60 words that are both cues in the association norms and pool words (lowercase), seed 11.

| condition | what it is |
|---|---|
| `head` | the loop above |
| `fixed` | the loop with the regulator off (k = 4 always) |
| `nn` | pure nearest-neighbour walk: k = 1, α = 1, no fatigue, no residue (expected to fall into 2-cycles) |
| `random` | each step a random pool word |
| `native` | Qwen's own token chain: `Word: <w>. Association:` with 3 shots (salt→pepper, needle→thread, king→queen), the last F words banned so it cannot ping-pong. Reference only |

**Ground truth:** human free-association norms. Primary: Small World of Words English (De Deyne et al.
2019, `strength.SWOW-EN.R123`); fallback if SWOW is unavailable: USF norms (Nelson, McEvoy & Schreiber
2004). The report prints which file was used. A step a → b is a **human associate** if b is a response
to cue a in the norms. Per start: the fraction of steps (with a as a norm cue) that are associates.
Trails for oak, river, bread, winter, iron, music and sauna (when they are pool words and not starts)
are printed as demos and not scored.

### Goal mode: Remote Associates

The 144 compound remote associate problems of Bowden & Jung-Beeman (2003), e.g. cottage / swiss /
cake → cheese (`data/rat144.tsv`: cues and solutions only, with the published 15-second solve rate;
checked to agree item for item between two independent public copies of the appendix). An item is kept only if all three cues are single tokens in the
frame; the answer is never required to be in the pool. A spoken answer is correct if it equals the
solution after lowercasing, allowing a plural `s`/`es` either way; `school/court` accepts both.
The three cues are banned from every condition's first answer token.

| condition | what it is |
|---|---|
| `head` | thought starts as the three cues (1/3 each), residue = their average, cues masked; 12 steps. Answer = the spoken word whose state has the highest **minimum** cosine to the three cues (one forward pass charged for each spoken word outside the pool) |
| `walk_fixed` | the same with the regulator off (equal steps) |
| `fixed_avg` | Category's fixed average: the three cues' mean state, spoken once |
| `nearest_min` | no language model at all: the pool word with the highest minimum cosine to the three cues |
| `native` | Qwen asked directly, 3 shots (skunk/kings/boiled → cabbage, widow/bite/monkey → spider, elephant/lapse/vivid → memory; none of their words appears in the 144) |
| `reason` | Qwen with the same shots plus a short "Thinking:" line, greedy, allowed **as many forward passes as the head used on that item** (minus 3 for the answer, at least 8), stopping early at a newline, then `Answer:` |

**Compute** is counted in language-model forward passes per item. The one-time capture of the
8000 pool states is a memory, shared by `head`, `walk_fixed`, `fixed_avg` and `nearest_min`, and is not
charged per item. That is stated in every report.

## Gates

All intervals: paired bootstrap 95% over start words or items, 4000 draws (`bridge.paired_ci`).

**P0 — the voice reports the slot.** Over 200 pool words (seed 5): (a) unclamped `Item: w. One word:`
returns w for ≥ 90%; (b) a 50/50 blend of two pool words returns one of the two for ≥ 50%.
If P0 fails, the trails do not say what the thought holds; every other gate is still printed, marked
*voice unreliable*.

**F0 — sanity.** `head` associate rate − `random`: lower bound > 0.

**F1 — the regulator earns its keep in free mode (primary, free).** `head` − `fixed` associate rate:
lower bound > 0, **and** `head` has at least as many distinct words per trail as `fixed`.

**H1 — the walk beats the lookup (primary, goal).** `head` accuracy − each of `walk_fixed`,
`fixed_avg` and `nearest_min`: all three lower bounds > 0.

**H2 — the walk beats the model's own tokens.** `head` − `native` and `head` − `reason`: both lower
bounds > 0.

**Kill.** If H1 fails, the head adds nothing over a fixed average or a static lookup on this task, and
the free-reign head line ends here for word-level thoughts. If H1 passes and H2 fails, the loop is a
better use of the model's geometry than a lookup, but not better than letting the model talk.

**Reported, not gated:** `native` in free mode, hit@trail (solution spoken anywhere in the head's 12
words), solution-in-pool coverage, mean passes per condition, mean k over time, distinct words, longest
repeat, all trails and all answers.

## Prediction before the run (Claude)

H2 fails: a 1.7B model answering compound associates directly is likely better than a walk through
its middle-layer geometry, because compounds (cottage cheese) are lexical co-occurrence, which the
token head carries and a semantic neighbourhood may not. F0 passes. F1 and H1 are open.

## What this is not

- Not a sentence-level stream of thought. Thoughts are weighted sets of single words
  (Qwenimagination is the bigger version).
- A deterministic walk on a finite memory eventually cycles. Fatigue only delays it. T = 12.
- Prior work to check before claiming anything: continuous-thought reasoning (Coconut, Hao et al.
  2024), recurrent-depth models, random walks on embedding graphs and spreading activation for the
  RAT (Kajić et al. 2017), and LLMs on the RAT.
- One model, one frame, greedy decoding.

## Run

```
python head.py --self-test
python head.py --device cuda --norms path\to\strength.SWOW-EN.R123.csv
```

Without `--norms`, free mode still runs and saves trails, but F0 and F1 are not scored.
Output: `runs/head_<model>/REPORT.md` and `head.json`.

## Amendment 1 — recorded 25 November 2026, after a crash and before any scored result

**What had been seen.** The first Qwen run (float32) captured the 8000 pool states and printed
**P0: copy 1.000, blend 0.925 (PASS)** and the static regulator band **[0.0415, 0.0675]**. It then
crashed in the first head trail of free mode. No association rate, trail or RAT answer was seen.

**The bug.** At k = 32 each target gets α / k = 0.016, below the prune threshold 0.02, so a
zoom-out step could prune every word and leave an empty thought. Fix: carry-over weights below
0.02 are still dropped, but **the current step's k targets are always kept** (cap max(48, k) words).
If fatigue plus the thought's own words would mask the whole pool, fatigue is lifted for that step
(only reachable with tiny test pools). Nothing else in the static engine changes.

**A second engine (Neon's and Sol's critique).** In the static engine every move is chosen by cosine
in a fixed layer-14 table; Qwen's upper layers only narrate. Remote associates are compound links
(cottage *cheese*, sore *throat*), not similarity, and a similarity table mostly finds siblings
(cottage ↔ cabin). So the head as frozen above may lose RAT for a reason unrelated to breathing.
Amendment 1 adds the **closed-loop engine**, run beside the static one in the same run:

- **Frame:** `ASSOC_FRAME` = `Item: salt. Association: pepper` (needle → thread, king → queen), then
  `Item: thing. Association:`. Every pool word's states are also captured in this frame, so the
  thought written here is again made only of real states of this frame (exactness checked in the
  self-test).
- **Step:** clamp the thought into the slot (band (−1, 12)), run Qwen to the end, take the
  next-token log-probabilities restricted to the pool words (renormalised over the pool). Score =
  that + β_L · cos(residue, word), β_L = 2.0 nats per unit cosine. Same masks (fatigue F = 6, own
  words, cues in goal mode). Take the top k; their weights are **Qwen's own probabilities**
  renormalised over the k (not uniform). Same move (α = 0.5), prune rule, voice, T = 12 and
  answer rule (max of min cosine over spoken words) as the static engine.
- **Regulator:** the same rule on the same surprise measure, with its own band: the 25th–75th
  percentiles of `loop_fixed` on the same 20 calibration words.
- **Cost:** one forward pass per step for the proposal, plus the voice.

New conditions: `loop` (regulated), `loop_fixed` (k = 4 always), `loop_nores` (regulated, β_L = 0,
no residue), in both modes; `reason_loop` = Qwen reasoning with the loop's pass count on that item.

**New gates (goal mode, paired bootstrap as before):**

- **L1 — Qwen-driven moves beat the static map:** `loop` − `head` lower bound > 0.
- **L2 — the loop beats the lookup:** `loop` − `fixed_avg` and `loop` − `nearest_min`, both > 0.
- **L3 — breathing earns its keep:** `loop` − `loop_fixed` > 0. `loop` − `loop_nores` reported.
- **L4 — the loop beats the model's own tokens:** `loop` − `native` and `loop` − `reason_loop`, both > 0.

**Free mode:** **LF** — `loop` − `loop_fixed` associate rate lower bound > 0. `loop` − `head`,
`loop` − `loop_nores` and `loop` − `native` reported.

P0, F0, F1, H1, H2 and the kill rule for the static head stand unchanged. Descriptive addition: for
every walk, whether the solution was the **first** spoken word.

Prediction (Claude), before the rerun: L1 passes (the loop finds compound partners the static map
cannot); L4 still fails against `native`; L3 is open.
