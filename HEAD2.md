# Head experiment 002 — the knobs, measured (free mode only)

Protocol recorded before any run, 25 November 2026 (Claude, Opus 5.5), after Experiment 001
(`HEAD.md`), which stays a failed experiment. Nothing here rescues its scores. Gates, conditions and
constants are frozen here; changes after seeing results go in a dated amendment.

## Why

Experiment 001 left four open facts:

- When Qwen's own upper layers propose the next thought (the closed loop), trails match human
  associations far better than walks on the static map (+0.27).
- The regulator latched. Movement per step was flat from k = 4 to 32 and large only at k = 1, so
  "stable → zoom out" and "jumpy → zoom in" are both positive feedback. Loop trails latched at k = 1
  scored 0.58 and those latched at k = 32 scored 0.28 (rank correlation of mean k with score: −0.48).
- The voice lagged: with α = 0.5 the old word keeps the largest weight and is spoken again
  (26–36% of steps repeated).
- **How widely the head looks (k) and how far it moves (α) were one coupled knob** (Sol). Averaging
  more neighbours does not widen the view. It pulls the thought toward the centre of what is already
  near.

Experiment 002 measures the two knobs separately and asks one question before any regulator is
trusted: **is there anything for a regulator to do?** Remote associates are not part of this experiment.

## Fixed setup

Same model (`Qwen/Qwen3-1.7B`, float32), pool (8000 generic words), frames, states, clamp band
(−1, 12), voice, fatigue (F = 6), pruning rule (amendment 1 of `HEAD.md`) and T = 12 steps as the
`loop` engine of Experiment 001. Changes:

- **Residue off everywhere** (β = 0, ρ = 0). It is not part of this test.
- **Clean words.** Calibration words, start words and demo words are lowercase pool words that are
  also cues in the association norms (no `conf`, `trab`, `uint`). 20 calibration words (seed 7),
  then 60 start words (seed 11) from the rest. `--norms` is required.
- **Static engine dropped.** Every walk is the closed loop: Qwen proposes from the association
  frame, the top k words (after masks) are weighted by Qwen's own probabilities, the thought moves
  a fraction α toward them.

## Conditions

| condition | k | α |
|---|---|---|
| `k{1,2,4,8,16,32}_a0.5` | fixed | 0.5 |
| `k{1,2,4,8,16,32}_a0.85` | fixed | 0.85 |
| `flip_a0.5`, `flip_a0.85` | regulated, **corrected sign**: if surprise < lo or the spoken word repeats, k halves (zoom in: accelerate); if surprise > hi, k doubles (zoom out: brake). Start k = 4. Band = 25th–75th percentiles of `k4` at the same α on the calibration words | 0.5 / 0.85 |
| `randk_a0.5`, `randk_a0.85` | k drawn uniformly from {1, 2, 4, 8, 16, 32} at every step (seed per start) | 0.5 / 0.85 |
| `discrete` | Qwen's top word (after fatigue and own-word masks) becomes the whole next thought: k = 1, α = 1. No blend of states | 1 |
| `native` | Qwen's own token chain, as in Experiment 001. Reference only | – |

`k4_a0.5` is Experiment 001's `loop_fixed` without residue.

## Measurements

Per trail, with the norms, a step a → b is a **human associate** if b is a response to cue a.

- **Primary score: the collapsed associate rate.** Consecutive repeats of the spoken word are merged
  before scoring (river → river → sea is scored as river → sea), then: associates / steps whose first
  word is a norm cue. This removes the voice lag. Ping-pong (cake → pie → cake) is not removed, so it is
  always read beside the next two.
- Raw associate rate (as in Experiment 001), strength, repeat fraction, distinct words per trail,
  back-tracks.
- **Convergence across trails:** share of all spoken words that occur in 5 or more different trails,
  and unique words / all words over the 60 trails.
- **Trace of the start** (descriptive, Sol's question): cosine between the thought at the last step
  and the start word, in the static centred geometry.

**Control surface (Sol).** 200 saved thoughts: the thoughts of `k4_a0.5` trails at steps 3, 6 and 9
(first 60 trails, 180 thoughts) and 20 of the start words themselves. From each, one step under every
(k, α) of the grid with no fatigue. Recorded: movement (1 − cosine between the thought before and after)
and the entropy of Qwen's next proposal after the move (Neon's candidate sensor). Descriptive: it shows
what each knob does causally, from identical states.

## Gates

Paired bootstrap over start words, 4000 draws. When a comparison picks the best of the 12 grid
cells, the interval is widened for that choice (Bonferroni, 99.58% two-sided).

**G1 — headroom: does the best setting depend on the trail?** Split every trail into steps 1–6 and
6–12. For each start, choose the grid cell with the best collapsed rate on the first half; choose
also the single cell that is best on average over all starts on the first half. Score both on the
**second half**. PASS if per-start choice − global choice has a 95% lower bound > 0. This is what any
regulator would have to exploit: earlier behaviour of the same trail predicting the best setting. The
in-sample best cell per start is reported, but not gated (the maximum of 12 noisy numbers is biased
upwards).

**G2 — the corrected regulator.** Evaluated at the α whose `k4` cell has the higher collapsed rate
(chosen by the fixed grid, not by the regulator's own result). PASS if `flip` − `k4` and `flip` −
`randk` at that α both have 95% lower bounds > 0. `flip` − the best grid cell is reported.

**G3 — does blending states add anything over choosing words?** PASS if the best grid cell (by mean
collapsed rate) − `discrete` has a Bonferroni lower bound > 0. If it fails, the latent mixture adds
nothing to association that a fatigued word chain does not.

**Reported, not gated:** α effect at each k (does α = 0.85 cure the voice lag?), `native` against
everything, the control surface, the start trace.

## What each outcome means

- **G1 FAIL:** in free association, one fixed setting is as good as choosing per trail. No regulator can
  beat it, whatever its sign. Breathing is dead in this mode; the right head is a fixed one.
- **G1 PASS, G2 FAIL:** there is something to regulate, but surprise is not the signal that finds it.
- **G1 and G2 PASS:** the first evidence that the head regulating its own grain helps.
- **G3 FAIL:** for this task, the head is a word chain and the latent blend is decoration.

Prediction (Claude), before the run: G1 fails or passes narrowly; the best cell is k = 1 or 2 at
α = 0.85; G3 fails (`discrete` ties the best blend). `native` stays ahead of every walk.

## Run

```
python head2.py --self-test
python head2.py --device cuda --norms "strength.SWOW-EN.R123.20180827.csv"
```

Output: `runs/head2_<model>/REPORT.md` and `head2.json`. About 45,000 forward passes, roughly an hour on
an RTX 3060 at the pace of Experiment 001.

## Result — Qwen3-1.7B, float32, 25 November 2026

Full report: [`runs/head2_Qwen_Qwen3-1.7B/REPORT.md`](runs/head2_Qwen_Qwen3-1.7B/REPORT.md).

**G1 FAIL**: per-start choice − one global setting, held out: **−0.079 [−0.138, −0.018]**. Choosing
per trail is *worse* than one setting. The best setting on first halves and overall is the same cell,
`k1_a0.85`. **G2 FAIL**: flip − k4 −0.012 [−0.046, +0.021], flip − randk +0.005 [−0.046, +0.059],
flip − best cell −0.150. **G3 FAIL**: `k1_a0.85` − `discrete` −0.010 [−0.048, +0.028].
All three predictions were right.

### What it shows

1. **The grid is monotone.** Sharper is better at every α, and moving further is better at every k.
   Collapsed rate from 0.445 (k = 32, α = 0.5) up to **0.773 (k = 1, α = 0.85)**. α = 0.85 removes the
   voice lag (repeats 0.000–0.062, against 0.044–0.401 at α = 0.5).
2. **The best head is Qwen's own association chain.** `k1_a0.85` 0.773, `discrete` 0.783, `native`
   0.780. None of these differences is outside noise (native − best cell +0.007 [−0.044, +0.061]).
   `k1_a0.85` and `discrete` share their first 5.2 words on average and give the identical trail on
   14 of 60 starts. Closing the loop through real states, sharp and fast, reproduces what the model
   already does with tokens. It does not improve on it.
3. **The knobs now work causally and in the expected directions.** From 200 identical saved
   thoughts: movement falls with k (α 0.85: 0.495 at k = 1 → 0.313 at k = 32) and rises with α. The
   entropy of Qwen's next proposal rises with k (1.73 → 2.04 nats at α 0.5): a wider blend makes the
   model less sure of the next word. The sign-corrected regulator no longer latches; it breathes over
   the whole range (mean k 7.1 at α 0.85). It just breathes around the wrong place: holding *change*
   in a band is not what association quality needs.
4. **Widening buys variety across trails, at a cost in each step.** Unique words across the 60
   trails: `k1_a0.85` 0.536, `discrete` 0.506, against 0.64–0.65 for k = 2–4 at α 0.85, `flip` and
   `randk`. The pure word chain converges most (26.8% of its words are shared by 5 or more trails;
   light, bright, dim). This is the trade-off a sampling temperature gives on tokens, and it may be
   no more than that. It was not gated, and it was not compared with temperature sampling of the
   token chain.
5. **The start fades fast.** Cosine with the start at step 12 is 0.06–0.20 in every condition, highest
   where the thought moves least. Nothing here keeps an early thought alive (Sol's question stays open).

### Caveats found after the run

- **k also sets how many words are fatigued per step** (k words for 6 steps each), so in the trail grid
  a wide k both averages more and masks more destinations (Sol). The control surface, one step from
  identical thoughts with no fatigue, is free of this. The grid is not a pure test of averaging.
- The variety gain from wider k (unique share 0.64–0.65 against 0.51–0.54) was not compared with any
  other way of buying variety, such as sampling the word chain at a temperature. Until it is, it is not
  evidence that blending does something sampling cannot. Highway share is noisy across cells (at α 0.5:
  0.071, 0.142, 0.039, 0.149, 0.101, 0.250 for k = 1…32).
- The start trace is highest where the thought moves least (α 0.5, wide k). A head that never moved would
  score 1.0, so it is not memory.

### Verdict

In free association, the head's grain has nothing to regulate that a trail's own first six steps can
find. One fixed setting (sharpest, fastest) is best, choosing per trail from the first half loses, and
that best setting is the model's own word chain written in states instead of tokens. (G1 tests one
family of policies, choosing from earlier behaviour of the same trail; it does not rule out every
adaptive policy. Sol's correction.) Breathing as a way to make a frozen model associate better is closed by
experiments 001 and 002. What remains is a measured fact about the knob: k behaves like a temperature
(wider: less certain next step, more varied trails, weaker single steps).
