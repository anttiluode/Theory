# Head experiment 003 — three cues kept apart (remote associates, no loop)

Protocol recorded before any run, 25 November 2026 (Claude, Opus 5.5). Gates and constants are frozen
here; changes after seeing results go in a dated amendment.

## Why

Experiment 001 blended the three cues of a remote associate problem into one state. A blend behaves as
a winner-take-most mixture (a 50/50 blend is read as one of the two words 92.5% of the time), so one cue
took over and every condition sat at 0–5%. Qwen3-1.7B asked directly solved 6/130.

Sol's next head keeps what it is looking for separate from what it looks at, and then decides. The
simplest version of that has no loop at all:

- **What is looked for:** the three cues, each kept apart.
- **What is looked at:** each cue's own association distribution from Qwen, p(w | cue).
- **The decision:** the word that fits all three.

If the cues are independent given the answer, Bayes' rule gives

  p(w | c₁, c₂, c₃) ∝ p(w | c₁) · p(w | c₂) · p(w | c₃) / p(w)²

so the score is Σⱼ log p(w | cⱼ) − λ · log p₀(w) with **λ = n − 1 = 2** (Neon's correction for words that
every cue likes: light, water, day). This experiment asks whether Qwen's separate associations already
contain answers its single prompt misses.

## Fixed setup

- `Qwen/Qwen3-1.7B`, float32, greedy, no clamping, no states.
- **All 144 items** of Bowden & Jung-Beeman (2003), `data/rat144.tsv`. Cues may be any number of tokens
  (they are only read, never written into a slot), so cottage / swiss / cake is back.
- **Association frame:** `Item: lemon. Association: sour` / `needle → thread` / `doctor → nurse`, then
  `Item: <cue>. Association:`. None of the six shot words occurs in any item. One forward pass per cue;
  the next-token log-probabilities are restricted to the candidates and renormalised over them.
- **Candidates:** single tokens that are a leading space plus 3 or more lowercase letters, and whose word
  occurs in the SWOW norms (as a cue or a response). SWOW is free association, not remote associates,
  so it carries no answer key; it only removes sub-word fragments. Candidates that equal a cue, are a cue
  plus "s", or that begin with a cue or are the beginning of one, are excluded for that item.
- **Base rate p₀:** the average (in probability, then log) of the candidate distribution over 200 words
  drawn (seed 7) from SWOW cues that are single lowercase tokens and occur in no item.
- A correct answer is scored as in Experiment 001 (lowercase, plural s/es either way, school/court both).

## Conditions

| condition | score for word w | passes per item |
|---|---|---:|
| `native` | Qwen asked directly, 3 shots, cue first tokens banned (as in Experiment 001) | ≤ 3 |
| `prod_raw` | Σⱼ log p(w \| cⱼ) (λ = 0) | 3 |
| `prod_l1` | λ = 1 | 3 |
| `prod_pmi` | **λ = 2 (gated)** | 3 |
| `mix` | log Σⱼ p(w \| cⱼ) − log p₀(w): the union with the same correction (what a blend does at the level of distributions) | 3 |

The 200 base-rate passes are a one-time cost shared by all items.

## Gates

Paired bootstrap over the 144 items, 4000 draws, 95%.

**G1 — the separate associations hold answers the prompt misses.** `prod_pmi` − `native` accuracy:
lower bound > 0.

**G2 — intersection beats union.** `prod_pmi` − `mix`: lower bound > 0.

**Reported, not gated:** λ = 0 and 1; top-10 hit rate and median rank of the solution for every scoring
condition; how many solutions are candidates at all; the per-cue rank of the solution (which cue
"knows" it); Spearman correlation between the human 15-second solve rate and the rank of the solution
under `prod_pmi` (does the model find hard what people find hard).

**Kill.** If G1 fails, Qwen3-1.7B's associations do not hold remote associates in a form an intersection
can reach, and the remote-associates line ends for this model; Sol's per-cue head is not built on it. If
G1 passes, a per-cue head (each cue its own iris, one decision) has a baseline to beat: `prod_pmi`.

Prediction (Claude), before the run: G1 passes (`prod_pmi` 10–20% against native 4–5%), G2 passes, the
solution is in the top 10 for about a third of the items, and the λ = 0 product loses to λ = 2 because
light/water/day win it.

## Run

```
python head3.py --self-test
python head3.py --device cuda --norms "strength.SWOW-EN.R123.20180827.csv"
```

About 1,100 forward passes: a few minutes. Output: `runs/head3_<model>/REPORT.md` and `head3.json`.

## Result — Qwen3-1.7B, float32, 25 November 2026

Full report: [`runs/head3_Qwen_Qwen3-1.7B/REPORT.md`](runs/head3_Qwen_Qwen3-1.7B/REPORT.md). 16,477 candidates;
the solution is a candidate for all 144 items; 924 forward passes.

**G1 PASS**: `prod_pmi` − `native` **+0.069 [+0.028, +0.118]**. **G2 PASS**: `prod_pmi` − `mix`
**+0.104 [+0.056, +0.160]**. The head line's first passed gate on a task the model fails natively.

| condition | correct | rank 1–3 | top 10 | top 100 |
|---|---:|---:|---:|---:|
| native (asked directly) | 7/144 (4.9%) | – | – | – |
| prod_raw (λ = 0) | 4 | 18 | 34 | 83 |
| prod_l1 (λ = 1) | 13 | 20 | **41** | **85** |
| **prod_pmi (λ = 2)** | **17 (11.8%)** | **23** | 37 | 74 |
| mix (union) | 2 | 2 | 12 | 33 |

### What it shows

1. **Qwen's separate associations hold answers its single prompt misses.** Three plain forward passes
   and a sum of logs solve 17 items against 7. The intersection nearly contains the prompt: 6 of the 7
   native solutions are among the 17 (union 18).
2. **Intersection, not union.** The same three distributions mixed (summed probabilities, what a
   blend does) solve 2. This is the distribution-level form of Experiment 001's finding that a
   blend is a winner-take-most mixture.
3. **The Bayes correction is needed, and λ = 2 is the predicted value.** Without it, words every cue
   likes win (light 8, medical 8, pain 7, water 7) and accuracy falls to 4. λ = 2 is best at rank 1;
   λ = 1 is best for top 10 and top 100: the full correction over-promotes rare words further down the
   list.
4. **Usually one cue does not know the answer.** One cue alone (best cue) has the solution in its top 10
   on 33 items, median rank 37; the worst cue's median rank is 990, and all three cues have it in their
   top 100 on only 10 items. A remote associate is remote from at least one cue for this model. That is
   the job Neon's per-cue iris would have: widen only the cue that disagrees.
5. **The model finds hard what people find hard.** Spearman between the human 15-second solve rate and
   the solution's rank: **+0.367** over 144 items.
6. **The frame leaks.** The shot word *needle* wins 3 items and *medical* (from doctor → nurse) wins 8
   under λ = 0. Shot words were kept out of the items but not out of the candidates. This costs the
   product, it does not help it.

People solve these items 31% of the time in 15 seconds on average; the intersection reaches 11.8% at
rank 1 and 26% (37/144) in its top 10.

### Verdict

Keeping the three cues apart and deciding by intersection is the first version of the head that beats the
model talking, on a task the model fails. Nothing here loops or breathes: it is three readings and one
decision. A per-cue head now has a baseline to beat, at equal forward passes: `prod_pmi`, 17/144, and
top 10 for 37.
