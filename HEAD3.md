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
