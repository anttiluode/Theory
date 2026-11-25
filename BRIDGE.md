# Bridge — can a label-free PCA knob repair the frozen model's own answers?

Protocol recorded before any real-model run, 25 November 2026 (Claude, Opus 5.5).
Thresholds below are frozen; changes after seeing results go in a dated amendment.

## Why

Two results from the same morning meet here.

- **Maatuska** (anttiluode/Maatuska): in Pythia-410m, Pythia-1b and Qwen3-1.7B, the top principal
  components of an *unlabeled* vocabulary carry more family structure than random directions of
  the same width. The advantage peaks at about 16 components. This is a label-free
  coarse-to-fine knob, but so far it has only been measured, never used.
- **Iris** (this repo, `IRIS.md`): blending a word's states with its 3 nearest neighbours from a
  word bank raised Pythia-410m's native family answers from 42/124 to 56/124, with no weight
  update. That repair is **transductive**: it needs the word's siblings to be in the bank.

The bridge asks whether the **inductive** knob can do the repair: a PCA basis fitted once on
generic vocabulary, with no neighbour bank at test time. If it can, it works on any word.

## Substrate and words

Exactly Iris: Pythia-410m at `step143000`, Theory's few-shot frame, Iris's 20-family bank
(`iris.BANK`, `iris.EXCLUDE`, single-token filter), the same neutral-calibrated family
questions, and the same clamp. The clamp writes a target state at the word position at every
band layer from the token embedding through `band()`, using Theory's `blend` mechanics.

## Label-free basis

At each band layer, PCA of the word-position state for up to 3000 single-token vocabulary
words from the model's own tokenizer, captured in the same frame. All Iris bank words and
all few-shot example words are excluded. No labels are used anywhere in the basis.

## Edits (all fixed in advance)

Each edit replaces the word's state s at each band layer by s + λ(T(s) − s).

| name | T(s) | λ | role |
|---|---|---|---|
| `pca16_l0.5` | μ + P₁₆(s − μ), top-16 PCA prefix | 0.5 | **primary** |
| `pca16_l1.0` | same | 1.0 | strength |
| `pca64_l0.5` | top-64 prefix | 0.5 | width |
| `lidpre16_l0.5` | components 2–17 (lid lifted, then prefix) | 0.5 | lid + prefix |
| `lid1_l1.0` | s minus its top-1 PCA component | 1.0 | lid only |
| `lid3_l1.0` | s minus its top-3 components | 1.0 | lid only |
| `rand16_l0.5` | μ + R₁₆(s − μ), random 16-dim subspace | 0.5 | subspace control |
| `pca16_l0.5_matched` | random direction, same move length as primary at each layer | — | movement control |
| `nn3_l0.5` | Iris's 3-NN bank blend | 0.5 | transductive reference (reproduces Iris) |
| `nn3_random_l0.5` | 3 random bank words, same blend | 0.5 | Sol's requested control |

Here μ is the vocabulary mean at that layer and P_d projects onto the top-d PCA directions.

## Questions answered for every edit

- **Category**: argmax over the 20 neutral-calibrated family questions (as Iris).
- **Identity**: argmax over "Is it a/an {x}?" for every member x of the word's own family,
  neutral-calibrated. This is where coarsening should hurt, so "always edit" pays a price.

## Gates

**B1 — inductive repair (primary edit only).** PASS if the category accuracy of `pca16_l0.5` is:

- at least native + 5 percentage points, **and**
- at least 3 points above `rand16_l0.5`, **and**
- at least 3 points above `pca16_l0.5_matched`, **and**
- with the paired bootstrap 95% lower bound (over words, 4000 draws) of
  (primary − `rand16_l0.5`) above 0.

**B2 — cost.** Report the identity accuracy change under the primary edit. Call it a "cheap
repair" if the identity drop is at most 5 points. It is not a pass/fail gate.

**B3 — versus transductive.** Report primary − `nn3_l0.5` on category, with a bootstrap
interval. Descriptive.

All other rows are descriptive. Choosing the best row after the fact is not a result.

## What would and would not follow

- B1 PASS: a label-free subspace fitted on generic words, written back into the frozen
  model's early-to-middle layers, improves its own category answers beyond equally large random
  moves and random subspaces. This is an inductive repair. It is not yet an adaptive reader;
  that needs a regime where the right knob setting differs (identity versus category).
- B1 FAIL with Iris's `nn3` still winning: global linear coarsening is not enough, and the useful
  operation is local, needing neighbours. The next step would be neighbours found inside the
  PCA prefix instead of the full space.
- Pythia-410m only, one frame, 124 words. Ambiguous words (tin, lead, pound…) remain, as in Iris.

## First real run — Pythia-410m step143000, 25 November 2026

Full table: [`runs/bridge_EleutherAI_pythia-410m_shots_step143000/REPORT.md`](runs/bridge_EleutherAI_pythia-410m_shots_step143000/REPORT.md).
The run crashed while writing REPORT.md on Windows (cp1252 could not encode "−"), after
`bridge.json` was complete. The report was regenerated from that JSON with
`--report-from`. No numbers changed.

**B1 inductive repair: FAIL.** The primary edit (`pca16_l0.5`) scored 0.306 on category against
0.339 native: 5 rescued, 9 harmed. It was not distinguishable from a random 16-dim subspace
(+0.024 [−0.024, +0.081]) or from an equally long random move (−0.008 [−0.073, +0.048]).
**B2:** identity fell 0.815 → 0.750 (costly). **B3:** 14.5 points below the 3-NN blend.

Plumbing check passed: `nn3_l0.5` reproduced Iris exactly (0.452, 15 rescued, 1 harmed).

Descriptive rows that matter:

| edit | category | identity | rescued / harmed |
|---|---:|---:|---|
| native | 0.339 | 0.815 | – |
| `nn3_l0.5` (Iris, transductive) | **0.452** | 0.637 | 15 / 1 |
| `nn3_random_l0.5` (3 random bank words) | 0.234 | 0.750 | 2 / 15 |
| `pca16_l0.5` (primary) | 0.306 | 0.750 | 5 / 9 |
| `pca64_l0.5` | 0.339 | 0.734 | 6 / 6 |
| `lid1_l1.0` | 0.339 | 0.823 | 1 / 1 |
| `pca16_l1.0` | 0.210 | 0.355 | 4 / 20 |

What this says:

1. **Iris's repair is semantic.** Blending with 3 *random* bank words harms 15 answers and
   rescues 2, while the 3 nearest neighbours rescue 15 and harm 1, at the same blend strength.
   This settles the open control in IRIS.md.
2. **Iris's repair has a price.** It costs 18 points of identity (0.815 → 0.637). Category goes
   up, identity goes down, so "always repair" is no longer free. A regulator now has
   two regimes to choose between, which is the situation it needs.
3. **The global label-free knob does not repair when written back.** Maatuska shows that a
   16-component PCA prefix *reads* family neighbourhoods better than random directions. Forcing
   the word's own state halfway onto that prefix inside the forward pass behaves like any
   random displacement of the same size. Neighbours work because they move the state toward
   other real word states, which the later layers know how to read. A projection moves it
   toward a shadow that no real word has.
4. Lifting the lid alone (`lid1`) does nothing in this band (layers 0–10). That matches
   Maatuska, where Pythia's lid matters in the upper layers, not in the clamp band.

Next, not yet run: use the PCA prefix only to **choose** the neighbours (where Maatuska
says it is better than full width), and keep blending real full states. Done once with the
bank and once with the 3000 unlabeled vocabulary words as the neighbour pool. The second is the
inductive version: no siblings are guaranteed to be there.
