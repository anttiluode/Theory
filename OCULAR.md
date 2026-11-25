# Ocular — choose neighbours through a narrow band, carry full states

Protocol recorded before any real-model run, 25 November 2026 (Claude, Opus 5.5).
Thresholds and the primary condition are frozen here. Changes after seeing results go in a
dated amendment.

## Why

The Bridge run (`BRIDGE.md`) separated two things:

- Writing a word's own state onto a label-free PCA band does **not** help (0.306 vs native 0.339).
- Blending the word with three real neighbour states **does** help (Iris: 0.452), and three
  random words harm (0.234). The repair is semantic, and it costs identity (0.815 → 0.637).

Maatuska's audit showed where the band reads family structure best: under the lid. In
Pythia-410m layer 12, the advantage over random directions is +0.37 for components 1–16 and
+0.56 for components 5–20.

So the question: **does choosing the neighbours through the band, while still carrying their full
real states, repair the frozen model better than choosing them at full width?** And can it work
**inductively**, when the neighbours must come from a generic unlabeled vocabulary instead of a bank
that contains the word's siblings?

The selection space and the transport space are separate. That is the ordinary Q/K versus V
division of attention, built here after training from the model's own activation geometry rather
than learned.

## Fixed setup

Iris's words, frame, neutral-calibrated family questions, identity questions (as in Bridge),
clamp band, and blend: own state 0.5, mean of 3 neighbour states 0.5, written at every band
layer. Only the **way neighbours are chosen** and the **pool they come from** change.

- **Selection layer:** n_layers // 2 (layer 12 of Pythia-410m), as in Iris.
- **Generic pool:** up to 8000 single-token words from the model's tokenizer, lowercase *or*
  title-case, 3 letters or more. All 124 bank words (case-insensitive), the few-shot words and the
  slot word are excluded. This includes Neon's fix: rivers, countries, languages and 3-letter words
  must be able to find peers.
- **Label-free basis:** PCA of the generic pool's states at the selection layer, centred on the pool mean.

## Selection spaces (all cosine, after centring on the generic-pool mean)

| space | coordinates |
|---|---|
| `full` | the whole state (Iris's metric) |
| `pre16` | PCA components 1–16 |
| `band4` | PCA components 5–20 (**primary**, chosen from Maatuska's audit before this run) |
| `band8` | PCA components 9–24 |
| `rand16` | a fixed random 16-dimensional subspace |

## Conditions

`native`; each space × each pool (`bank` = the other 123 bank words, `vocab` = the generic pool);
`iris` = Iris's exact neighbour function (must reproduce 56/124); `random3_vocab` = 3 random
generic words.

**Primary condition: `band4_vocab`.**

## Gates

**O1 — inductive ocular repair.** PASS if `band4_vocab` category accuracy is:

- at least native + 5 points, **and**
- at least 3 points above `rand16_vocab`, **and**
- at least 3 points above `random3_vocab`, **and**
- with the paired bootstrap 95% lower bound (words, 4000 draws) of (`band4_vocab` − `rand16_vocab`) above 0.

**O2 — band beats full width at choosing (inductive).** PASS if the paired 95% lower bound of
(`band4_vocab` − `full_vocab`) is above 0.

**O3 — transductive.** Report `band4_bank` − `iris`, with an interval. Descriptive.

**Reported for every condition:** category, identity, rescued/harmed against native, mean
neighbour coherence (pairwise cosine among the 3 chosen neighbours in the full centred space), and,
for bank conditions only, the fraction of chosen neighbours in the word's true family. Labels are used
for scoring only. Every chosen neighbour is logged per word.

## What would follow

- **O1 PASS:** a frozen model's own answers repaired with no labels, no weight update, no sibling
  bank and no prompt change. The neighbours are drawn from generic vocabulary through a 16-number
  window.
- **O1 FAIL, but bank conditions work:** the repair needs a curated pool, so it is retrieval-like,
  not inductive. Generic-pool dilution (Neon's first cliff) is the likely reason, and the logged
  neighbours will show it.
- **O2 FAIL:** the band reads families better in Maatuska's kNN test but does not choose better
  carriers. Reading quality and causal usefulness come apart.
- Pythia-410m only, one frame, 124 words.

## First real run — Pythia-410m step143000, 25 November 2026

Full table and neighbour samples: [`runs/ocular_EleutherAI_pythia-410m_shots_step143000/REPORT.md`](runs/ocular_EleutherAI_pythia-410m_shots_step143000/REPORT.md).

**O1 inductive ocular repair: FAIL.** `band4_vocab` 0.347 vs native 0.339
(+0.008 [−0.056, +0.073]). It beats three random generic words (+0.081 [+0.024, +0.137]) but not a
random 16-dim window by the preregistered margin (+0.065 [−0.008, +0.137]).
**O2: FAIL.** Choosing through the band is not better than choosing at full width (−0.024 [−0.089, +0.032]).
**O3:** `band4_bank` is 8.9 points below Iris. The `iris` row reproduced 56/124, so the plumbing is right.

### What it shows

1. **The narrow window chooses worse neighbours than the full state, in both pools.** With the
   bank, the share of chosen neighbours in the word's true family is 0.855 at full width, 0.621
   through components 5–20, 0.605 through 1–16, and 0.411 through a random window. Maatuska's result
   holds (PCA beats random at equal width), but equal width was never the comparison that mattered here.
   Maatuska's K3 had already failed: truncating never improved neighbourhoods. That predicted O2,
   and the protocol should have taken it into account.
2. **In an open pool, the band's coarse axes are not the taxonomy.** Through components 5–20, `oak`
   picks *breast, elder, leather* and `salmon` picks *Turkey, carcinoma, Muslim*. Sixteen numbers
   carry several coarse axes at once (anatomy and medicine, food, nationality). Inside Maatuska's
   curated 192 words only the taxonomy could show. Among 8000 generic words, the other axes take over.
   This is PEEL's non-nesting in practice: a short prefix gives a different tree.
3. **Full-width selection from the generic pool finds good neighbours.** Examples: oak → *wood, tree*;
   violin → *instrument, viol*; zinc → *calcium, sodium, silicon*; anger → *angry, anxiety, emotion*;
   Nile → *River, river, Egypt*. Often the category word itself is chosen.
4. **Inductive repair exists, but is small (descriptive, not preregistered).** `full_vocab` is
   +0.032 [−0.024, +0.089] over native and +0.105 [+0.040, +0.169] over random words. It costs only
   5.6 points of identity, against Iris's 17.7.
5. **The harms are not bad neighbours.** `tennis` with *basketball, football, baseball* and `Japan` with
   *Korea, Russia, India* both turn into "profession". This is Pythia's default sink (it answers
   profession for 45/124 words natively). The limit has moved from finding neighbours to the
   small model's answer reader.

The generic pool also contains word-start fragments (*con, pro, whe, res, comp*) because 3-letter
tokens were allowed. Full-width selection avoided them in the samples shown.

**Verdict on the ocular:** as a narrow *selector* it is killed here. The PCA band remains a label-free
compressed reading, better than random at equal width, but the full state chooses carriers better.

## Amendment 1 — Qwen3-1.7B run (recorded 25 November 2026, before running)

After the Pythia run the narrow-window selector is killed (O1, O2), so the Qwen run tests a
different, now primary, question. Does the **inductive full-width** repair, with neighbours from the
generic pool chosen with the full state, help a stronger model that does not have Pythia's
"profession" sink? This is Neon's strong-model cliff. Pythia's +3.2 points on this row were
descriptive. Here the rule is fixed in advance.

- Model `Qwen/Qwen3-1.7B`, few-shot frame (`--frame shots`, as in Pythia and chord.py),
  float32. Same words, pool rule (8000 words), selection layer n_layers // 2 = 14, clamp band
  `band()` = (−1, 12), and blend.
- **Q1 — full-width inductive repair. PASS** if `full_vocab` category accuracy is:
  - at least native + 3 points, **and**
  - at least `random3_vocab` + 3 points, **and**
  - with the paired bootstrap 95% lower bound of (`full_vocab` − native) above 0.
- Identity change under `full_vocab` is reported with its interval. O1–O3 are still computed by the
  same code and reported mechanically.
- Q1 is also printed for the Pythia report, but there it is post hoc, not a test.

Note, 25 November 2026: the first Qwen attempt exited silently while loading the float32 checkpoint
(no traceback, no output). `--dtype` was added so the run can fall back to bfloat16. The protocol is
still float32. Any bfloat16 run is a recorded deviation, and its report prints the dtype.
