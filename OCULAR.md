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
