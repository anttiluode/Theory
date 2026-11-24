# Map before reader

*What a frozen model and a model in training show about how word categories are stored and read.*

Antti Luode (PerceptionLab), with Claude (Opus 5.5), Astra (ChatGPT) and Neon (Gemini). 24 November 2026.

Status: exploratory. One word set (8 families, 36–46 single-token words), one seed per model, two models: Qwen3-1.7B and Pythia-410m. Every number below is in a JSON file under [`runs/`](runs/), and every script is in this repo. Methods and the pre-registered criteria are in the [README](README.md).

## Summary

1. **In Qwen, category lives in the family average.** A fake word made of the family average plus random noise of the right size reads as the category as strongly as a real member.
2. **The prototype effect depends on how the model is asked.** Moving a word's internal state toward the average of its siblings raises category evidence for 46/46 words in Qwen's chat format, against 2/46 for a random move of the same length. In a plain few-shot format it is 26 against 20. It is not explained by normalization.
3. **Word states behave like a face space.** Shrinking a word's own deviation from the average raises typicality. Halving it keeps identity (45/46) while giving 86% of the category gain, the anti-caricature. A word's real deviation costs far less category than a random deviation of the same length.
4. **When Qwen reads two words itself, the question decides what is noise.** In "oak, pine", swapping pine for elm moves the category answer 1.2 nats and the identity answer 36 nats. Both identities stay available (retention 0.94).
5. **In training, the map comes before the reader, and the reader never catches up.** In Pythia-410m, kingdoms form before families (step 512 against step 1000). By step 1000 an outside reader with nothing to tune finds the right family for 78% of held-out words. It places families it has never seen into the right kingdom 86% of the time, and links words to their category's own name 69% of the time. Pythia's own yes/no answers are at 8%, and peak at 58% at the end of training.

## 1. Setup

**Words.** Eight families in three kingdoms:
- plant: tree, vegetable;
- animal: bird, fish, insect;
- made by people: tool, vehicle, instrument.

Up to six single-token words per family. Words that are not one token for a model are dropped.

**Reading.** Evidence is yes-minus-no log-odds, calibrated against the neutral item "thing". The questions are "Is it a tree?" (category), "Is it a plant?" (kingdom) and "Is it an oak?" (identity).

**Editing.** A word's state is overwritten at its own position, from the token embedding through layer 12 of 28 (Qwen) or the matching band of Pythia, with nothing left to bypass the edit.

**Two formats.**
- *Few-shot*: plain text with worked examples; it works on base models.
- *Chat*: Qwen's chat template, zero-shot, "Is the item above a …? Answer yes or no."

## 2. Category lives in the family average (Qwen)

The fake sibling is the family average plus a random deviation of the same size as a real sibling's.
- **Few-shot.** The fake reads as the category at 15.7 nats against 15.9 for real words. Blends with it overshoot as much as blends with real siblings (difference −0.01 [−0.10, +0.07]).
- **Chat.** Same result: +2.68 against +2.87 (difference +0.20 [−0.17, +0.64]).

What a word holds beyond its family average adds nothing that sums across siblings.

## 3. The prototype effect is a property of the reader (Qwen)

| | few-shot | chat |
|---|---|---|
| category rose: 75% toward siblings / same-length random move | 26 / 20 of 46 | **46 / 2** of 46 |
| sibling − random, nats [95% CI] | +0.82 [+0.39, +1.27] | **+9.38 [+6.89, +12.31]** |
| 50/50 sibling blend above both words | 21 / 110 | **88 / 110** |
| overshoot vs. normalization gain (Spearman, p) | −0.01, 0.92 | +0.11, 0.28 |

- **The overshoot has no geometric explanation here.** A blend's renormalization gain predicts nothing in either format.
- **The reader's slope is what differs.** In chat, a random step away from the family average costs 4.6 nats, and the same length taken off the word's own deviation gains 4.75. In few-shot the random step costs 0.54, about nine times flatter.
- **Caveat.** States are captured in each format's own context, so this is not strictly "same states, different reader".

The effect is the classic prototype effect of category learning: a never-seen average is classified as well as or better than its exemplars (Posner and Keele 1968).

## 4. A face space for words (Qwen, chat)

The siblings' average plus λ times a deviation. The category change is measured against the word itself.

| λ | word's own deviation: category | identity | random deviation: category | identity |
|---|---|---|---|---|
| 1.5 (caricature) | −7.04 | 46/46 | −13.91 | 5/46 |
| 1 | 0 (the word) | 46/46 | −4.20 | 1/46 |
| 0.5 | **+4.09** | **45/46** | +2.01 | 4/46 |
| 0.25 | +4.75 | 9/46 | +3.78 | 4/46 |
| 0 (average) | +4.55 | 5/46 | +4.55 | 5/46 |

- **Typicality rises toward the average, but distance is not all that counts.** A word's real deviation costs about half as much category as a random one of the same length. It lies mostly in directions the category reading tolerates.
- **Identity is a threshold.** It survives to λ = 0.5 and collapses between 0.5 and 0.25.
- **This is norm-based coding as in face perception.** Faces are stored as deviations from an average face, identity is read along the deviation, and caricatures stay recognisable but read as less typical (Valentine 1991; Leopold, Bondar and Giese 2006 for single neurons in monkey IT).

## 5. The model's own fork keeps identity and lets the question choose (Qwen, chat)

**Design.** Nothing is overwritten. Qwen reads "Items: a, b" normally, and all inputs have the same length:
- oak, oak and pine, pine (each identity repeated);
- oak, pine and pine, oak (same category, different identities);
- oak, salmon (different categories).

Results, over 110 sibling pairs:
- **Identity kept.** "Is one of the items a pine?" gets yes whenever pine is present (220/220) and no for 185/220 absent items. Retention against the repeated pair is 0.94 (median).
- **Query dependence.** Swapping the second sibling for another moves the category answer by 1.21 nats (median) and the identity answer about the swapped word by 36.19 nats; identity moved more in 826/840 swaps. Swapping in another kingdom moves the category answer 26.97.
- **Overshoot, but a different one.** "oak, pine" reads above the line between the repeated pairs by +1.27 [+0.33, +2.22], above both for 64/110 pairs. Pair by pair it does not correlate with the overshoot of the manufactured blend (Spearman 0.00). Word order moves the category answer more than this (3.13 nats).
- **Where the pooling happens: not found.** The answer position puts under 1% of its attention directly on the two words, so it must read them through intermediate positions.

**Reading.** An average written into a slot destroys identity: only 9/46 survive at 75%. The model's own reading keeps every identity and suppresses the difference only when the question is about the category. What counts as residue depends on the question.

## 6. In training: coarse first, then the map is fixed (Pythia-410m)

Twelve checkpoints, step 0 to 143000, with 36 words in 8 families. "Excess" is the share of the spread at each tree level above 200 random relabelings; z is in brackets. "Mid" is layer 12 of 24.

| step | mid kingdom | mid family | embedding kingdom / family z |
|---|---|---|---|
| 128 | +0.002 (0.4) | −0.003 (−0.3) | 0.5 / 0.0 |
| 512 | **+0.052 (5.5)** | +0.022 (1.5) | 1.0 / 0.5 |
| 1000 | +0.133 (13.2) | **+0.073 (4.4)** | **4.9** / 2.2 |
| 2000 | +0.120 (13.2) | +0.097 (6.3) | 9.6 / 5.8 |
| 4000 | +0.103 (13.1) | +0.101 (8.3) | 12.2 / 8.0 |
| 143000 | +0.113 (14.6) | +0.092 (7.5) | 13.1 / 9.0 |

- **Kingdom before family, at every depth.** That was the pre-registered prediction, after Saxe, McClelland and Ganguli (2019). It held in both word sets.
- **The layers' tree is fixed by about step 2000,** 1.4% of training.
- **The embedding table gets the tree several thousand steps later.** Early on, the first blocks *compute* the tree from the word, because nothing else in the context differs between words. Only later is it written into the embeddings.

## 7. Map before reader (Pythia-410m)

Four readers of the same word states, at layer 12 unless marked. Chance is 0.125 for 8 families and 0.33 for 3 kingdoms.
- **Prototype.** Leave one word out, take family centroids from the rest, and pick the closest by angle. It uses labelled examples.
- **Unseen family.** The kingdom of a word whose whole family is removed before the centroids are built.
- **Category word.** No labels: the family whose own name ("tree", "bird"), read in the same frame, lies closest.
- **Native.** Pythia's own yes/no answer to "Is it a tree?" etc.

| step | native | prototype | unseen-family kingdom | category word |
|---|---|---|---|---|
| 128 | 0.08 | 0.06 | 0.36 | 0.17 |
| 512 | 0.08 | **0.44** | 0.50 | 0.22 |
| 1000 | **0.08** | **0.78** | **0.86** | **0.69** |
| 2000 | 0.19 | 0.86 | 0.81 | 0.83 |
| 4000 | 0.58 | 0.94 | 0.89 | 0.92 |
| 16000 | 0.42 | 0.97 | 0.92 | 1.00 |
| 143000 | 0.39 | 0.97 | 0.89 | 0.97 |

- **The order is map, then names, then answers.**
  - At step 512 the clusters are usable (0.44), while the category words are not yet near them (0.22).
  - By step 1000 all three outside readers work. Clusters 0.78; never-seen families placed in the right kingdom 0.86; clusters next to their own names 0.69, reaching 1.00 by step 16000.
  - The native reader is at 0.08.
- **The gap is in answering, not in knowing or naming.** At the end of training, Pythia's states sit next to the right category word 97% of the time, but its yes/no answer is right 39% of the time. For this base model, the bottleneck is using the knowledge in a question format, and it never closes.
- **Kingdom is a real abstraction early.** With the family removed, kingdom accuracy at step 1000 is 0.86 against 0.89 with it. The coarse level is not just recognition of a familiar cluster.
- **Neon's observation.** Moving a word toward its siblings does nothing at step 4000: 0.58 → 0.56. At the end it lifts accuracy from 0.39 to 0.83. So the native reader keeps getting better on the prototype while getting worse on real words; what hurts is the words' own deviations.
  - The size of those deviations does not grow: the leaf share at layer 12 is 0.59–0.60 from step 1000 to the end.
  - So if they hurt more, it is their *direction* relative to the reader, not their size.
  - Repeating §4's deviation curve at each checkpoint would test this.

**The developmental parallel, kept honest.** Infants show category structure in visual cortex at two months (*Nature Neuroscience*), and understand common nouns at 6–9 months, before they speak them (Bergelson and Swingley 2012). In human visual cortex, forward connections (V1→V2) are mature by 4 months of age, while the feedback connections are still immature (Burkhalter, Bernardo and Charles 1993). Map first, reader later fits all of that. Pythia shows the same order in a system where both can be measured directly. That makes the order *possible* without any developmental program, but it does not show the brain does it for the same reason.

The idea that more complex brains take longer to learn to read themselves (moose against human) is not tested here. Pythia makes a clean test, because its sizes share the same data order. One caution goes with it: larger language models usually learn *faster* per token. A delay in the reader that grows with size would therefore be a real finding, not an expected one.

## 8. What is known and what may be new

**Known:**
- wavelet bases of hierarchical statistics, and coarse-first learning in networks (Saxe et al. 2019);
- the symmetry argument for model geometry (Karkada et al. 2026);
- prototype effects (Posner and Keele 1968) and norm-based face coding (Valentine 1991; Leopold et al. 2006);
- coarse-to-fine category learning in vision models trained on child's-view video (NeurIPS 2026).

**Possibly new, as far as we have searched:**
- the prototype and anti-caricature curves measured by moving a language model's own internal states;
- that this curve depends on the question format;
- the model's own two-word reading keeping identity while suppressing sibling differences only under the category question;
- the order kingdom → family → names → answers measured by tree level and by reader across Pythia checkpoints, including the embedding lag.

**Not claimed:** anything about dendrites or brains beyond analogy, and any engineering use. The anti-caricature trade-off is one measurement in one model.

## 9. Limitations

- One seed per model, one word set of 36–46 words, two models. Families with 3 words are thin.
- Pythia-410m is a small base model, and its native answers are low-margin (calibrated evidence near 0 nats).
- In Qwen, the two formats differ in context as well as reader.
- fork.py has not located where the pooling happens. Its attention measure looked at the wrong position.
- The four readers in §7 differ in what they are given (labels, names, a question), so "the gap is in answering" is a statement about these readers.

## 10. Next

1. §4's deviation curve at every Pythia checkpoint: does the reader turn against real deviations late in training?
2. Pythia-160m, 410m and 1b, with the same data order: when do the map, the names and the answers each appear, and does the gap close with size?
3. fork.py with attention from every position after the words, then interventions on the heads found.
4. A second seed, or a second word set, for the headline numbers.

## Files

| claim | file |
|---|---|
| §2–4 | [`runs/Qwen_Qwen3-1.7B/chord.json`](runs/Qwen_Qwen3-1.7B/chord.json), [`runs/Qwen_Qwen3-1.7B_chat/chord.json`](runs/Qwen_Qwen3-1.7B_chat/chord.json) |
| §5 | [`runs/Qwen_Qwen3-1.7B_chat/fork.json`](runs/Qwen_Qwen3-1.7B_chat/fork.json) |
| §6 | [`runs/timeline_EleutherAI_pythia-410m_min3/timeline.json`](runs/timeline_EleutherAI_pythia-410m_min3/timeline.json) (and the first 30-word run) |
| §7 | [`runs/access_EleutherAI_pythia-410m_min3/access.json`](runs/access_EleutherAI_pythia-410m_min3/access.json) |
