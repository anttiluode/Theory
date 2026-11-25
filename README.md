# Theory

> **November 25 — Iris:** [protocol and run instructions](IRIS.md) · [`iris.py`](iris.py). Does the final answer-logit nullspace predict when a fixed, label-free neighbor blend will repair the native answer? Whole-family-held-out supervised monitor; matched re-read budgets. First Pythia run: registered gate failed; always-edit 56/124 vs native 42/124, residue router 53/124 with 59 edits; full-state router identical. [Results and limits](IRIS.md#first-real-run--pythia-410m-final-checkpoint-25-november-2026). Independent of Maatuska's multiresolution-axis test.

> **November 25 — Bridge:** [protocol](BRIDGE.md) · [`bridge.py`](bridge.py). Can [Maatuska](https://github.com/anttiluode/Maatuska)'s label-free PCA prefix (fitted on unlabeled vocabulary, no neighbour bank) repair the same 124 Iris answers? Against random subspaces, equally long random moves and Iris's 3-NN blend, with identity questions as the cost side. First run: **B1 FAIL**, the global PCA knob does not repair when written back (0.306 vs native 0.339, same as a random move). The 3-NN blend reproduced Iris (0.452), random neighbours harmed (0.234), and the blend costs 18 points of identity. [Results](BRIDGE.md#first-real-run--pythia-410m-step143000-25-november-2026).

> **November 25 — Ocular:** [protocol](OCULAR.md) · [`ocular.py`](ocular.py). Choose the 3 neighbours through Maatuska's lid-lifted PCA band (components 5–20), carry their full states as Iris does. Bank pool vs an 8000-word generic unlabeled pool (the inductive test). First run: **O1 and O2 FAIL**. The band chooses worse neighbours than the full state (true-family 0.62 vs 0.86); full-width choice from generic words finds good neighbours (oak → wood, tree) and repairs a little (+3 points, CI crosses 0) at a small identity cost. Harms come from Pythia's 'profession' sink, not from bad neighbours. [Results](OCULAR.md#first-real-run--pythia-410m-step143000-25-november-2026). Qwen3-1.7B (preregistered Q1): **FAIL** — Qwen already answers 96% natively, so every edit can only cost; the repair is a remedy for under-read models. [Qwen results](OCULAR.md#qwen3-17b-run-amendment-1-25-november-2026).

> **November 25 — Head:** [protocol](HEAD.md) · [`head.py`](head.py). The four parts of today's runs put into one loop outside frozen Qwen3-1.7B: a thought held as a weighted set of real middle-layer word states, read at full width against 8000 generic words, breathing between one neighbour (zoom in) and the average of many (zoom out), with fatigue, a slow residue, and the ECG coupler's regulator holding change inside a band. Qwen speaks the thought as one word per step. Tested against human free-association norms (free mode) and the 144 Remote Associates problems (goal mode), against a fixed-grain walk, a fixed average, a pure lookup, Qwen asked directly, and Qwen reasoning at equal forward passes. [Amendment 1](HEAD.md#amendment-1--recorded-25-november-2026-after-a-crash-and-before-any-scored-result): a prune bug fixed after the first run crashed (P0 had passed: copy 1.000, blend 0.925), and a second, closed-loop engine where Qwen's own upper layers propose each move (Neon and Sol's critique). [First run](HEAD.md#first-real-run--qwen3-17b-float32-25-november-2026-with-amendment-1): every gate after the sanity check **FAILS**. Qwen-proposed moves beat the static map on human associations (+0.27), but stay far below Qwen's own token chain (0.43 vs 0.78). The regulator had the wrong sign: in this state space averaging calms and a single neighbour jumps. RAT is at floor for a 1.7B model (native 6/130).

> **November 25 — Head experiment 002:** [protocol](HEAD2.md) · [`head2.py`](head2.py). Free mode only, closed loop, no residue. Separates how widely the head looks (k) from how far it moves (α) on a 6 × 2 grid, measures the knobs causally from 200 identical saved thoughts, and asks first whether any regulator could help (held-out per-trail choice vs one global setting), then tests the sign-corrected regulator and a pure word chain. [Result](HEAD2.md#result--qwen3-17b-float32-25-november-2026): **all three gates FAIL**, as predicted. Sharpest and fastest is best at every setting (0.773). Choosing per trail loses to one setting (−0.079). The best head equals a plain word chain (0.783) and Qwen's own chain (0.780). Widening acts like a temperature: more variety across trails, weaker steps. Breathing for free association is closed.

What the sibling-average result in [RakeisuusClaude](https://github.com/anttiluode/RakeisuusClaude) is, mathematically, and the experiments that could kill each part of that explanation.

## Did we find something? The blunt version

**The write-up of everything so far is in [FINDINGS.md](FINDINGS.md).**

**We found a clean, controlled effect, not a discovery.** In frozen Qwen3-1.7B, moving a word 75% of the way toward the average of its siblings raised category evidence for all 22 held-out words. Exact identity fell from 22/22 to 3/22. Moving toward the category word did not do this, and a random move did the opposite. ([note](https://github.com/anttiluode/RakeisuusClaude/blob/main/HIERARCHICAL_GEOMETRY.md))

Most of it is what the known mathematics predicts. Below is that mathematics, and exactly which parts are known and which are not.

**After the first chord.py run (24 Nov 2026, [results](#results-chordpy-on-qwen3-17b-few-shot-frame)):** the shared-plus-private picture held up, and the parts that could have been new did not show up. In the few-shot frame, a fake sibling made of the family average plus random noise reads as the category as strongly as a real word, and blends with it overshoot exactly as much. The "louder kind" shrank to about 1% of the scale and does not follow the normalization gain. One open question remained: whether the louder kind belongs to the chat frame RakeisuusClaude used.

**After the chat-frame run ([results](#results-chordpy-on-qwen3-17b-chat-frame)): it does.** In the chat frame the effect is back in full:
- The sibling move raised category evidence for 46/46 words; the same-length random move raised it for 2/46.
- 80% of sibling blends beat both of their ends.
- It still does not follow the normalization gain, and the fake sibling (family average plus noise) does it just as well.

So what lifts the category is getting closer to the family average, not anything the siblings' private parts hold. Psychology has a name for this, the prototype effect: a never-seen average is classified better than the members it was made from. Part D of `chord.py` measures that curve directly, and `fork.py` asks whether Qwen builds the average itself when it reads two words.

**After part D and fork.py ([D](#part-d-the-typicality-curve-chat-frame), [fork](#results-forkpy-on-qwen3-17b-chat-frame)):**
- *Distance from the prototype is not the whole story.* A word's own private part costs far less category than a random one of the same length, and halving it keeps identity (45/46) while giving most of the category gain. This is the anti-caricature of face-space research.
- *When Qwen reads "oak, pine" itself,* it keeps both identities (retention 0.94). Under the category question it is about 30 times less moved by swapping pine for elm than the identity question is. One input, two different residues, depending on the question.
- Its own overshoot is smaller (+1.3 nats) and does not track the manufactured blend pair by pair.

**After access.py ([results](#results-accesspy-on-pythia-410m)): the map comes before the reader, and the reader never catches up.** At step 1000, a prototype reader with nothing to tune classifies held-out words into the right family 78% of the time (chance 12.5%). Pythia's own answers are at 8%. From step 4000 to the end the outside reader stays at 94–97% at the middle layer, and the model's own answers never pass 58%.

**After the second access run:** the outside reader also places never-seen families into the right kingdom (86% at step 1000). It links words to their category's own name ("tree", 69% at step 1000, 97–100% later) with no labelled examples. So Pythia knows and names its categories long before, and long after, it can answer a yes/no question about them. The order is map, then names, then answers. Details are in [FINDINGS.md §7](FINDINGS.md#7-map-before-reader-pythia-410m).

**After the first timeline run ([results](#results-timelinepy-on-pythia-410m)):** coarse came first. In Pythia-410m the kingdom level appears at step 512, the family level at step 1000, and both are fixed by step 2000, about 1.4% of training. The token embeddings get the same tree roughly ten times later. One caveat matters: trees and birds were dropped for tokenization, so the plant kingdom held only vegetables. A rerun keeping 3-leaf families, now the default, fixes that.

## 1. Symmetry decides the basis (known)

Karkada, Korchinski, Nava, Wyart and Bahri ([arXiv 2602.15029](https://arxiv.org/abs/2602.15029)) explain circles in models:
- If co-occurrence statistics depend only on the distance between two months, the statistics matrix commutes with translations.
- So its eigenvectors are Fourier modes. Long waves give the coarse circle, harmonics give the ripples.
- On page 9 they say hierarchical geometry "currently remains unexplained" in their framework.

The same argument for a **tree** works like this:
- Suppose statistics depend only on how far up the tree two words must go to meet.
- Then the matrix is invariant under shuffling siblings, and families within a kingdom.
- Its natural basis is the **Haar wavelets of the tree**: one constant per kingdom, then contrasts between families, then contrasts between leaves inside a family, each summing to zero over the family.

This is known mathematics. Ultrametric matrices have wavelet eigenbases. Saxe, McClelland and Ganguli (PNAS 2019) showed that networks trained on hierarchical data learn these modes, coarse levels first.

A side note on NumberHell's chessboards. They are **Walsh** functions: the basis of the *hypercube*, where every bit is an independent attribute. Haar wavelets live on the same points but belong to the *tree*, where a fine bit only means something inside its parent. The paper's Theorem 5 is the hypercube case. Nested categories are the tree case.

## 2. What the tree picture explains, and where it breaks

**Moving toward the siblings is a low-pass filter on the tree.** In the symmetric model the family mean is exactly the projection onto the coarser Haar levels. Moving a word a fraction *t* toward the mean of the other *n − 1* members gives

  h′ − m = (1 − nt/(n−1)) · (h − m)

so only the leaf-level part shrinks. That much is arithmetic. It makes the trees more alike. It does *not* predict that Qwen says "tree" *louder*. A linear read-out of the category would give the blend the average of its members' evidence, never more than both.

**But a 50/50 blend of two siblings did read as more category than both ends** (77% of the 22 RakeisuusClaude blends; the better end beaten by +1.10 nats, 95% CI +0.27 to +2.18). So the read-out bends. The obvious candidate is the model's own normalization:
- Two siblings of similar length lie near a sphere, and their midpoint cuts inside it.
- RMSNorm or LayerNorm stretches the midpoint back to the sphere.
- Whatever the two share is stretched with it, by

  g(t) = ((1 − t)|a| + t|b|) / |(1 − t)a + tb|,  so  g(½) = 1/cos(φ/2) for equal lengths,

  where φ is the angle between them.

**chord.py** tests whether that is all there is. If the overshoot follows g, the "louder kind" is geometry plus normalization, and that is the honest name for it.

**Where the clean tree picture is wrong.** In a pure one-tree model, a held-out leaf's residue is minus the sum of its siblings' residues, so it lies inside their span. RakeisuusClaude's principal-component arm should then have brought the leaf back. It recovered at most 3/22. That suggests most of each word's private part lies *outside* its family's span. Every word is a leaf in many trees at once (colour, size, use, other senses), and its identity is spread over all of them. That high-dimensional private part also explains why one sign bit per dimension kept the identity (18/22): for a high-dimensional vector, the sign keeps √(2/π) ≈ 0.80 of its direction.

## 3. The dendrite, in one honest paragraph

A passive dendritic fork with two identical branches does this exact Haar step. The trunk sees left + right and cannot see left − right. Where you listen decides what is forgotten. A linear fork cannot make the balanced input *louder*, though; that needs a bend. In a dendrite, the bend can come from local saturation: inputs on one thin branch sum sigmoidally (Polsky, Mel and Schiller 2004). In a transformer, the candidate bend is normalization, which is divisive like shunting inhibition. This is an analogy for building intuition, not evidence that Qwen does dendritic physics.

## The experiments

### chord.py: is the louder kind just normalization geometry? (Qwen3-1.7B, minutes)

```
python3.13 chord.py --self-test
python3.13 chord.py                  (plain few-shot frame)
python3.13 chord.py --frame chat     (RakeisuusClaude's chat frame; results go to runs/Qwen_Qwen3-1.7B_chat/)
```

| Part | What it measures | What would count |
|---|---|---|
| **A** | The pending control: each leaf 75% toward its siblings, against a random move of the **same length as the sibling move** at every layer. | The sibling move raises category evidence for clearly more leaves than the random move, with the paired difference's CI above 0. Otherwise the RakeisuusClaude random comparison was carried by its mismatched length. |
| **B** | Every sibling pair, blended at ¼, ½ and ¾. The overshoot of category evidence above the straight line between the ends, against the chord gain g. | A positive rank correlation between g and the overshoot (permutation p < 0.05) supports the normalization explanation. No correlation means the bend comes from somewhere else. |
| **C** | Real sibling against a **synthetic** sibling: family mean plus a random private part of the same size, sharing nothing else with *a*. | Equal overshoot at equal g means pure geometry. Real siblings overshooting more means their private parts carry something that adds up. That would be the interesting outcome, and the one not predicted by anything above. |
| **D** | The typicality curve: the siblings' mean *m* plus λ times the word's own residue, λ = 1.5, 1, ½, ¼, 0, and the same with a random residue of the same length. | If category evidence falls with distance from *m* along both curves alike, the effect is a prototype effect: distance from *m* in any direction costs category. Identity should survive only on the real curve. |

It uses every single-token leaf (no fitting, so no train/test split), the plain few-shot frame of `common.py`, and the clamp from the embedding through layer ≈ 12/28 of the depth. Results go to `runs/<model>/chord.json`.

### fork.py: does Qwen build the average itself? (Qwen3-1.7B, minutes)

```
python3.13 fork.py --self-test
python3.13 fork.py                 (chat frame; results in runs/Qwen_Qwen3-1.7B_chat/fork.json)
```

Everything above *manufactures* the abstraction by writing a blended state into one position. Here nothing is clamped. Qwen reads "Items: oak, pine" in a normal forward pass, and attention does whatever pooling it does. This is the reader-sets-the-conductance fork. The inputs all have the same length, so more tokens cannot pass for more evidence:

| Input | What it tests |
|---|---|
| oak, oak | repeated identity |
| pine, pine | another repeated identity |
| oak, pine (and pine, oak) | shared category, different identities |
| oak, salmon | different identities and categories |

Each input is asked the category question ("Are both of the items above trees?") and an identity question for every family member ("Is one of the items above a pine?").

1. **Overshoot at equal length.** Does "oak, pine" read as more tree than both "oak, oak" and "pine, pine"? A yes does *not* locate the bend: attention weights, the per-position MLPs and later blocks could each produce it. It says only whether the model's own fork shows the effect.
2. **Identity kept.** In the same "oak, pine" context, is "pine" still available when asked? Retention is 1 when pine is as available as in "pine, pine". The manufactured average lost identity (9/46 in the chat frame); a fork that depends on the query would not.
3. **Query dependence.** Swap pine for elm. Under the category question the answer should barely move. Under "Is one of the items a pine?" it should move a lot. The same input is then a sieve whose residue depends on the question.
4. **Attention, descriptive only.** From the answer position: how much attention each layer's heads put on the two items, and what share goes to the second item, under the category question and under the identity question about it. An even share is a balanced fork. These are weights, not interventions, so they do not prove what does the work. Locating the bend causally needs head or position interventions, the next step if 1–3 come out clean.

### access.py: is the map there before the model can read it? (Pythia, minutes, downloads cached)

```
python3.13 access.py --self-test
python3.13 access.py
```

In Pythia the tree's geometry is there by step 1000 while the model's own answers are below chance. access.py gives the same states to the simplest external reader. A prototype reader holds out one word, builds each family's centroid from the rest, and picks the closest centroid by angle. It has nothing to tune and never sees the held-out word.
- **Prediction, fixed before the run:** at step 1000 this reader is well above chance at the middle layers while the native reader is not. That would be a map before a reader.
- **If it fails:** the geometry is real but not yet usable on unseen words.
- **Two harder readers.** The kingdom of a word whose whole family the reader never saw, and the family whose *name* ("tree") lies closest, with no labelled examples at all.

### timeline.py: does the tree grow coarse-first? (Pythia, about an hour plus downloads)

```
python3.13 timeline.py --self-test
python3.13 timeline.py                 (keeps families with 3+ one-token leaves; results in runs/timeline_<model>_min3/)
python3.13 timeline.py --min-leaves 4  (the first run's word set)
```

This uses EleutherAI's Pythia-410m at 12 training checkpoints (`--steps` and `--model EleutherAI/pythia-1b` change that). Each checkpoint is about 0.9 GB to download, once. It records:
- **Geometry, no prompts.** The spread of the word states splits exactly into kingdom, family and leaf parts. Each part is reported as excess over 200 random relabelings of the same tree shape, at five depths.
- **Behaviour.** Printed-word category and identity accuracy, then the sibling move against the same-length random move, as in part A.

Predictions, fixed before any run:
1. The kingdom and family excess rise early in training (coarse first). The family excess may then fall back as leaves pull apart.
2. Once printed category accuracy passes 0.6, the sibling move beats the matched random move at every later checkpoint.

If fine structure comes before coarse, the coarse-first explanation is wrong for this model. Results go to `runs/timeline_<model>_min3/timeline.json`, and a stopped run resumes where it left off.

## Results: chord.py on Qwen3-1.7B, few-shot frame

Run on 24 Nov 2026 (RTX 3060, 524 s). The file is [`runs/Qwen_Qwen3-1.7B/chord.json`](runs/Qwen_Qwen3-1.7B/chord.json). The run used 46 single-token leaves in 8 families, with the clamp from the embedding through layer 12. Six leaves were dropped because they are not one token: birch, willow, spruce, sparrow, parrot and swan. Calibrated category evidence for the unedited words is large: a median of 15.9 nats, ranging from 11.1 to 20.1.

**A. Sibling move against a same-length random move.** Criterion: clearly more leaves rise, and the paired CI is above 0. **Half met.**
- Category evidence rose for 26/46 leaves under the sibling move and 20/46 under the random move.
- Mean change: siblings +0.28 nats [+0.03, +0.53], random −0.54 [−0.92, −0.20]. The paired difference is +0.82 [+0.39, +1.27], so the CI part is met.
- The count part is not: 26 against 20 is not "clearly more". The effect is uneven across families. Trees rose 5/5 and tools 4/6; vegetables, birds, fish and vehicles show essentially nothing.
- RakeisuusClaude's 22/22 does not reproduce in this frame.
- Identity is where the two moves really differ. The sibling move erased exact identity for 29/46 leaves (46 → 17); the same-length random move erased none (46 → 46). That is the arithmetic of §2: the sibling move removes exactly the private part, and a random direction of the same length barely touches it.
- Category itself never broke: 46/46 correct after the sibling move and 45/46 after the random move.

**B. Overshoot against chord gain.** Criterion: a positive correlation with p < 0.05. **Not met.**
- Across 110 sibling pairs, a 50/50 blend sat above the straight line between its ends by +0.12 nats [+0.06, +0.18], under 1% of the evidence scale. It was above the line for 61/110 pairs and above both ends for 21/110 (19%), against 77% in RakeisuusClaude.
- Spearman(g, overshoot) = −0.01, permutation p = 0.92.
- Caveat: g varied little (1.09 to 1.18), so this test is weak. Still, nothing here points to normalization as the source of the small overshoot that remains.

**C. Real sibling against a synthetic sibling.** Criterion: real siblings overshoot more. **Not met; this is the clearest result.**
- A synthetic sibling is the family mean plus a random private part of the same size. It reads as the category almost exactly as strongly as a real member: a median of 15.7 nats, against 15.9 for the real words.
- Blends with it overshoot by the same amount: synthetic +0.13 [+0.05, +0.22], real +0.12. The difference is −0.01 [−0.10, +0.07].
- So in this band, category evidence is carried by the family average alone. What a word's private part contains adds nothing that sums up across siblings. The interesting outcome, "private parts carry shared signal", is ruled out at this resolution.

**What it means.** The Haar picture of §1–2 describes this model well: family mean equals category, residue equals identity. The mechanism named in §2 for the overshoot (normalization stretching a chord) is not supported. But in this frame there is also little overshoot left to explain.

The plain unknown is the frame. RakeisuusClaude used Qwen's chat template, zero-shot, with "Is the item above a …?" questions; this run used plain few-shot text. `--frame chat` repeats all three parts in that frame:
- If A goes back to nearly all leaves rising and B back to most blends beating both ends, the louder kind is a property of the chat frame, not of the word states.
- If not, the earlier 22/22 and 77% came from that smaller word set (22 held-out words, 22 blends), not from anything general.

## Results: chord.py on Qwen3-1.7B, chat frame

Run on 24 Nov 2026 (235 s). The file is [`runs/Qwen_Qwen3-1.7B_chat/chord.json`](runs/Qwen_Qwen3-1.7B_chat/chord.json). This is RakeisuusClaude's frame: Qwen's chat template, zero-shot, "Is the item above a …? Answer yes or no." The words, band and clamp are the same as in the few-shot run. Calibrated category evidence is much larger: a median of 38.1 nats, ranging from 11.2 to 44.1.

| | few-shot frame | chat frame |
|---|---|---|
| A: category rose, sibling move / same-length random | 26 / 20 of 46 | **46 / 2** of 46 |
| A: mean change, sibling / random (nats) | +0.28 / −0.54 | **+4.75 / −4.63** |
| A: sibling − random, 95% CI | +0.82 [+0.39, +1.27] | **+9.38 [+6.89, +12.31]** |
| A: exact identity, own → sibling move / random move | 46 → 17 / 46 | 46 → 9 / 43 |
| B: 50/50 blend above both ends | 21 / 110 | **88 / 110** |
| B: above the straight line (nats) | +0.12 [+0.06, +0.18] | +2.87 [+2.28, +3.56] |
| B: Spearman(chord gain, overshoot) | −0.01, p = 0.92 | +0.11, p = 0.28 |
| C: overshoot, real − synthetic sibling | −0.01 [−0.10, +0.07] | +0.20 [−0.17, +0.64] |

**A: met, fully.** The sibling move raised category evidence in every family, 46/46 words, against 2/46 for the random move. The RakeisuusClaude 22/22 was not an artifact of the length-mismatched control. It belongs to this frame.

**B: the overshoot is back, but normalization is not its cause.** 80% of blends beat both ends, matching the 77% seen before. The chord gain still predicts nothing about which pairs overshoot. The overshoot is largest for trees (+8.4 nats) and birds (+6.7), and smallest for tools (+0.6) and instruments (+1.0).

**C: the content of the private parts does not matter.** The fake sibling, family mean plus a random residue of the right size, overshoots as much as a real one. The difference of +0.20 nats has a CI that includes 0.

**What it means.** The few-shot run said category lives in the family mean. The chat run adds that, in this frame, **every step away from the family mean costs category, in any direction.** A random move of a given length lowers category evidence by 4.6 nats. Taking the same length off the word's own residue raises it by 4.75. Blending two siblings, real or fake, shrinks the residue, so the blend reads as more of the category than either end.

That is the classic prototype effect of category learning: the never-shown average of a category's exemplars is classified as well as or better than the exemplars themselves (Posner and Keele 1968). As far as we know, it has not been shown this way, by moving a model's own internal states.

In the few-shot frame the same move barely matters. There the random move costs only 0.54 nats, so that read-out is about nine times flatter around the prototype. One caveat: each frame's states are captured in that frame's own context, so this is not quite "same states, different reader".

The open part is the shape of the curve, and whether it is the same for every direction away from the mean. Part D measures it, and `python3.13 chord.py --frame chat` reruns A–D together.

### Part D: the typicality curve (chat frame)

Same run, rerun with D. The table shows the category change against the word itself (nats, 95% CI) and exact identity, for the siblings' mean plus λ times a residue:

| λ | real residue: category | identity | random residue: category | identity |
|---|---|---|---|---|
| 1.5 (caricature) | −7.04 [−9.58, −4.91] | 46/46 | −13.91 [−17.55, −10.75] | 5/46 |
| 1 | 0 (the word) | 46/46 | −4.20 [−6.86, −1.78] | 1/46 |
| 0.5 | **+4.09 [+2.61, +5.92]** | **45/46** | +2.01 [+0.33, +4.07] | 4/46 |
| 0.25 | +4.75 [+3.10, +6.81] | 9/46 | +3.78 [+2.16, +5.85] | 4/46 |
| 0 (prototype) | +4.55 [+2.84, +6.64] | 5/46 | +4.55 [+2.84, +6.64] | 5/46 |

**The prediction that both curves fall together is half right.**
- Both fall monotonically with distance from the prototype, so distance matters.
- But the word's real residue costs much less category than a random residue of the same length: 0 against −4.2 at λ = 1, and −7.0 against −13.9 at λ = 1.5. A real word's private part lies mostly in directions the category read-out tolerates. A random part does not.
- So it is not a pure prototype effect. It is distance from the norm, weighted by direction.

**Identity is a threshold; category saturates.**
- Identity survives down to λ = 0.5 and collapses between 0.5 and 0.25.
- Category gets 86% of its total gain by λ = 0.5.
- So half the residue gives a word that is both recognisably itself and strongly typical. Face-space research calls this the **anti-caricature**: faces are coded as deviations from a norm, with identity read along the deviation and typicality falling with its length. Caricatures (λ > 1) stay recognisable but read as less typical (Valentine 1991; norm-based single neurons in monkey IT, Leopold, Bondar and Giese 2006).
- Qwen's word states behave the same way. As far as we know, this word-level face space has not been measured by moving a model's own states.

## Results: fork.py on Qwen3-1.7B, chat frame

Run on 24 Nov 2026 (54 s). The file is [`runs/Qwen_Qwen3-1.7B_chat/fork.json`](runs/Qwen_Qwen3-1.7B_chat/fork.json). It used 46 leaves and 110 sibling pairs, with both orders averaged. Nothing is clamped.

**1. Overshoot at equal length: present, smaller, and not the same effect.**
- "oak, pine" read above the line between "oak, oak" and "pine, pine" by +1.27 nats [+0.33, +2.22], and above both for 64/110 pairs.
- The manufactured blend gave +2.87 and 88/110.
- Pair by pair, the two overshoots are uncorrelated (Spearman 0.00). The families differ too:
  - natural: trees, tools and instruments up; vegetables and birds down;
  - manufactured: trees and birds largest.
- Order matters more than the overshoot: |cat(a, b) − cat(b, a)| = 3.13 nats.
- So the model's own reading of two siblings is not the manufactured average. One plausible ordinary reason: "both are trees" is simply better supported by two different trees than by one tree said twice.
- The other-kingdom control behaves: "oak, salmon" read below "oak, oak" for 44/46.

**2. Identity kept: yes, almost fully.**
- In the pair, the model said yes for a present item 220/220 times and no for an absent one 185/220 times.
- Retention is a median of 0.94, a mean of 0.92 [0.90, 0.93].
- The manufactured average kept identity for 9/46.

**3. Query dependence: strong.**
- Swapping the second sibling for another moved the category answer by a median of 1.21 nats. It moved "Is one of the items a pine?" by 36.19. The identity question moved more in 826/840 swaps.
- For scale, swapping in an other-kingdom item moves the category answer 26.97.
- Under the category question the model is about 20 times less sensitive to *which* sibling than to *which kingdom*. Under the identity question the same difference is decisive.
- One input, read through two sieves; what counts as residue depends on the question. This is what Astra predicted, and it is the clean result of the run.

**4. Attention: measured at the wrong place.**
- The answer position puts at most 0.7% of its attention directly on the two items. It reads them through intermediate positions, such as the question's own tokens.
- Its share on the second item drops from 0.5 to 0.2–0.4 in the middle layers, under both questions alike.
- The fork, if there is one, sits earlier in the sequence. Finding it needs attention from every position after the items, then an intervention on the heads found.

## Results: timeline.py on Pythia-410m

Run on 24 Nov 2026, 12 checkpoints. The file is [`runs/timeline_EleutherAI_pythia-410m/timeline.json`](runs/timeline_EleutherAI_pythia-410m/timeline.json).
- **Word set.** 30 leaves in 6 families. Trees and birds fell below 4 one-token leaves and were dropped, so the plant kingdom is vegetables only, and its kingdom split is the same as its family split.
- **Columns.** "Excess" is the Haar share above 200 random relabelings, with z in brackets. "Mid" is layer 12 of 24 and "emb" is the token embedding. "Cat" columns are the fraction of words whose own family question wins among the 6 family questions; chance is 0.17. "Sib − rnd" is in nats, with its 95% CI.

| step | mid kingdom | mid family | emb kingdom | emb family | printed cat | sibling-moved cat | random-moved cat | cat rose: sib / rnd | sib − rnd | identity: own → sib |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | +0.000 (0.0) | +0.000 (0.1) | +0.005 | −0.005 | 0.23 | 0.30 | 0.23 | 12 / 18 | −0.00 | 0.20 → 0.20 |
| 128 | +0.004 (0.6) | +0.003 (0.5) | +0.005 | −0.005 | 0.10 | 0.00 | 0.10 | 9 / 19 | −0.00 | 0.13 → 0.20 |
| 512 | **+0.065 (5.7)** | +0.022 (1.4) | +0.007 | −0.004 | 0.10 | 0.00 | 0.17 | 23 / 9 | +0.02 [+0.01, +0.02] | 0.07 → 0.20 |
| 1000 | +0.146 (12.8) | **+0.077 (4.9)** | +0.014 | +0.001 | 0.10 | 0.13 | 0.10 | 5 / 25 | −0.17 [−0.21, −0.13] | 0.17 → 0.20 |
| 2000 | +0.128 (11.5) | +0.093 (6.5) | +0.028 | +0.016 | 0.23 | 0.30 | 0.10 | 21 / 12 | +0.08 [+0.05, +0.12] | 0.33 → 0.23 |
| 4000 | +0.119 (12.9) | +0.090 (7.2) | +0.045 | +0.033 | **0.70** | 0.67 | 0.60 | 27 / 7 | +0.22 [+0.17, +0.26] | 0.87 → 0.33 |
| 8000 | +0.118 (13.3) | +0.075 (6.2) | +0.057 | +0.043 | 0.57 | 0.70 | 0.50 | 19 / 16 | +0.05 [−0.01, +0.12] | 1.00 → 0.27 |
| 16000 | +0.124 (14.4) | +0.077 (6.6) | +0.063 | +0.051 | 0.50 | 0.87 | 0.43 | 22 / 9 | +0.09 [+0.03, +0.14] | 1.00 → 0.27 |
| 33000 | +0.122 (13.6) | +0.076 (6.6) | +0.063 | +0.050 | 0.27 | 0.67 | 0.23 | 28 / 16 | +0.19 [+0.15, +0.23] | 1.00 → 0.30 |
| 66000 | +0.123 (13.5) | +0.078 (6.6) | +0.063 | +0.049 | 0.40 | 0.70 | 0.33 | 22 / 17 | +0.10 [+0.06, +0.15] | 0.47 → 0.23 |
| 100000 | +0.121 (13.0) | +0.082 (7.0) | +0.063 | +0.048 | 0.47 | **1.00** | 0.43 | 28 / 12 | +0.23 [+0.17, +0.29] | 0.73 → 0.33 |
| 143000 | +0.120 (13.0) | +0.085 (7.2) | +0.063 | +0.047 | 0.50 | **1.00** | 0.50 | 26 / 11 | +0.24 [+0.19, +0.29] | 0.93 → 0.30 |

**Prediction 1 (coarse first): met, with the caveat above.**
- At every depth, the kingdom split shows up one checkpoint before the family split.
  - Layers 6–23: kingdom z ≈ 5 at step 512 while family z ≈ 1; family reaches z ≈ 5 at step 1000.
  - Embedding: kingdom z 6.5 at step 1000 while family z 0.3.
- The kingdom share overshoots, peaking at step 1000, then settles about 15% lower (+0.146 → +0.12). The leaf share rises slightly to match. That is the predicted "falls back as leaves pull apart", but it is small.
- **After step 2000 the geometry is frozen.** The remaining 98.6% of training changes it by about 0.01.
- **The embedding table lags the layers.** Its tree reaches its plateau only around step 16000, about ten times later. At step 1000 the word position at layer 6 already carries +0.14 kingdom excess, but the embedding it was computed from carries +0.014. Nothing else feeds that position the word's identity, because the few-shot text before it is the same for every word. So early in training the tree is *computed* from the word by the first blocks, and only later *written into* the embedding table.
- One search found coarse-to-fine category learning shown in a vision model trained on child's-view video, and in Saxe et al.'s linear networks. It found nothing measuring it this way, by Haar level, across Pythia checkpoints, as far as we know.

**Prediction 2 (once printed category accuracy passes 0.6, the sibling move beats random at every later checkpoint): met on the mean, missed once on the CI.**
- Accuracy first passes 0.6 at step 4000. At all 7 later checkpoints the sibling move's mean is higher. The CI is above 0 at 6 of the 7; at step 8000 it is −0.01 to +0.12.
- Step 1000 shows the opposite (−0.17), but there the model cannot read categories at all (printed 0.10, below chance).

**The striking line, and its limit.** In the last two checkpoints, the sibling-moved state picks the right family for 30/30 words, while the printed words manage 15/30. The same-length random move stays at 0.43–0.50. In shared-plus-private terms, this is denoising: each word's private part leaks into the family questions, and averaging over siblings removes the leak.

The limit is that Pythia-410m barely uses this frame. Calibrated category evidence for the printed words has a median of 0.00 nats at the end, and 0.20 after the sibling move. These are rankings among small margins, not confident answers. The printed accuracy also swings between 0.27 and 0.70 across checkpoints while the geometry stays flat, which says the read-out, not the representation, is what varies.

**Next.**
- Rerun with the 3-leaf rule, so trees and birds come back and the plant kingdom has two families. Done; see below.
- Refine around the transition with `--steps 64,128,256,512,1000,2000,3000`; Pythia has nothing between 512 and 1000.
- If PolyPythias seeds (`EleutherAI/pythia-410m-seed1` and so on) load with `--model`, a second seed tells whether the 512/1000 order is fixed or luck.

### Rerun with trees and birds back (`--min-leaves 3`)

The file is [`runs/timeline_EleutherAI_pythia-410m_min3/timeline.json`](runs/timeline_EleutherAI_pythia-410m_min3/timeline.json). It used 36 leaves in 8 families; trees and birds have 3 leaves each, and the plant kingdom now has two families.

| step | mid kingdom (z) | mid family (z) | layer 6 kingdom / family z | emb kingdom / family z | printed cat | sibling-moved cat | sib − rnd |
|---|---|---|---|---|---|---|---|
| 0 | −0.001 (−0.2) | +0.002 (0.4) | −0.2 / 0.2 | 0.6 / 0.0 | 0.19 | 0.25 | −0.00 |
| 128 | +0.002 (0.4) | −0.003 (−0.3) | 0.3 / −0.2 | 0.5 / 0.0 | 0.08 | 0.00 | −0.00 |
| 512 | **+0.052 (5.5)** | +0.022 (1.5) | **4.8** / 0.9 | 1.0 / 0.5 | 0.08 | 0.00 | +0.02 |
| 1000 | +0.133 (13.2) | **+0.073 (4.4)** | 12.9 / 5.4 | **4.9** / 2.2 | 0.08 | 0.11 | −0.15 |
| 2000 | +0.120 (13.2) | +0.097 (6.3) | 13.1 / 8.1 | 9.6 / 5.8 | 0.19 | 0.25 | +0.07 [+0.03, +0.11] |
| 4000 | +0.103 (13.1) | +0.101 (8.3) | 13.7 / 9.3 | 12.2 / 8.0 | **0.58** | 0.56 | +0.20 [+0.15, +0.24] |
| 8000 | +0.110 (14.1) | +0.085 (6.9) | 14.6 / 7.8 | 13.2 / 8.0 | 0.47 | 0.67 | +0.04 [−0.02, +0.09] |
| 16000 | +0.113 (15.3) | +0.084 (7.1) | 15.4 / 8.4 | 12.6 / 8.7 | 0.42 | 0.81 | +0.07 [+0.02, +0.11] |
| 33000 | +0.111 (14.3) | +0.086 (7.0) | 16.0 / 8.3 | 12.4 / 9.6 | 0.22 | 0.64 | +0.17 [+0.12, +0.21] |
| 66000 | +0.117 (15.4) | +0.085 (7.0) | 17.2 / 7.5 | 12.7 / 9.2 | 0.31 | 0.44 | +0.10 [+0.05, +0.14] |
| 100000 | +0.115 (14.9) | +0.089 (7.2) | 16.5 / 8.1 | 13.0 / 9.2 | 0.33 | 0.92 | +0.23 [+0.17, +0.29] |
| 143000 | +0.113 (14.6) | +0.092 (7.5) | 16.7 / 8.3 | 13.1 / 9.0 | 0.39 | 0.83 | +0.21 [+0.15, +0.26] |

- **Coarse first survives the fix.** The kingdom level comes one checkpoint before the family level at every depth. It is at step 512 in the layers (z ≈ 5 against ≈ 1). In the embedding, at step 1000 kingdom z is 4.9 and family z is 2.2.
- **The embedding still lags the layers by several thousand steps.**
- **Prediction 2 cannot be scored strictly.** Native category accuracy peaks at 0.58 (chance 0.125 with 8 families) and never crosses the 0.6 trigger. After step 4000, the sibling move has the higher mean at every checkpoint, with the CI above 0 everywhere except step 8000 (again).
- **The gap between map and reader is plain.** From step 1000 the geometry is at z 13. The model's own answers reach their best only at step 4000 and then drift down to 0.2–0.4. The same states, moved toward their siblings, read 0.83–0.92 at the end. access.py asks whether an external reader can already use the map at step 1000.

## Results: access.py on Pythia-410m

Run on 24 Nov 2026. The file is [`runs/access_EleutherAI_pythia-410m_min3/access.json`](runs/access_EleutherAI_pythia-410m_min3/access.json). It used the 36 words of the rerun timeline. The prototype reader is leave-one-word-out nearest centroid by angle. Chance is 0.125 for 8 families and 0.33 for 3 kingdoms. "Native" is the model's own printed-word answer from timeline.py.

| step | native family | prototype family: emb | layer 6 | layer 12 | layer 23 | prototype kingdom, layer 12 |
|---|---|---|---|---|---|---|
| 0 | 0.19 | 0.06 | 0.11 | 0.08 | 0.17 | 0.31 |
| 128 | 0.08 | 0.06 | 0.06 | 0.06 | 0.14 | 0.39 |
| 512 | 0.08 | 0.14 | 0.31 | **0.44** | 0.36 | 0.64 |
| 1000 | **0.08** | 0.36 | 0.75 | **0.78** | 0.75 | 0.89 |
| 2000 | 0.19 | 0.81 | 0.83 | 0.86 | 0.78 | 0.92 |
| 4000 | 0.58 | 0.86 | 0.94 | 0.94 | 0.92 | 0.97 |
| 8000 | 0.47 | 0.92 | 0.94 | 0.97 | 0.86 | 0.97 |
| 16000 | 0.42 | 0.86 | 0.94 | 0.97 | 0.92 | 1.00 |
| 33000 | 0.22 | 0.89 | 0.94 | 0.97 | 0.92 | 1.00 |
| 66000 | 0.31 | 0.94 | 0.97 | 0.97 | 0.92 | 1.00 |
| 100000 | 0.33 | 0.89 | 0.97 | 0.94 | 0.94 | 1.00 |
| 143000 | 0.39 | 0.92 | 0.97 | 0.97 | 0.94 | 1.00 |

**The prediction is met.**
- At step 1000 the prototype reader gets 28/36 held-out words into the right family at layer 12. Chance is 4.5 words.
- The model's own answers get 3/36.
- Already at step 512 the reader is at 3.5 times chance, while the native reader is below chance.

**The gap does not close.**
- The outside reader is at 0.94–0.97 from step 4000 to the end.
- The native reader peaks at 0.58 and then drifts down to 0.2–0.4.
- In Astra's table that is the last row: the question format is a persistent bottleneck, not only a developmental delay. For a base model of this size answering yes/no in a few-shot frame, the map is nearly perfect and the reading is poor.
- The embedding follows the layers again: 0.36 at step 1000, 0.81 at 2000.

**What the outside reader is given that the native reader is not.** It has labelled examples: 35 other words with their families. The native reader has to link each cluster to the *English word* for it ("tree") and also follow the question format. So the gap mixes at least two things. The rerun separates them with two readers between these two:
- **Unseen-family kingdom.** The held-out word's whole family is removed before the kingdom centroids are built (Astra's control). This tests whether the coarse level is a real abstraction or the recognition of a familiar family cluster.
- **Category word.** There are no labelled examples. The word goes to the family whose own name, read as an item in the same frame, lies closest. This tests whether the cluster is linked to the word for it.

If the category-word reader rises late, near step 4000, while the prototype reader is high at 1000, then the delay is in linking the map to names. Getting a native yes/no reader on top of that is a further step. `python3.13 access.py` reruns everything with both readers. Old steps are recomputed, because they lack the new readers.

## What this repo is not

- Not a new theory. Haar wavelets on trees, coarse-first learning and normalization geometry are all published. The repo checks whether they *account for* the Qwen measurement, which is the honest next step before claiming more.
- Not a claim about brains. The dendrite paragraph is an analogy.

## Lineage

[RakeisuusClaude](https://github.com/anttiluode/RakeisuusClaude) (the effect and its note), [NumberHell](https://github.com/anttiluode/NumberHell) (Walsh chessboards), [Sihti](https://github.com/anttiluode/Sihti) (core plus residue), [Category](https://github.com/anttiluode/Category) (live tool), [CrossWord](https://github.com/anttiluode/CrossWord) (meaning pins the core, letters pin the leaf).
