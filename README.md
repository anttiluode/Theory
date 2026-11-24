# Theory

What the sibling-average result in [RakeisuusClaude](https://github.com/anttiluode/RakeisuusClaude) is, mathematically, and the experiments that could kill each part of that explanation.

## Did we find something? The blunt version

**We found a clean, controlled effect, not a discovery.** In frozen Qwen3-1.7B, moving a word 75% of the way toward the average of its siblings raised category evidence for all 22 held-out words. Exact identity fell from 22/22 to 3/22. Moving toward the category word did not do this, and a random move did the opposite. ([note](https://github.com/anttiluode/RakeisuusClaude/blob/main/HIERARCHICAL_GEOMETRY.md))

Most of it is what the known mathematics predicts. Below is that mathematics, and exactly which parts are known and which are not.

**After the first chord.py run (24 Nov 2026, [results](#results-chordpy-on-qwen3-17b-few-shot-frame)):** the shared-plus-private picture held up, and the parts that could have been new did not show up. In the few-shot frame, a fake sibling made of the family average plus random noise reads as the category as strongly as a real word, and blends with it overshoot exactly as much. The "louder kind" shrank to about 1% of the scale and does not follow the normalization gain. One open question remained: whether the louder kind belongs to the chat frame RakeisuusClaude used.

**After the chat-frame run ([results](#results-chordpy-on-qwen3-17b-chat-frame)): it does.** In the chat frame the effect is back in full:
- The sibling move raised category evidence for 46/46 words; the same-length random move raised it for 2/46.
- 80% of sibling blends beat both of their ends.
- It still does not follow the normalization gain, and the fake sibling (family average plus noise) does it just as well.

So what lifts the category is getting closer to the family average, not anything the siblings' private parts hold. Psychology has a name for this, the prototype effect: a never-seen average is classified better than the members it was made from. Part D of `chord.py` measures that curve directly, and `fork.py` asks whether Qwen builds the average itself when it reads two words.

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
- Rerun with the 3-leaf rule, so trees and birds come back and the plant kingdom has two families. Results go to a separate folder: `python3.13 timeline.py`.
- Refine around the transition with `--steps 64,128,256,512,1000,2000,3000`; Pythia has nothing between 512 and 1000.
- If PolyPythias seeds (`EleutherAI/pythia-410m-seed1` and so on) load with `--model`, a second seed tells whether the 512/1000 order is fixed or luck.

## What this repo is not

- Not a new theory. Haar wavelets on trees, coarse-first learning and normalization geometry are all published. The repo checks whether they *account for* the Qwen measurement, which is the honest next step before claiming more.
- Not a claim about brains. The dendrite paragraph is an analogy.

## Lineage

[RakeisuusClaude](https://github.com/anttiluode/RakeisuusClaude) (the effect and its note), [NumberHell](https://github.com/anttiluode/NumberHell) (Walsh chessboards), [Sihti](https://github.com/anttiluode/Sihti) (core plus residue), [Category](https://github.com/anttiluode/Category) (live tool), [CrossWord](https://github.com/anttiluode/CrossWord) (meaning pins the core, letters pin the leaf).
