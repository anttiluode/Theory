# Theory

What the sibling-average result in [RakeisuusClaude](https://github.com/anttiluode/RakeisuusClaude) is, mathematically, and two experiments that could kill each part of that explanation.

## Did we find something? The blunt version

**We found a clean, controlled effect, not a discovery.** In frozen Qwen3-1.7B, moving a word 75% of the way toward the average of its siblings raised category evidence for all 22 held-out words. Exact identity fell from 22/22 to 3/22. Moving toward the category word did not do this, and a random move did the opposite. ([note](https://github.com/anttiluode/RakeisuusClaude/blob/main/HIERARCHICAL_GEOMETRY.md))

Most of it is what the known mathematics predicts. Below is that mathematics, and exactly which parts are known and which are not. Two parts could still be new: a measured overshoot that the geometry may or may not fully explain, and a timeline of the tree growing during training that nobody has measured this way, as far as we know. Both are runnable here.

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
python3.13 chord.py
```

| Part | What it measures | What would count |
|---|---|---|
| **A** | The pending control: each leaf 75% toward its siblings, against a random move of the **same length as the sibling move** at every layer. | The sibling move raises category evidence for clearly more leaves than the random move, with the paired difference's CI above 0. Otherwise the RakeisuusClaude random comparison was carried by its mismatched length. |
| **B** | Every sibling pair, blended at ¼, ½ and ¾. The overshoot of category evidence above the straight line between the ends, against the chord gain g. | A positive rank correlation between g and the overshoot (permutation p < 0.05) supports the normalization explanation. No correlation means the bend comes from somewhere else. |
| **C** | Real sibling against a **synthetic** sibling: family mean plus a random private part of the same size, sharing nothing else with *a*. | Equal overshoot at equal g means pure geometry. Real siblings overshooting more means their private parts carry something that adds up. That would be the interesting outcome, and the one not predicted by anything above. |

It uses every single-token leaf (no fitting, so no train/test split), the plain few-shot frame of `common.py`, and the clamp from the embedding through layer ≈ 12/28 of the depth. Results go to `runs/<model>/chord.json`.

### timeline.py: does the tree grow coarse-first? (Pythia, about an hour plus downloads)

```
python3.13 timeline.py --self-test
python3.13 timeline.py
```

This uses EleutherAI's Pythia-410m at 12 training checkpoints (`--steps` and `--model EleutherAI/pythia-1b` change that). Each checkpoint is about 0.9 GB to download, once. It records:
- **Geometry, no prompts.** The spread of the word states splits exactly into kingdom, family and leaf parts. Each part is reported as excess over 200 random relabelings of the same tree shape, at five depths.
- **Behaviour.** Printed-word category and identity accuracy, then the sibling move against the same-length random move, as in part A.

Predictions, fixed before any run:
1. The kingdom and family excess rise early in training (coarse first). The family excess may then fall back as leaves pull apart.
2. Once printed category accuracy passes 0.6, the sibling move beats the matched random move at every later checkpoint.

If fine structure comes before coarse, the coarse-first explanation is wrong for this model. Results go to `runs/timeline_<model>/timeline.json`, and a stopped run resumes where it left off.

## What this repo is not

- Not a new theory. Haar wavelets on trees, coarse-first learning and normalization geometry are all published. The repo checks whether they *account for* the Qwen measurement, which is the honest next step before claiming more.
- Not a claim about brains. The dendrite paragraph is an analogy.

## Lineage

[RakeisuusClaude](https://github.com/anttiluode/RakeisuusClaude) (the effect and its note), [NumberHell](https://github.com/anttiluode/NumberHell) (Walsh chessboards), [Sihti](https://github.com/anttiluode/Sihti) (core plus residue), [Category](https://github.com/anttiluode/Category) (live tool), [CrossWord](https://github.com/anttiluode/CrossWord) (meaning pins the core, letters pin the leaf).
