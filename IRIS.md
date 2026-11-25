# Iris — a second reader for the frozen model

Protocol recorded before a real-model run, 25 November 2026. No empirical result yet.
This is a separate first gate from Claude's Maatuska: Maatuska asks whether an
ordinary LLM has a natural multiresolution axis. Iris needs no such axis. It asks
whether the part of a state invisible to the selected answer logits predicts
whether a fixed internal intervention will improve the native answer.

## Exact distinction

At the input h of the final linear output head, take the rows W of the unembedding
for every yes/no token used by common.Reader. Let P be the orthogonal projection
onto their row span. Then

    h = Ph + (I-P)h
    W(I-P)h = 0

This applies to the final normalized state, not to arbitrary earlier states or
to the whole vocabulary. The residue may affect other logits and does not imply
that the model never used it. It is invisible only to this immediate selected
linear readout. Every real capture checks both output-logit reconstruction and
the nullspace identity. We subtract the neutral question state before splitting,
and use the best and runner-up native questions, selected without labels.

## Fixed repair operation

One native pass and one offline counterfactual pass per item, using all candidate
family questions in Theory's few-shot frame. The repair is deliberately crude:
find the three nearest OTHER vocabulary items in the centered middle-layer word
states by cosine, without using the family labels or native answers. Replace the
word's state along Theory's existing embedding-to-band clamp by half its own
state and half the mean of those neighbors. The edited answer is still the
model's own calibrated yes/no ranking; no external classifier supplies it.

The vocabulary is an explicitly transductive, unlabeled bank (20 candidate
families before tokenizer filtering). Its other items, including items of
held-out families, can be neighbors. This is NOT evidence of inductive repair
on a new vocabulary. Candidate family names are supplied as questions, as in
Theory; their true correspondence to items is not supplied to repair. Ambiguous
words and the shared few-shot format are limitations.

## Gate 1: predict EDIT BENEFIT, not just correctness

Target for the offline monitor is edited_correct - native_correct, in {-1,0,+1}.
One entire true family is held out per fold. No training target from that family
enters its monitor. A fixed kernel ridge regression (penalty 1; no selection)
fits the other families. The monitor is supervised; there is no claim of
unlabeled homeostatic learning. Compare:

* output: all native calibrated/raw candidate margins, selected-token logits,
  entropy, top log probability and total yes/no log probability. No hidden state.
* output_read: same plus the row-space component.
* output_residue: same plus the candidate-logit-nullspace component.
* output_full: same plus the whole neutral-subtracted state.
* output_shuffled_residue: same plus independently permuted residue association.
* lowest_margin: ordinary native winner/runner-up uncertainty.
* never, always, expected random selection and omniscient oracle.

Every selective policy gets floor(half) of each held-out family's items (minimum
one); ranks are isolated within the fold, not compared across folds trained on
each other's labels. Family grouping is used for this OFFLINE budget only; a
deployable per-query threshold or online policy is not tested. Native and always
edit are endpoints with different budgets. Both potential answers are computed
offline so the result is a routing simulation, not a measured latency saving.
The full-state control tests whether the projection is necessary, rather than
just whether hidden-state access helps.

Locked pilot pass: at least five rescues and five harms exist; output_residue
beats output, output_read, output_shuffled_residue, lowest_margin, expected
random, never and always by at least 5 percentage points; paired family bootstrap
95% interval (fixed out-of-fold choices, not refitting) for residue minus output
has lower bound >0. Report full-state and oracle even if they win. If full
matches residue, do not claim the projection itself is necessary. A null can
mean the repair is not useful or the monitor cannot discover when; report both.
This is a small exploratory gate, not a confirmatory population-level inference.

## Run

From Theory (uses existing requirements, no new dependencies):

    python3.13 iris.py --self-test
    python3.13 iris.py --device cuda

Defaults to Pythia-410m step143000 in float32, few-shot frame. The result goes to
runs/iris_EleutherAI_pythia-410m_shots_step143000/iris.json.

Optional independent replication after the primary run:

    python3.13 iris.py --model Qwen/Qwen3-1.7B --frame chat --device cuda

Self-test checks algebra, synthetic routing, held-out target isolation and a
random GPT-NeoX model hook. It is not empirical evidence. The actual checkpoint
must be run locally; no claim of a completed real run. A positive pilot needs
independent vocabulary/prompt/model replication, a nontransductive repair,
and a true online policy before attempting an unlabeled controller.

## Lineage

[DendriticAttentionSystem](https://github.com/anttiluode/DendriticAttentionSystem):
the original iris measures full-input mismatch before narrowing its aperture.
[Sihti](https://github.com/anttiluode/Sihti): inspect what is discarded.
[AdaptiveObserverCache](https://github.com/anttiluode/AdaptiveObserverCache):
the reader can change with fixed model parameters, but a source-order axis is
not a semantic identity. Theory's own anti-caricature and access results motivate
this specific repair, not its expected success.

Not a new error-probing method or a claim of biological equivalence. The narrow
question is whether the answer-nullspace predicts the *causal benefit of a
particular repair* beyond output-only and ordinary hidden-state controls.
