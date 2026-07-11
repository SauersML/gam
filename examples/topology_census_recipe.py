#!/usr/bin/env python
"""Recipe: an honest unsupervised topology census of real LLM SAE features.

This is executable documentation, not a one-command runner: it records the
exact validated recipe (and the control discipline that makes the numbers
meaningful) for censusing SAE feature-group geometry with
``gamfit.adjudicate_atom_shape``. The measured facts below come from a census
of real Qwen3-8B layer-20 features (4M wikitext-103 tokens, 16,384-feature
SGD SAE) with matched structureless controls and a synthetic positive
control.

Stages
------
1. HARVEST residual activations (``examples/harvest_residual_activations.py``).
2. REAL CENSUS: train an overcomplete SAE into the sparse regime, group
   features by co-activation, project each group's codes to 2-D by PCA, and
   race shapes with ``gamfit.adjudicate_atom_shape`` (the in-repo pipeline:
   ``tests/sae/qwen_real_sae_pipeline.py``).
3. NEGATIVE CONTROLS (mandatory): a per-dimension-shuffled copy AND a
   covariance-matched Gaussian copy of the same activation matrix, pushed
   through the byte-identical pipeline. Every "circle" they produce is an
   artifact; report real rates against this per-run floor, never raw.
   Measured floors reached double digits (up to 19-23% of groups at looser
   sparsity), and real data was statistically indistinguishable from the
   controls in the healthy-dictionary census (Fisher p = 0.62 tier-1,
   0.72 tier-2).
4. SECOND TIER: run the centroid circular-ordering test
   (``examples/centroid_ordering.py``) on every mixture verdict — discrete
   clusters arranged on a circle adjudicate as ``mixture_k``, so raw circle
   wins under-count curved structure.
5. POSITIVE CONTROL: inject a synthetic ring through the same projection +
   adjudication path to measure detection reach. Measured: a dominant ring
   is recovered near-perfectly (circle margin +0.867, angle recovery
   r = 0.997), but a ring masked by a linear factor at >= ~1x its radius is
   already lost after top-2 PCA — absence of circle wins is not absence of
   circles.

Key measured gates a reuser should know
---------------------------------------
* DICTIONARY HEALTH: circle verdicts only appear in the sparse regime
  (mean L0 below ~300; 0 circle wins across 377 dense-dictionary groups).
  With this recipe, dictionaries only reached that regime at l1 ~ 1-2;
  token count alone did not fix a dense dictionary. Report mean L0 next to
  any verdict rate.
* CONTROLS ARE NOT OPTIONAL: without the matched shuffle + Gaussian floors,
  a census's circle rate is uninterpretable (see stage 3 numbers).
* EM ABORTS: ``adjudicate_atom_shape`` raises when the mixture EM cannot
  certify convergence; catch and skip, and count the skips in the report.

Sketch (each stage validated at small scale first)
--------------------------------------------------
  python examples/harvest_residual_activations.py --model Qwen/Qwen3-8B \\
      --layer 20 --n-tokens 4000000 --seq-len 256 --out $ACTS
  # train the SAE + group + adjudicate (see tests/sae/qwen_real_sae_pipeline.py)
  # ... run the identical pipeline on the shuffled + Gaussian controls ...
  # ... centroid_ordering.centroid_circular_ordering on mixture verdicts ...
"""

if __name__ == "__main__":
    print(__doc__)
