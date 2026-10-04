# Existing one-hour VPD pilot evaluation

This evaluates checkpoint15679, not a new training run. The original batch1 pilot used9,738 steps/4,985,856 tokens and3,638 allocated L40 seconds, including400 parameter-only warmup updates. It is unconverged and is not the published413GPU-hour result or evidence of a matched-compute method win.

The frozen80-episode panel uses two full512-token passages (Pile validation1040/1041). Autonomous selection starts all U/V components on, excludes Delta, then runs CI>0 on its own states for at most10 updates. All80 failed to reach a fixed point. Native-fed CI explicitly reads the native model's states and is a separate oracle diagnostic. Execution is upstream FP32 CUDA; KL is a FP64 GPU diagnostic without interval certificates. Native and all-on converted execution passed bitwise parity against the original nano wrappers at16tokens.

Recorded clean/worst-group mean KL: autonomous4.62427/6.17684; native-fed1.31802/3.31461. Evaluation used61 allocated GPU seconds. Dense U/V, CI, and retained native embedding/norm numeric payload is22,362,275,840 bits at32bits/value. **This is not full C32:** threshold/iteration structure, architecture constants, wiring and bindings remain unencoded in the current Artifact grammar. Per-token traces are not charged.

To replay with the recorded checkpoint and public model/data already present, copy this directory to `~/mpd-data/bench/vpd-matched-codex/eval-1h`, and its `sources/*` to `~/mpd-data/bench/vpd-matched-codex/`. Submit `python run.py convert` on1CPU/16GiB for15minutes, then `python evaluate_guard.py` on1L40/4CPU/24GiB for30minutes with an `afterok` dependency. The protocol pins all source/config/data hashes; conversion freezes the raw and optimizer-free checkpoint hashes. Use a fresh output path if replaying. Raw checkpoints/data are excluded from this archive.

`results/` records actual conversion/evaluation jobs15847/15849, source and checkpoint hashes, numeric counts, per-episode results and nonconvergence. No fitting, checkpoint updates, coefficient sweep or threshold tuning occurs during evaluation.
