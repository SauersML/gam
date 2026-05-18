# `gamfit.torch` — Specification

## Purpose

A growing class of users want to use gamfit's components — smooth basis evaluations, REML-optimized fits, response-geometry transforms — inside PyTorch training loops, jointly with neural network components. The immediate motivating use case is manifold-style sparse autoencoders, where a learned NN encoder produces design variables that flow through gamfit-fit smooths and back. The same pattern appears anywhere a differentiable smooth term needs to live inside a larger learned model: density-estimation networks, neural ODEs with smooth bases, hybrid scientific-ML models, anything where part of the model is a GAM-style component and part is a vanilla deep network.

gamfit's Rust engine already exposes the right primitives via numpy-direct FFI — that's the correct boundary for the engine. But it leaves a real gap: every PyTorch user has to write their own autograd wrappers, thread the analytic backward gradients correctly, and handle device/dtype coercion. This is exactly the kind of work where centralizing pays for itself many times over. The wrappers are subtle to get right, and silently incorrect backward gradients are the worst class of bug — the model trains, just badly, and debugging takes weeks. Almost every user writing their own wrappers will get them wrong on first attempt.

The goal of `gamfit.torch` is to be the canonical, tested bridge between gamfit and PyTorch. Optional dependency, framework purity preserved for non-torch users, single source of truth for everyone training models that include gamfit-style smooths.

## Design philosophy

Three principles, in priority order.

**Centralize the autograd glue.** The handful of gamfit primitives that have analytic backward implementations are the things that hurt to wrap. Threading the forward-state cache between forward and backward, matching exact tensor shape contracts for ragged batched layouts, handling the cases where some upstream gradients are None — these are all error-prone. Doing this work once and trusting it forever is the central value proposition.

**Expose primitives, not architectures.** Ship the building blocks. Users assemble. Don't ship SAE classes, NN-encoder templates, manifold-steering wrappers, or anything else research-specific. Users will bring their own loss functions, sparsity schemes, encoder architectures, and the value of the module shrinks fast if it tries to encode opinions about those choices. The bar for shipping a class is "would virtually every torch user reach for this regardless of their research direction" — that's a high bar, and most candidates fail it.

**Single source of truth.** Anything that exists in core gamfit as a numpy-input function should have a parallel in the torch module that accepts torch tensors, produces torch tensors, and produces numerically identical results to the numpy version. No "wait, does the torch version handle the rank-deficient null space the same way?" ambiguity. This is what makes the dual numpy/torch availability sustainable.

## Coverage

The functionality breaks into three conceptual groups, and the team should expose all three.

**The differentiable primitives with analytic backward.** Every gamfit FFI entry point that already has an analytic VJP in Rust — the closed-form Gaussian REML fits and their positions-based variants, including the batched ragged forms — gets a PyTorch counterpart that accepts tensors, computes the forward in Rust, and routes upstream gradients into the corresponding Rust backward. These are the things users absolutely cannot reasonably write themselves. They're also where the bulk of the implementation work lives.

**The basis and penalty primitives.** Differentiable basis evaluations for the supported smooth types (B-spline, Duchon, both with their periodic variants where applicable), the analytic basis derivatives that feed `t`-gradients, and the penalty matrix construction (which is fixed at init and doesn't itself need autograd, but does need a clean torch-tensor interface). These are individually easier to wrap than the REML primitives but should still be centralized for consistency with the rest of the module.

**The response-geometry transforms.** Simplex closure, CLR, ALR, their inverses, sphere log and exp maps, Fréchet means. These are simple enough that they could in principle be reimplemented in pure torch, and indeed many of them will be — they're just compositions of standard ops that autograd handles natively. The value of centralizing them is parity: behavior, numerical edge cases, and corner-case handling should match the numpy versions exactly so users don't have to verify equivalence themselves.

**A fitted-model loading utility.** A way to take an already-trained gamfit model and use it as a frozen, differentiable component inside a larger torch model. This addresses a real use case ("I have a GAM I trained on data, I want it as a fixed building block in a neural network") that's distinct from "train a smooth inside torch from scratch". The latter is gamfit's job. The former is what gamfit.torch should make easy.

## What NOT to ship

**Research-grade architectures.** No SAE classes, no autoencoder wrappers, no manifold-steering objects. These are research code, owned by the people doing the research, evolving on their schedule. Once gamfit ships them it owns their bug reports, API stability, and the implicit "this is the blessed way to do X" message that comes with shipping a named class.

**Trainable-from-scratch GAM components in torch.** It's tempting to write `nn.Module` versions of smooth terms with learnable spline coefficients that get trained by Adam. Resist. gamfit's REML is doing real model selection that pure end-to-end Adam can't replicate, and shipping a half-version of GAM fitting in torch undercuts gamfit's core value. The supported pattern is: train with gamfit, embed with `gamfit.torch.from_fitted`, or for joint-training research use cases compose the primitives directly.

**Convenience layers that hide the underlying Rust call.** No bundling multiple FFI calls into one for "ergonomics", no implicit caching, no pure-python fallbacks. The torch module is a faithful exposition of what the engine provides, with the autograd plumbing handled.

## Correctness requirements

Two non-negotiables.

**Gradcheck for every autograd-differentiable primitive.** For each wrapper with an analytic backward, there must be a test that perturbs each input by small amounts, computes finite-difference gradients, and compares them to the analytic gradient flowing back through the wrapper. Tolerance in the $10^{-5}$ to $10^{-6}$ range for f64. This catches bugs in both the Rust backward (which I noticed has no gradcheck against finite differences in the current Rust test suite either — that's an upstream gap worth fixing in the same effort) and in the wrapper plumbing. Without this, the wrappers can be silently wrong for a long time before someone notices their model is converging to garbage.

**Numerical parity with the numpy versions.** Anywhere a torch function and a numpy function are conceptually the same operation, they must produce numerically identical results on identical inputs, tested explicitly. This is what makes the "single source of truth" principle actually true in practice rather than aspirationally true.

## Device, dtype, and performance

The Rust backend runs on f64 CPU. Inputs arrive from torch as whatever dtype and device the user has. The torch module is responsible for the detach-cast-move dance: moving inputs to CPU at f64 for the Rust call, then casting the outputs back to the user's original dtype and device. This dance must be consistent across every primitive — not handled differently per function. The exact factoring is the team's call.

Worth being explicit in user-facing documentation: this implies a CPU↔GPU transfer per call. For training-loop usage that's potentially a bottleneck, which means primitives should be designed to be called once per batch at the batch level, not per-token in a loop. The batched-positions primitive is the right design pattern; the documentation should make clear that hot loops should always use batched variants and avoid per-feature Python iteration.

If transfer cost turns out to dominate, a future direction is keeping intermediate tensors on GPU and only transferring at the final reduction. That's out of scope for v1 but worth designing the API to not preclude.

## Dependency and packaging

`pip install gamfit` should not require torch. `pip install gamfit[torch]` should add it as an optional extra. The torch submodule should fail gracefully with a helpful error message ("install with `pip install gamfit[torch]` to enable") rather than crash with an opaque traceback when accessed without torch installed. The pattern is standard; the team should pick the cleanest factoring.

## Testing and validation

Beyond the gradcheck and parity tests, there should be at least one end-to-end example test that exercises the realistic compose-with-NN pattern: load a fitted gamfit model as a torch component, place it inside a small neural network, train end-to-end on synthetic data, verify sensible convergence. This catches integration bugs that pass unit tests but break in real usage.

The example should also serve as runnable documentation — the canonical "here's how to use this" reference for a new user.

## Documentation

A dedicated section in the gamfit docs covering: when to use this module, how it differs from the numpy use of gamfit, the device/dtype/performance caveats, the canonical patterns (frozen-fitted-model-inside-NN, smooths-as-differentiable-decoder), a brief note on what's intentionally out of scope and why, and pointers to the runnable example. Audience is ML researchers comfortable with PyTorch who may be encountering gamfit for the first time.
