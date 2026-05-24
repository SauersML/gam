from __future__ import annotations

import hashlib
import json
from typing import Any

from opentelemetry import trace

from app.models import Completion, ConceptTarget, SteerRequest
from app.services.diagnostics import build_diagnostics


tracer = trace.get_tracer("steering-server.steering")


def concept_targets_hash(concept_targets: list[ConceptTarget]) -> str:
    payload = [target.model_dump(mode="json") for target in concept_targets]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def fit_gauge(model: str, layer: int, concept_targets: list[ConceptTarget]) -> tuple[dict[str, Any], dict[str, Any]]:
    with tracer.start_as_current_span("gauge_fit") as span:
        span.set_attribute("model", model)
        span.set_attribute("layer", layer)
        span.set_attribute("concept_count", len(concept_targets))
        target_hash = concept_targets_hash(concept_targets)
        gauge = {
            "basis": "deterministic-anchor-projection",
            "model": model,
            "layer": layer,
            "target_hash": target_hash,
            "components": [
                {
                    "concept": target.name,
                    "target": target.target,
                    "anchor_count": len(target.anchors),
                    "locality_weight": target.locality_weight,
                }
                for target in concept_targets
            ],
            "stub": "to-be-implemented: replace with real gauge fitting over hidden-state anchors",
        }
        diagnostics = build_diagnostics(concept_targets, layer)
        return gauge, diagnostics


class MockLLMBackend:
    """Deterministic LLM backend used until a production model runner is integrated."""

    def complete(
        self,
        *,
        model: str,
        layer: int,
        gauge: dict[str, Any],
        request: SteerRequest,
    ) -> list[Completion]:
        with tracer.start_as_current_span("steer_call") as span:
            span.set_attribute("model", model)
            span.set_attribute("layer", layer)
            span.set_attribute("prompt_count", len(request.prompts))
            span.set_attribute("alpha", request.alpha)
            concept_phrase = ", ".join(
                f"{target.name}={target.target:+.2f}" for target in request.concept_targets
            )
            completions: list[Completion] = []
            for prompt in request.prompts:
                seed = hashlib.sha256(
                    f"{model}|{layer}|{gauge.get('target_hash')}|{request.alpha}|{prompt}".encode("utf-8")
                ).hexdigest()[:12]
                text = (
                    f"{prompt}\n\n"
                    f"[steered:{seed}] Applied alpha {request.alpha:+.2f} at layer {layer} "
                    f"toward {concept_phrase}."
                )
                tokens = text.split()[: request.max_tokens]
                completions.append(
                    Completion(
                        prompt=prompt,
                        text=" ".join(tokens),
                        tokens=tokens,
                        finish_reason="length" if len(text.split()) > request.max_tokens else "stop",
                        steering_trace={
                            "backend": "mock",
                            "gauge_basis": gauge.get("basis"),
                            "gauge_target_hash": gauge.get("target_hash"),
                            "stub": "to-be-implemented: stream from a real LLM runtime",
                        },
                    )
                )
            return completions


backend = MockLLMBackend()
