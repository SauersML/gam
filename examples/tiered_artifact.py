"""One serialized tiered dictionary artifact: T0 ∪ T1 ∪ T2 ∪ Σ ∪ hash.

WS-J. This is the single canonical, content-hashed object of the SAC done-state
(SAC_PLAN Part 4): the union of the corpus statistics (T0), the linear sparse
dictionary (T1), the typed curved atoms (T2), the structured-residual whitening
model (Σ), and a content hash that folds every tier's bytes.

Per SPEC.md this file is *serialization / composition glue only* — the fitters
are Rust; the only arithmetic here is (a) the explained-variance ledger already
used by ``compose_tiers`` and (b) the canonical-byte layout of the content hash,
which is a byte-format definition, not a statistic. The T2 sub-hash is a faithful
Python mirror of ``crates/gam-sae/src/dictionary_artifact.rs`` (v1), the same port
W9's ``seed_stability.py`` validated against the Rust hash; extended here to cover
all tiers so the whole artifact is content-addressed.

The contract each producing workstream conforms to is ``ARTIFACT_SCHEMA.md``. The
loaders below read exactly those emitted formats and degrade gracefully (an absent
optional tier is recorded, never faked).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

_HASH_VERSION = b"gam-tiered-artifact-v2"
_T2_HASH_VERSION = b"gam-sae-dictionary-artifact-v1"  # matches dictionary_artifact.rs
_EPS = 1.0e-12


# --------------------------------------------------------------------------- #
# Canonical-byte helpers (mirror dictionary_artifact.rs; f64 little-endian).   #
# --------------------------------------------------------------------------- #
def _canonical_zero(v: float) -> float:
    return 0.0 if abs(v) < _EPS else float(v)


def _canonical_decoder_block(frame: np.ndarray) -> tuple[np.ndarray, float]:
    """Scale to unit Frobenius norm, orient so the max-|·| entry is positive.

    Byte-identical convention to ``dictionary_artifact.rs::canonical_decoder_block``
    followed by ``orient_in_place``: the finite-gauge representative of the atom.
    """
    frame = np.asarray(frame, dtype=np.float64)
    norm = float(np.sqrt(np.sum(frame * frame)))
    scale = 1.0 / norm if (norm > 0.0 and np.isfinite(norm)) else 1.0
    out = frame * scale
    out = np.vectorize(_canonical_zero)(out) if out.size else out
    if out.size:
        flat = out.reshape(-1)
        j = int(np.argmax(np.abs(flat)))
        if flat[j] < 0.0:
            out = -out
    return out.astype(np.float64), norm


def _hash_array_f64(h: "hashlib._Hash", arr: np.ndarray) -> None:
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
    a = np.vectorize(_canonical_zero)(a) if a.size else a
    h.update(a.astype("<f8").tobytes())


def _atom_canonical(topology: str, frame: np.ndarray, residual_gauge: str) -> bytes:
    """Canonical bytes of one atom (mirror ``hash_atom_into``)."""
    block, _ = _canonical_decoder_block(frame)
    h = hashlib.sha256()
    h.update(f"{topology}|{block.shape[0]}|{block.shape[1]}|".encode())
    _hash_array_f64(h, block)
    h.update(residual_gauge.encode())
    return h.digest()


def t2_dictionary_hash(atoms: list[dict[str, Any]], gauge_certificate: str) -> str:
    """Content hash of the T2 typed atoms — mirror of ``canonical_dictionary_artifact``.

    ``atoms`` is a list of ``{"topology": str, "frame": (b, d) array,
    "residual_gauge": str}``. Atoms are sorted by their own canonical hash so the
    dictionary hash is invariant to atom order / scale / reflection, exactly as
    the Rust version and W9's port. Returns a hex sha256.
    """
    digests = []
    for a in atoms:
        frame = np.atleast_2d(np.asarray(a["frame"], dtype=np.float64))
        digests.append(
            (
                _atom_canonical(str(a["topology"]), frame, str(a.get("residual_gauge", ""))),
                str(a["topology"]),
                frame,
                str(a.get("residual_gauge", "")),
            )
        )
    digests.sort(key=lambda t: t[0])
    h = hashlib.sha256()
    h.update(_T2_HASH_VERSION)
    h.update(gauge_certificate.encode())
    for _, topology, frame, gauge in digests:
        block, _ = _canonical_decoder_block(frame)
        h.update(f"{topology}|{block.shape[0]}|{block.shape[1]}|".encode())
        _hash_array_f64(h, block)
        h.update(gauge.encode())
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# The artifact.                                                                #
# --------------------------------------------------------------------------- #
@dataclass
class TieredArtifact:
    """The union artifact. Every field is optional except a tier pointer; an
    absent tier is recorded as ``None`` and never fabricated."""

    # T0 — corpus statistics
    t0: dict[str, Any] | None = None
    # T1 — linear sparse dictionary
    t1_decoder: np.ndarray | None = None          # (K1, P) f32
    t1_indices: np.ndarray | None = None          # (N, active) int, optional
    t1_codes: np.ndarray | None = None            # (N, active) f32, optional
    # T2 — typed curved atoms (list of per-atom typed reports)
    t2_atoms: list[dict[str, Any]] = field(default_factory=list)
    t2_manifold_payload: dict[str, Any] | None = None   # ManifoldSAE.to_dict()
    t2_meta: dict[str, Any] | None = None               # ev_trace, birth_log, EVs
    # Σ — structured residual whitening model
    sigma_root: np.ndarray | None = None          # (P, P) or (P, r) f32, optional
    # provenance
    provenance: dict[str, Any] = field(default_factory=dict)

    # ---- content hash ---------------------------------------------------- #
    def _t2_atom_specs(self) -> list[dict[str, Any]]:
        """Extract the (topology, frame, residual_gauge) specs the T2 hash needs
        from whichever T2 representation is present."""
        if self.t2_atoms:
            return self.t2_atoms
        specs: list[dict[str, Any]] = []
        payload = self.t2_manifold_payload
        if payload is not None:
            for atom in payload.get("atoms", []):
                frame = atom.get("frame")
                if frame is None:
                    frame = atom.get("decoder_block")
                if frame is None:
                    frame = atom.get("B")
                if frame is None:
                    continue
                # Accept either spelling of the gauge key: in-memory ``t2_atoms``
                # carry ``residual_gauge``; a ManifoldSAE.to_dict() payload carries
                # ``residual_finite_gauge``. Reading only one drops the gauge string
                # on reload and silently changes the content hash.
                gauge = atom.get("residual_gauge")
                if gauge is None:
                    gauge = atom.get("residual_finite_gauge", "")
                specs.append(
                    {
                        "topology": atom.get("topology", "circle"),
                        "frame": np.atleast_2d(np.asarray(frame, dtype=np.float64)),
                        "residual_gauge": gauge,
                    }
                )
        return specs

    def content_hash(self) -> str:
        """Canonical content hash folding every present tier (see ARTIFACT_SCHEMA.md)."""
        h = hashlib.sha256()
        h.update(_HASH_VERSION)
        # T0 — folds both the provisional (``mean``/``norm``) and the finalized
        # WS-D shape (per-dim ``mean``/``std``/``rms``, ``scale_median_std``, and
        # the nested ``rogue_dims`` massive-activation block).
        if self.t0 is not None:
            h.update(b"T0")
            for key in ("mean", "norm", "std", "rms", "scale",
                        "scale_median_std", "scale_median_rms"):
                v = self.t0.get(key)
                if v is not None:
                    _hash_array_f64(h, np.atleast_1d(np.asarray(v, dtype=np.float64)))
            rogue = self.t0.get("rogue_dims")
            # rogue_dims is either a flat index list (provisional convenience) or
            # the finalized dict ``{"index": [...], ...}``; fold only the indices,
            # which are the identity of the massive-activation set.
            if isinstance(rogue, dict):
                rogue = rogue.get("index")
            if rogue is not None:
                h.update(np.ascontiguousarray(np.asarray(rogue, dtype="<i8")).tobytes())
        # T1 — canonicalize each decoder row (unit norm, orient) then fold
        if self.t1_decoder is not None:
            h.update(b"T1")
            dec = np.atleast_2d(np.asarray(self.t1_decoder, dtype=np.float64))
            for row in dec:
                block, _ = _canonical_decoder_block(row.reshape(1, -1))
                _hash_array_f64(h, block)
        # T2 — the Rust-mirrored dictionary hash
        specs = self._t2_atom_specs()
        if specs:
            h.update(b"T2")
            gauge = (self.t2_meta or {}).get("gauge_certificate", "unspecified")
            h.update(t2_dictionary_hash(specs, gauge).encode())
        return "sha256:" + h.hexdigest()

    # ---- (de)serialization ----------------------------------------------- #
    def save(self, artifact_dir: str) -> str:
        """Write the artifact directory; returns the content hash."""
        os.makedirs(artifact_dir, exist_ok=True)
        manifest: dict[str, Any] = {
            "schema": "gam-tiered-artifact",
            "schema_version": 2,
            "tiers_present": [],
            "provenance": self.provenance,
        }
        if self.t0 is not None:
            manifest["t0"] = _jsonable_t0(self.t0)
            manifest["tiers_present"].append("t0")
        if self.t1_decoder is not None:
            np.save(os.path.join(artifact_dir, "t1_decoder.npy"),
                    np.ascontiguousarray(self.t1_decoder, dtype=np.float32))
            manifest["t1"] = {"decoder": "t1_decoder.npy",
                              "shape": list(np.asarray(self.t1_decoder).shape)}
            if self.t1_indices is not None and self.t1_codes is not None:
                np.savez(os.path.join(artifact_dir, "t1_codes.npz"),
                         indices=np.asarray(self.t1_indices),
                         codes=np.asarray(self.t1_codes, dtype=np.float32))
                manifest["t1"]["codes"] = "t1_codes.npz"
            manifest["tiers_present"].append("t1")
        if self.t2_manifold_payload is not None or self.t2_atoms:
            payload = self.t2_manifold_payload or {"atoms": self.t2_atoms}
            with open(os.path.join(artifact_dir, "t2_manifold.json"), "w") as f:
                json.dump(_jsonable(payload), f)
            manifest["t2"] = {"manifold": "t2_manifold.json",
                              "n_atoms": len(self._t2_atom_specs())}
            if self.t2_meta is not None:
                manifest["t2"]["meta"] = _jsonable(self.t2_meta)
            manifest["tiers_present"].append("t2")
        if self.sigma_root is not None:
            np.savez(os.path.join(artifact_dir, "sigma.npz"),
                     root=np.asarray(self.sigma_root, dtype=np.float32))
            manifest["sigma"] = {"root": "sigma.npz"}
            manifest["tiers_present"].append("sigma")
        else:
            manifest["sigma"] = None
        content_hash = self.content_hash()
        manifest["content_hash"] = content_hash
        with open(os.path.join(artifact_dir, "artifact.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        return content_hash

    @classmethod
    def load(cls, artifact_dir: str) -> "TieredArtifact":
        with open(os.path.join(artifact_dir, "artifact.json")) as f:
            manifest = json.load(f)
        if manifest.get("schema") != "gam-tiered-artifact" or manifest.get(
            "schema_version"
        ) != 2:
            raise ValueError(
                "TieredArtifact.load requires schema 'gam-tiered-artifact' version 2"
            )
        art = cls(provenance=manifest.get("provenance", {}))
        art.t0 = manifest.get("t0")
        t1 = manifest.get("t1")
        if t1:
            art.t1_decoder = np.load(os.path.join(artifact_dir, t1["decoder"]))
            if "codes" in t1:
                z = np.load(os.path.join(artifact_dir, t1["codes"]))
                art.t1_indices, art.t1_codes = z["indices"], z["codes"]
        t2 = manifest.get("t2")
        if t2:
            with open(os.path.join(artifact_dir, t2["manifold"])) as f:
                art.t2_manifold_payload = json.load(f)
            art.t2_meta = t2.get("meta")
        sigma = manifest.get("sigma")
        if sigma:
            art.sigma_root = np.load(os.path.join(artifact_dir, sigma["root"]))["root"]
        # Verify the recorded hash still matches the bytes on disk.
        recorded = manifest.get("content_hash")
        if recorded is not None and recorded != art.content_hash():
            art.provenance["hash_mismatch"] = {
                "recorded": recorded, "recomputed": art.content_hash()}
        return art


# --------------------------------------------------------------------------- #
# Loaders — consume exactly what each producing workstream emits.              #
# --------------------------------------------------------------------------- #
def load_t0_from_manifest(manifest_path: str) -> dict[str, Any]:
    """T0 from a WS-D harvest manifest (``residual_shard_io.py`` / ``finalize_harvest``).

    A *finalized* WS-D manifest carries a full ``t0`` block (the ``compute_t0``
    output: per-dim ``mean``/``std``/``rms``, ``scale_median_std`` /
    ``scale_median_rms``, and the nested ``rogue_dims`` = massive-activation dims
    ``{"index":[...], "rms":[...], "rms_over_median":[...], "mad_z":[...]}``). It is
    already the canonical T0 dict, so it is passed through verbatim (with
    ``d_model`` / ``total_tokens`` attached). A *provisional* manifest carries only
    ``stats: {mean, norm}``; T0 is then built from those, with ``rogue_dims`` /
    ``scale`` absent (recorded, not fabricated).
    """
    with open(manifest_path) as f:
        m = json.load(f)
    if isinstance(m.get("t0"), dict) and m["t0"]:
        t0 = dict(m["t0"])
        t0.setdefault("d_model", m.get("d_model"))
        t0["total_tokens"] = m.get("total_tokens")
        t0["provisional"] = bool(m.get("provisional", False))
        return t0
    stats = m.get("stats", {})
    return {
        "mean": stats.get("mean"),
        "norm": stats.get("norm"),
        "d_model": m.get("d_model"),
        "total_tokens": m.get("total_tokens"),
        "provisional": bool(m.get("provisional", True)),
    }


def load_tier1_artifact(source: Any) -> tuple[np.ndarray, dict[str, Any] | None]:
    """T1 decoder (+ baked T0) from WS-C.

    Accepts a ``gamfit.SparseDictionaryFit`` / ``SparseDictStreamArtifact`` (reads
    ``.decoder``) or a directory / json path that WS-C's ``dictionary_artifact``
    export writes (``{"decoder": path_or_list, "t0": {...}}``). Returns
    ``(decoder (K1,P) f32, t0_or_None)``.
    """
    t0 = None
    if hasattr(source, "decoder"):
        return np.ascontiguousarray(source.decoder, dtype=np.float32), t0
    if isinstance(source, str):
        if os.path.isdir(source):
            cand = os.path.join(source, "dictionary_artifact.json")
            source = cand if os.path.exists(cand) else os.path.join(source, "t1_decoder.npy")
        if source.endswith(".npy"):
            return np.load(source).astype(np.float32), t0
        with open(source) as f:
            payload = json.load(f)
        dec = payload["decoder"]
        decoder = (np.load(os.path.join(os.path.dirname(source), dec))
                   if isinstance(dec, str) else np.asarray(dec, dtype=np.float32))
        return np.ascontiguousarray(decoder, dtype=np.float32), payload.get("t0")
    raise TypeError(f"cannot load T1 from {type(source)!r}")


def load_sac_result(source: Any) -> tuple[dict[str, Any] | None, list[dict[str, Any]], dict[str, Any]]:
    """T2 from WS-A SAC output (``examples/sac_results/`` or a ``SacResult``).

    Returns ``(manifold_payload_or_None, atom_specs, meta)`` where ``atom_specs`` is
    the list of ``{"topology","frame","residual_gauge"}`` the T2 hash needs and
    ``meta`` carries ``ev_trace`` / ``birth_log`` / EVs. Reads either a single
    assembled ``ManifoldSAE.to_dict()`` payload or the per-atom
    ``atom_<k>.json`` + ``sac_meta.json`` interim form (see ARTIFACT_SCHEMA.md).
    """
    # In-memory SacResult (examples/sac_prototype.py).
    if hasattr(source, "atoms") and hasattr(source, "ev_trace"):
        specs = []
        for a in source.atoms:
            frames = a.fit.get_decoder() if hasattr(a.fit, "get_decoder") else []
            frame = np.atleast_2d(np.asarray(frames[0])) if frames else np.zeros((1, 1))
            specs.append({"topology": a.topology, "frame": frame,
                          "residual_gauge": a.hybrid_verdict})
        meta = {"ev_trace": list(source.ev_trace), "birth_log": list(source.birth_log),
                "t1_ev": source.t1_ev, "combined_ev": source.combined_ev, "k": source.k}
        return None, specs, meta

    if not isinstance(source, str):
        raise TypeError(f"cannot load SAC result from {type(source)!r}")

    # Single assembled ManifoldSAE payload.
    if source.endswith(".json") and os.path.isfile(source):
        with open(source) as f:
            payload = json.load(f)
        return payload, [], {}

    # Directory: atom_<k>.json + sac_meta.json interim form.
    meta = {}
    meta_path = os.path.join(source, "sac_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    specs = []
    atom_files = sorted(fn for fn in os.listdir(source)
                        if fn.startswith("atom_") and fn.endswith(".json"))
    for fn in atom_files:
        with open(os.path.join(source, fn)) as f:
            atom_payload = json.load(f)
        atoms = atom_payload.get("atoms", [atom_payload])
        for atom in atoms:
            frame = atom.get("frame") or atom.get("decoder_block") or atom.get("B")
            if frame is None:
                continue
            specs.append({"topology": atom.get("topology", meta.get("atom_topology", "circle")),
                          "frame": np.atleast_2d(np.asarray(frame, dtype=np.float64)),
                          "residual_gauge": atom.get("residual_finite_gauge", "")})
    return None, specs, meta


# --------------------------------------------------------------------------- #
# JSON coercion helpers.                                                        #
# --------------------------------------------------------------------------- #
def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _jsonable_t0(t0: dict[str, Any]) -> dict[str, Any]:
    return _jsonable(t0)
