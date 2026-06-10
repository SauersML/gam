from __future__ import annotations

import math
import re
import typing

import numpy as np


def configure(context: dict[str, typing.Any]) -> None:
    globals().update(context)


def _scenario_fit_mapping(scenario_name: typing.Any) -> typing.Any:
    geo_eas_cfg = _geo_disease_eas_scenario_cfg(scenario_name)
    papuan_cfg = _papuan_oce_scenario_cfg(scenario_name)
    subpop_cfg = _geo_subpop16_scenario_cfg(scenario_name)
    latlon_cfg = _geo_latlon_scenario_cfg(scenario_name)
    if geo_eas_cfg is not None:
        return {
            "family": "binomial-logit",
            "smooth_cols": geo_eas_cfg["smooth_cols"],
            "smooth_basis": geo_eas_cfg["smooth_basis"],
            "linear_cols": geo_eas_cfg["linear_cols"],
            "knots": int(geo_eas_cfg["knots"]),
        }
    if papuan_cfg is not None:
        return {
            "family": "binomial-logit",
            "smooth_cols": papuan_cfg["smooth_cols"],
            "smooth_basis": papuan_cfg["smooth_basis"],
            "linear_cols": papuan_cfg["linear_cols"],
            "knots": int(papuan_cfg["knots"]),
        }
    if subpop_cfg is not None:
        return {
            "family": "binomial-logit",
            "smooth_cols": subpop_cfg["smooth_cols"],
            "smooth_basis": subpop_cfg["smooth_basis"],
            "linear_cols": subpop_cfg["linear_cols"],
            "knots": int(subpop_cfg["knots"]),
        }
    if latlon_cfg is not None:
        return {
            "family": "binomial-logit",
            "smooth_cols": latlon_cfg["smooth_cols"],
            "smooth_basis": latlon_cfg["smooth_basis"],
            "linear_cols": latlon_cfg["linear_cols"],
            "knots": int(latlon_cfg["knots"]),
        }
    return {
        "small_dense": dict(
            family="binomial-logit",
            smooth_cols=["x1", "x2"],
            linear_cols=[],
            smooth_basis="ps",
            knots=7,
            double_penalty=False,
        ),
        "medium": dict(
            family="binomial-logit",
            smooth_cols=["x1", "x2"],
            linear_cols=[],
            smooth_basis="ps",
            knots=7,
        ),
        "pathological_ill_conditioned": dict(
            family="binomial-logit",
            smooth_cols=["x1", "x2"],
            linear_cols=[],
            smooth_basis="ps",
            knots=7,
        ),
        "lidar_semipar": dict(family="gaussian", smooth_col="range", linear_cols=[], smooth_basis="ps", knots=24),
        "bone_gamair": dict(family="binomial-logit", smooth_col="t", linear_cols=["trt_auto"], smooth_basis="ps", knots=8),
        "prostate_gamair": dict(
            family="binomial-logit",
            smooth_cols=["pc1", "pc2"],
            linear_cols=[],
            smooth_basis="ps",
            knots=8,
            double_penalty=False,
        ),
        "horse_colic": dict(
            family="binomial-logit",
            smooth_cols=["pulse", "rectal_temp", "packed_cell_volume"],
            linear_cols=[],
            smooth_basis="ps",
            knots=8,
            double_penalty=False,
        ),
        "wine_gamair": dict(
            family="gaussian",
            smooth_cols=["s_temp", "year", "h_rain", "w_rain", "h_temp"],
            linear_cols=[],
            smooth_basis="ps",
            knots=7,
            double_penalty=True,
        ),
        "wine_temp_vs_year": dict(
            family="gaussian", smooth_col="year", linear_cols=[], smooth_basis="ps", knots=7
        ),
        "wine_price_vs_temp": dict(
            family="gaussian", smooth_col="temp", linear_cols=[], smooth_basis="ps", knots=7
        ),
        "us48_demand_5day": dict(
            family="gaussian",
            smooth_cols=["hour", "demand_forecast", "net_generation", "total_interchange"],
            linear_cols=[],
            smooth_basis="ps",
            knots=8,
        ),
        "us48_demand_31day": dict(
            family="gaussian",
            smooth_cols=["hour", "demand_forecast", "net_generation", "total_interchange"],
            linear_cols=[],
            smooth_basis="ps",
            knots=12,
        ),
        "haberman_5yr": dict(
            family="binomial-logit",
            smooth_cols=["age", "op_year", "axil_nodes"],
            linear_cols=[],
            smooth_basis="ps",
            knots=8,
        ),
        "icu_survival_death": dict(
            family="binomial-logit",
            smooth_cols=["age", "bmi", "hr_max", "sysbp_min"],
            linear_cols=[],
            smooth_basis="ps",
            knots=7,
        ),
        "icu_survival_los": dict(
            family="binomial-logit",
            smooth_cols=["age", "bmi", "hr_max", "sysbp_min", "temp_apache"],
            linear_cols=[],
            smooth_basis="ps",
            knots=7,
        ),
        "heart_failure_survival": dict(
            family="binomial-logit",
            smooth_cols=[
                "age",
                "log_creatinine_phosphokinase",
                "ejection_fraction",
                "log_platelets",
                "log_serum_creatinine",
                "serum_sodium",
            ],
            linear_cols=["anaemia", "diabetes", "high_blood_pressure", "sex", "smoking"],
            smooth_basis="ps",
            knots=7,
        ),
        "cirrhosis_survival": dict(
            family="binomial-logit",
            smooth_cols=[
                "age",
                "bilirubin",
                "cholesterol",
                "albumin",
                "copper",
                "alk_phos",
                "sgot",
                "tryglicerides",
                "platelets",
                "prothrombin",
                "edema",
                "stage",
            ],
            linear_cols=["drug", "sex_male", "ascites", "hepatomegaly", "spiders"],
            smooth_basis="ps",
            knots=7,
        ),
        "geo_disease_tp": dict(
            family="binomial-logit",
            smooth_cols=[
                f"pc{i}" for i in range(1, _fixed_joint_spatial_pc_count("geo_disease", 16) + 1)
            ],
            smooth_basis="thinplate",
            linear_cols=[],
            knots=24,
        ),
        "geo_disease_duchon": dict(
            family="binomial-logit",
            smooth_cols=[
                f"pc{i}" for i in range(1, _fixed_joint_spatial_pc_count("geo_disease", 16) + 1)
            ],
            smooth_basis="duchon",
            linear_cols=[],
            knots=24,
        ),
        "geo_disease_matern": dict(
            family="binomial-logit",
            smooth_cols=[
                f"pc{i}" for i in range(1, _fixed_joint_spatial_pc_count("geo_disease", 16) + 1)
            ],
            smooth_basis="matern",
            linear_cols=[],
            knots=24,
        ),
        "geo_disease_shrinkage": dict(
            family="binomial-logit",
            smooth_cols=[
                f"pc{i}" for i in range(1, _fixed_joint_spatial_pc_count("geo_disease", 16) + 1)
            ],
            smooth_basis="thinplate",
            linear_cols=[],
            knots=24,
        ),
        "geo_disease_ps_per_pc": dict(
            family="binomial-logit",
            smooth_cols=[f"pc{i}" for i in range(1, 17)],
            smooth_basis="ps",
            linear_cols=[],
            knots=24,
        ),
        "geo_subpop16_randomprev_randomscale_duchonfull_k50": dict(
            family="binomial-logit",
            smooth_cols=[f"pc{i}" for i in range(1, 17)],
            smooth_basis="duchon",
            linear_cols=[],
            knots=50,
        ),
        "geo_subpop16_margslope_aniso_duchon16d_k50": dict(
            family="binomial-logit",
            smooth_cols=[f"pc{i}" for i in range(1, 17)],
            smooth_basis="duchon",
            linear_cols=[],
            knots=50,
            scale_dimensions=True,
        ),
        "continuous_order_fractional_spde_nu18": dict(
            family="gaussian",
            smooth_col="x",
            linear_cols=[],
            smooth_basis="matern",
            knots=24,
            double_penalty=True,
        ),
        "continuous_order_boundary_rough": dict(
            family="gaussian",
            smooth_col="x",
            linear_cols=[],
            smooth_basis="matern",
            knots=24,
            double_penalty=True,
        ),
        "continuous_order_boundary_smooth": dict(
            family="gaussian",
            smooth_col="x",
            linear_cols=[],
            smooth_basis="matern",
            knots=24,
            double_penalty=True,
        ),
        "thread3_admixture_cliff": dict(
            family="binomial-logit",
            smooth_cols=[f"pc{i}" for i in range(1, 17)],
            smooth_basis="matern",
            linear_cols=[],
            knots=16,
            double_penalty=True,
        ),
    }.get(scenario_name)


def _effective_scenario_fit_mapping(scenario_name: str, override: dict[str, typing.Any] | None = None) -> typing.Any:
    cfg = _scenario_fit_mapping(scenario_name)
    if cfg is None:
        return None
    if not override:
        return dict(cfg)
    merged = dict(cfg)
    merged.update(override)
    return merged


def _canonical_smooth_basis(basis: typing.Any) -> typing.Any:
    b = str(basis or "ps").strip().lower()
    # Legacy alias used to mean one P-spline per feature; canonical basis is "ps".
    if b == "bspline_per_pc":
        return "ps"
    return b


def _is_joint_spatial_basis(basis: str) -> bool:
    return _canonical_smooth_basis(basis) in {"thinplate", "tps", "duchon", "matern"}


# ---------------------------------------------------------------------------
# Joint-PC contract
#
# PCs are ALWAYS a single joint smooth, never N independent 1D smooths and
# never a mixture of smoothed-leading-PCs + linear-trailing-PCs. The PC
# eigenbasis has been deliberately decorrelated, so per-axis additivity is
# both statistically misspecified (the meaningful heterogeneity lives on the
# joint manifold) and a wallclock disaster at large scale (16 separate
# `s(pcN, ...)` blocks instead of one multi-D Duchon).
#
# Every formula builder in this file routes PC-named columns through these
# helpers so the contract is enforced from a single place rather than at
# every call site.
# ---------------------------------------------------------------------------

_PC_COL_PATTERN = re.compile(r"^pc\d+(?:_std)?$")


def _is_pc_column(name: typing.Any) -> bool:
    """A column is a PC grouping axis iff its name matches `pc{N}` or `pc{N}_std`."""
    return bool(_PC_COL_PATTERN.match(str(name).strip()))


def _split_pc_columns(cols: typing.Any) -> tuple[list[str], list[str]]:
    """Return (pc_cols, other_cols) preserving original order."""
    pc_cols: list[str] = []
    other_cols: list[str] = []
    for c in cols or []:
        s = str(c)
        if _is_pc_column(s):
            pc_cols.append(s)
        else:
            other_cols.append(s)
    return pc_cols, other_cols


def _joint_pc_basis(requested_basis: typing.Any) -> str:
    """Choose the basis used when emitting the joint-PC smooth.

    Multi-D-capable bases (thinplate, duchon, matern) pass through unchanged.
    1D-only picks like `ps` are routed to `duchon`, which supports arbitrary
    dimension and is the canonical joint-PC contract.
    """
    canonical = _canonical_smooth_basis(requested_basis)
    if canonical in {"thinplate", "tps", "duchon", "matern"}:
        return typing.cast(str, canonical)
    return "duchon"


def _backend_supports_joint_pc(backend: str) -> bool:
    """Return True iff the backend can fit a single multi-D smooth over PCs.

    `r_gamlss` and `gamboostlss` do not have a clean multi-D smoother in
    their standard formula DSLs (gamlss only exposes 1D `pb()` / `cs()` /
    `pbm()` reliably; gamboostlss's `bspatial()` is 2D-only). Lanes targeting
    those backends on PC scenarios should be filtered out at the harness
    level rather than emitting per-axis workarounds that violate the
    joint-PC contract.
    """
    return backend in {"rust", "mgcv", "bamlss", "brms"}


def _emit_joint_pc_term(
    backend: str,
    pc_cols: list[str],
    *,
    knot_count: int,
    pc_basis: str,
    double_penalty: bool = True,
) -> str:
    """Build a single joint multi-D smooth over the PC grouping axes for `backend`.

    Returns one term string. Always emits a multi-D term — never N separate
    1D smooths. Raises if the backend can't represent a multi-D smooth in
    its standard formula DSL.
    """
    if not _backend_supports_joint_pc(backend):
        raise RuntimeError(
            f"backend '{backend}' has no clean multi-D smoother for the "
            f"joint-PC contract; this lane must be filtered out at the "
            f"harness level (PCs cannot be split into per-axis smooths)"
        )
    cols = ", ".join(pc_cols)
    pc_basis = _canonical_smooth_basis(pc_basis)
    if backend == "rust" and pc_basis == "duchon":
        min_centers = len(pc_cols) + 2
        if knot_count < min_centers:
            raise RuntimeError(
                f"joint-PC Duchon over {len(pc_cols)} PCs needs centers/k >= {min_centers} "
                f"to leave at least one kernel degree of freedom after the linear "
                f"polynomial nullspace; got {knot_count}"
            )
    if backend == "rust":
        dp = ", double_penalty=true" if double_penalty else ", double_penalty=false"
        if pc_basis in {"thinplate", "tps"}:
            return f"thinplate({cols}, centers={knot_count}{dp})"
        if pc_basis == "duchon":
            # With order=0 (p_order=1), the Duchon center-collision derivative
            # phi^(2)(0) exists iff 2*(p+s) > dimension+2, i.e. s > dimension/2.
            # The smallest integer power satisfying this strictly is dim//2 + 1.
            return (
                f"duchon({cols}, centers={knot_count}, "
                f"order=0, power={len(pc_cols) // 2 + 1}, length_scale=1.0)"
            )
        if pc_basis == "matern":
            return f"matern({cols}, centers={knot_count}{dp})"
        raise RuntimeError(f"unsupported joint-PC rust basis '{pc_basis}'")
    # mgcv / bamlss / brms all use mgcv-style `s(...)`. brms's `bf(...)` and
    # bamlss's formula DSL accept the same surface here.
    k_expr = f"min({knot_count}, nrow(train_df)-1)"
    if pc_basis in {"thinplate", "tps"}:
        return f"s({cols}, bs='tp', k={k_expr})"
    if pc_basis == "duchon":
        return f"s({cols}, bs='ds', m=c(1,0), k={k_expr})"
    if pc_basis == "matern":
        return f"s({cols}, bs='gp', m=c(-4,1.0), k={k_expr})"
    raise RuntimeError(f"unsupported joint-PC mgcv-family basis '{pc_basis}'")


def _requires_joint_spatial_term(cfg: dict[str, typing.Any] | None) -> bool:
    """A scenario requires a joint multi-D smooth iff:

    * its declared `smooth_basis` is multi-D-capable AND there are 2+ smooth
      columns (the original criterion), OR
    * its smooth columns include 2+ PCs (the joint-PC contract — the basis
      may be `ps` in the YAML for legacy reasons but the runtime emission
      will route to `_emit_joint_pc_term`, which is multi-D).

    `linear_cols` containing PCs also flips this true: those columns get
    folded into the joint smooth by the formula builders, so the effective
    fit has a multi-D term even if `smooth_cols` alone wouldn't trigger it.
    """
    if not cfg:
        return False
    smooth_cols = list(cfg.get("smooth_cols") or [])
    linear_cols = list(cfg.get("linear_cols") or [])
    pc_smooth = [c for c in smooth_cols if _is_pc_column(c)]
    pc_linear = [c for c in linear_cols if _is_pc_column(c)]
    if len(pc_smooth) + len(pc_linear) >= 2:
        return True
    if _is_joint_spatial_basis(cfg.get("smooth_basis", "ps")) and len(smooth_cols) >= 2:
        return True
    return False


def _rust_duchon_options_for_dimension(dimension: int) -> str:
    # With order=0 (p_order=1), the Duchon center-collision derivative
    # phi^(2)(0) exists iff 2*(p+s) > dimension+2, i.e. s > dimension/2.
    # The smallest integer power satisfying this strictly is dim//2 + 1.
    power = dimension // 2 + 1
    return f", order=0, power={power}, length_scale=1.0"


def _rust_joint_spatial_term(basis: str, smooth_cols: list[str], knot_count: int, dp_opt: str) -> str:
    basis = _canonical_smooth_basis(basis)
    cols = ", ".join(str(c) for c in smooth_cols)
    if basis in {"thinplate", "tps"}:
        return f"thinplate({cols}, centers={knot_count}{dp_opt})"
    if basis == "duchon":
        return f"duchon({cols}, centers={knot_count}{_rust_duchon_options_for_dimension(len(smooth_cols))})"
    if basis == "matern":
        return f"matern({cols}, centers={knot_count}{dp_opt})"
    raise RuntimeError(f"Unsupported joint Rust spatial basis '{basis}'")


def _mgcv_joint_spatial_term(basis: str, smooth_cols: list[str], knot_count: int) -> str:
    basis = _canonical_smooth_basis(basis)
    cols = ", ".join(str(c) for c in smooth_cols)
    if basis in {"thinplate", "tps"}:
        return f"s({cols}, bs='tp', k=min({knot_count}, nrow(train_df)-1))"
    if basis == "duchon":
        return f"s({cols}, bs='ds', m=c(1,0), k=min({knot_count}, nrow(train_df)-1))"
    if basis == "matern":
        return f"s({cols}, bs='gp', m=c(-4,1.0), k=min({knot_count}, nrow(train_df)-1))"
    raise RuntimeError(f"Unsupported joint mgcv spatial basis '{basis}'")


def _rust_formula_for_scenario(scenario_name: typing.Any, ds: typing.Any, *, cfg_override: dict[str, typing.Any] | None = None) -> typing.Any:
    cfg = _effective_scenario_fit_mapping(scenario_name, cfg_override)
    if cfg is None:
        raise RuntimeError(f"No Rust formula mapping configured for scenario '{scenario_name}'")
    target = ds["target"]
    # Fold any PC linear_cols into the joint-PC smooth — PCs are always one
    # joint object, never partly smooth and partly linear.
    raw_linear = list(cfg.get("linear_cols", []))
    pc_linear, true_linear = _split_pc_columns(raw_linear)
    raw_smooth = list(cfg.get("smooth_cols") or [])
    if pc_linear:
        # Move PC linear terms into smooth_cols so they participate in the
        # joint smooth below.
        existing = set(str(c) for c in raw_smooth)
        for c in pc_linear:
            if c not in existing:
                raw_smooth.append(c)
        cfg = dict(cfg)
        cfg["smooth_cols"] = raw_smooth
        cfg["linear_cols"] = true_linear
    terms = [f"linear({c})" for c in cfg.get("linear_cols", [])]
    basis = _canonical_smooth_basis(cfg.get("smooth_basis", "ps"))
    knot_count = int(cfg.get("knots", 8))
    if knot_count < 0:
        raise RuntimeError(
            f"Invalid knot count {knot_count} for scenario '{scenario_name}'; expected >= 0."
        )
    # Keep shrinkage policy explicit and aligned with mgcv `select`.
    use_double_penalty = bool(cfg.get("double_penalty", True))
    dp_opt = f", double_penalty={'true' if use_double_penalty else 'false'}"
    smooth_cols = cfg.get("smooth_cols")
    if smooth_cols:
        # PCs are always a single joint object regardless of the scenario's
        # nominal basis. If the scenario asked for `ps` over PC columns, route
        # them through a joint multi-D smooth (default duchon) — never N
        # independent 1D smooths.
        pc_smooth_cols, other_smooth_cols = _split_pc_columns(smooth_cols)
        if len(pc_smooth_cols) >= 2:
            pc_basis = _joint_pc_basis(basis)
            terms.append(_rust_joint_spatial_term(pc_basis, pc_smooth_cols, knot_count, dp_opt))
        elif len(pc_smooth_cols) == 1:
            # Only one PC — emit the single-axis smooth using the scenario basis.
            other_smooth_cols = pc_smooth_cols + other_smooth_cols
            pc_smooth_cols = []
        if other_smooth_cols:
            if _is_joint_spatial_basis(basis) and len(other_smooth_cols) >= 2:
                terms.append(_rust_joint_spatial_term(basis, other_smooth_cols, knot_count, dp_opt))
            else:
                for col in other_smooth_cols:
                    if basis in {"ps", "bspline", "p-spline"}:
                        terms.append(f"s({col}, type=ps, knots={knot_count}{dp_opt})")
                    elif basis in {"thinplate", "tps"}:
                        terms.append(f"s({col}, type=tps, centers={knot_count}{dp_opt})")
                    elif basis == "duchon":
                        terms.append(
                            f"s({col}, type=duchon, centers={knot_count}{_rust_duchon_options_for_dimension(1)})"
                        )
                    elif basis == "matern":
                        terms.append(f"s({col}, type=matern, centers={knot_count}{dp_opt})")
                    else:
                        raise RuntimeError(
                            f"Unsupported Rust smooth basis '{basis}' for scenario '{scenario_name}'"
                        )
    else:
        col = cfg.get("smooth_col")
        if col:
            if basis in {"thinplate", "tps"}:
                terms.append(f"s({col}, type=tps, centers={knot_count}{dp_opt})")
            elif basis in {"ps", "bspline", "p-spline"} and "double_penalty" in cfg:
                dp = "true" if bool(cfg["double_penalty"]) else "false"
                terms.append(f"s({col}, type=ps, knots={knot_count}, double_penalty={dp})")
            elif basis in {"ps", "bspline", "p-spline"}:
                terms.append(f"s({col}, type=ps, knots={knot_count})")
            elif basis in {"duchon", "matern"}:
                if basis == "duchon":
                    terms.append(
                        f"s({col}, type=duchon, centers={knot_count}{_rust_duchon_options_for_dimension(1)})"
                    )
                else:
                    terms.append(f"s({col}, type={basis}, centers={knot_count}{dp_opt})")
            else:
                raise RuntimeError(
                    f"Unsupported Rust smooth basis '{basis}' for scenario '{scenario_name}'"
                )

    formula = f"{target} ~ " + _formula_rhs_from_terms(terms)
    return cfg["family"], formula


def _cfg_with_excluded_columns(cfg: dict[str, typing.Any], excluded_cols: typing.Any) -> dict[str, typing.Any]:
    excluded = {str(col) for col in (excluded_cols or [])}
    if not excluded:
        return dict(cfg)
    trimmed = dict(cfg)
    trimmed["linear_cols"] = [str(c) for c in cfg.get("linear_cols", []) if str(c) not in excluded]
    if cfg.get("smooth_cols") is not None:
        trimmed["smooth_cols"] = [str(c) for c in cfg.get("smooth_cols", []) if str(c) not in excluded]
        trimmed.pop("smooth_col", None)
    else:
        smooth_col = cfg.get("smooth_col")
        if smooth_col is not None and str(smooth_col) in excluded:
            trimmed.pop("smooth_col", None)
    return trimmed


def _formula_rhs_from_terms(terms: list[str]) -> str:
    return " + ".join(str(term) for term in terms) if terms else "1"


def _formula_rhs_text(formula: str) -> str:
    if "~" not in formula:
        raise RuntimeError(f"cannot extract RHS from malformed formula: {formula!r}")
    _lhs, rhs = formula.split("~", 1)
    rhs = rhs.strip()
    if not rhs:
        raise RuntimeError(f"formula is missing RHS terms: {formula!r}")
    return rhs


def _select_marginal_slope_z_column(scenario_name: str, ds: dict[str, typing.Any]) -> str:
    cfg = _scenario_fit_mapping(scenario_name)
    if cfg is None:
        raise RuntimeError(f"No marginal-slope mapping configured for scenario '{scenario_name}'")
    smooth_cols = list(cfg.get("smooth_cols") or ([cfg["smooth_col"]] if cfg.get("smooth_col") else []))
    linear_cols = list(cfg.get("linear_cols", []))
    candidates = smooth_cols + [c for c in linear_cols if c not in smooth_cols] + list(ds.get("features", []))
    seen = set()
    for col in candidates:
        name = str(col)
        if name in seen:
            continue
        seen.add(name)
        if name in ds.get("features", []):
            return name
    raise RuntimeError(f"could not choose a marginal-slope z column for scenario '{scenario_name}'")


def _apply_exact_train_fold_standardization(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    column: str,
) -> None:
    train_vals = train_raw[column].to_numpy(dtype=float)
    mu = float(np.mean(train_vals))
    var = float(np.mean((train_vals - mu) ** 2))
    sd = math.sqrt(var) if np.isfinite(var) and var >= 1e-16 else 1.0
    train_df[column] = (train_raw[column].to_numpy(dtype=float) - mu) / sd
    test_df[column] = (test_raw[column].to_numpy(dtype=float) - mu) / sd


def _rust_marginal_slope_formulas_for_scenario(scenario_name: str, ds: dict[str, typing.Any]) -> tuple[str, str, str]:
    base_cfg = _effective_scenario_fit_mapping(scenario_name) or {}
    z_column = _select_marginal_slope_z_column(scenario_name, ds)
    marginal_cfg = _cfg_with_excluded_columns(base_cfg, [z_column])
    _, marginal_formula = _rust_formula_for_scenario(
        scenario_name,
        ds,
        cfg_override=marginal_cfg,
    )
    _, logslope_formula = _rust_formula_for_scenario(
        scenario_name,
        ds,
        cfg_override=marginal_cfg,
    )
    return z_column, marginal_formula, _formula_rhs_text(logslope_formula)


def _mgcv_formula_for_scenario(scenario_name: typing.Any, ds: typing.Any) -> typing.Any:
    cfg = _scenario_fit_mapping(scenario_name)
    if cfg is None:
        raise RuntimeError(f"No shared smooth mapping configured for scenario '{scenario_name}'")
    target = ds["target"]
    # Fold any PC linear_cols into the joint-PC smooth — PCs are always one
    # joint object, never partly smooth and partly linear.
    raw_linear = list(cfg.get("linear_cols", []))
    pc_linear, true_linear = _split_pc_columns(raw_linear)
    raw_smooth = list(cfg.get("smooth_cols") or [])
    if pc_linear:
        existing = set(str(c) for c in raw_smooth)
        for c in pc_linear:
            if c not in existing:
                raw_smooth.append(c)
        cfg = dict(cfg)
        cfg["smooth_cols"] = raw_smooth
        cfg["linear_cols"] = true_linear
    terms = [str(c) for c in cfg.get("linear_cols", [])]
    basis = _canonical_smooth_basis(cfg.get("smooth_basis", "ps"))
    knot_count = int(cfg.get("knots", 8))

    if basis in {"ps", "bspline", "p-spline"}:
        bs_code = "ps"
    elif basis in {"thinplate", "tps"}:
        bs_code = "tp"
    elif basis == "duchon":
        # Keep Duchon settings explicit in the external contender formula.
        bs_code = "ds"
    elif basis == "matern":
        # Use explicit stationary Matérn GP in mgcv:
        #   m[1] = -4 -> Matérn with kappa = 2.5, stationary (no linear trend term)
        #   m[2] = 1.0 -> fixed range on z-scored predictors
        # This avoids hidden mgcv defaults and keeps comparison with Rust's
        # explicit Matérn basis fair and reproducible.
        bs_code = "gp"
    else:
        raise RuntimeError(
            f"Unsupported mgcv smooth basis '{basis}' for scenario '{scenario_name}'"
        )

    smooth_cols = cfg.get("smooth_cols")
    if smooth_cols:
        k_val = knot_count + 4 if bs_code == "ps" else knot_count
        # Mirror the rust contract: PCs always enter as a single joint smooth.
        pc_smooth_cols, other_smooth_cols = _split_pc_columns(smooth_cols)
        if len(pc_smooth_cols) >= 2:
            pc_basis = _joint_pc_basis(basis)
            terms.append(_mgcv_joint_spatial_term(pc_basis, pc_smooth_cols, k_val))
        elif len(pc_smooth_cols) == 1:
            other_smooth_cols = pc_smooth_cols + other_smooth_cols
            pc_smooth_cols = []
        if other_smooth_cols:
            if _is_joint_spatial_basis(basis) and len(other_smooth_cols) >= 2:
                terms.append(_mgcv_joint_spatial_term(basis, other_smooth_cols, k_val))
            else:
                for col in other_smooth_cols:
                    if basis == "matern":
                        terms.append(
                            f"s({col}, bs='gp', m=c(-4,1.0), k=min({k_val}, nrow(train_df)-1))"
                        )
                    elif basis == "duchon":
                        terms.append(
                            f"s({col}, bs='ds', m=c(1,0), k=min({k_val}, nrow(train_df)-1))"
                        )
                    else:
                        terms.append(f"s({col}, bs='{bs_code}', k=min({k_val}, nrow(train_df)-1))")
    else:
        col = cfg.get("smooth_col")
        if col:
            k_val = knot_count + 4 if bs_code == "ps" else knot_count
            if basis == "matern":
                terms.append(
                    f"s({col}, bs='gp', m=c(-4,1.0), k=min({k_val}, nrow(train_df)-1))"
                )
            elif basis == "duchon":
                terms.append(
                    f"s({col}, bs='ds', m=c(1,0), k=min({k_val}, nrow(train_df)-1))"
                )
            else:
                terms.append(f"s({col}, bs='{bs_code}', k=min({k_val}, nrow(train_df)-1))")
    if not terms:
        raise RuntimeError(f"empty mgcv term list for scenario '{scenario_name}'")
    return f"{target} ~ " + " + ".join(terms)


def _survival_formula_mapping(scenario_name: str) -> dict[str, typing.Any]:
    cfg = _scenario_fit_mapping(scenario_name)
    if cfg is None:
        raise RuntimeError(f"No survival formula mapping configured for scenario '{scenario_name}'")
    basis = _canonical_smooth_basis(cfg.get("smooth_basis", "ps"))
    if basis != "ps":
        raise RuntimeError(
            f"Unsupported survival smooth basis '{basis}' for scenario '{scenario_name}'; expected ps"
        )
    smooth_cols = list(cfg.get("smooth_cols") or ([cfg["smooth_col"]] if cfg.get("smooth_col") else []))
    linear_cols = list(cfg.get("linear_cols", []))
    overlap = sorted(set(smooth_cols) & set(linear_cols))
    if overlap:
        raise RuntimeError(
            f"survival formula mapping overlaps smooth and linear terms for '{scenario_name}': {overlap}"
        )
    if not smooth_cols and not linear_cols:
        raise RuntimeError(f"empty survival term mapping for scenario '{scenario_name}'")
    return {
        "smooth_cols": smooth_cols,
        "linear_cols": linear_cols,
        "knots": max(4, int(cfg.get("knots", 8))),
    }


def _rust_survival_formula_for_scenario(scenario_name: str, *, exclude_cols: typing.Any=None) -> str:
    cfg = _cfg_with_excluded_columns(_survival_formula_mapping(scenario_name), exclude_cols or [])
    terms = [f"linear({c})" for c in cfg["linear_cols"]]
    terms.extend(f"s({c}, type=ps, knots={cfg['knots']})" for c in cfg["smooth_cols"])
    return _formula_rhs_from_terms(terms)


def _rust_survival_marginal_slope_formulas_for_scenario(
    scenario_name: str,
    ds: dict[str, typing.Any],
) -> tuple[str, str, str]:
    z_column = _select_marginal_slope_z_column(scenario_name, ds)
    rhs = _rust_survival_formula_for_scenario(scenario_name, exclude_cols=[z_column])
    return z_column, rhs, rhs


def _coxph_survival_formula_for_scenario(scenario_name: str, ds: dict[str, typing.Any]) -> str:
    cfg = _survival_formula_mapping(scenario_name)
    terms = list(cfg["linear_cols"])
    terms.extend(
        f"pspline({c}, df=min({cfg['knots']}, nrow(train_df)-1))" for c in cfg["smooth_cols"]
    )
    return f"Surv({ds['time_col']}, {ds['event_col']}) ~ " + " + ".join(terms)


def _mgcv_survival_formula_for_scenario(scenario_name: str, ds: dict[str, typing.Any]) -> str:
    cfg = _survival_formula_mapping(scenario_name)
    k_val = cfg["knots"] + 4
    terms = list(cfg["linear_cols"])
    terms.extend(
        f"s({c}, bs='ps', k=min({k_val}, nrow(train_df)-1))" for c in cfg["smooth_cols"]
    )
    return f"{ds['time_col']} ~ " + " + ".join(terms)


def _append_formula_link_term(formula: str, link_name: str | None) -> str:
    if not link_name:
        return formula
    # Avoid duplicating explicit link(...) terms if callers already injected one.
    if re.search(r"(?<![A-Za-z0-9_])link\s*\(", formula):
        return formula
    if "~" not in formula:
        raise RuntimeError(f"cannot append link term to malformed formula: {formula!r}")
    lhs, rhs = formula.split("~", 1)
    rhs = rhs.strip()
    link_term = f"link(type={str(link_name).strip()})"
    if not rhs:
        return f"{lhs.strip()} ~ {link_term}"
    return f"{lhs.strip()} ~ {rhs} + {link_term}"


def _flexible_link_name(link_name: str) -> str:
    raw = str(link_name).strip()
    if raw.startswith("flexible(") and raw.endswith(")"):
        return raw
    return f"flexible({raw})"


def _default_rust_formula_link_for_family(family: str) -> str:
    if family == "binomial":
        return "probit"
    if family == "gaussian":
        return "identity"
    if family == "survival":
        return "probit"
    raise RuntimeError(f"unsupported family for flexible-link benchmark companion: {family}")


def _is_matern_rust_scenario(s_cfg: typing.Any) -> bool:
    cfg = _scenario_fit_mapping(s_cfg["name"])
    if cfg is None:
        return False
    return bool(_canonical_smooth_basis(cfg.get("smooth_basis", "ps")) == "matern")



