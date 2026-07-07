//! HTML surface for the SAE dossiers (App A Certified Feature Dossier + M6
//! Convergence Dossier).
//!
//! Like the rest of `gam-report`, this module is free of gam library types: it
//! renders from the already-serialized dossier JSON (`serde_json::Value`) that
//! `gam-sae` produces, so the report crate never depends on the SAE crate. The
//! output is a single self-contained HTML document (inline CSS, no external
//! assets) — the same "one artifact you can open" contract the JSON dossiers
//! provide, in a human-readable surface.

use serde_json::Value;
use std::fmt::Write as _;

/// Escape a string for safe inclusion in HTML text / attribute content.
fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Render an f64-ish JSON number compactly (6 significant figures), or `—`.
fn num(v: Option<&Value>) -> String {
    match v.and_then(Value::as_f64) {
        Some(x) if x.is_finite() => {
            if x == 0.0 {
                "0".to_string()
            } else if x.abs() >= 1e-3 && x.abs() < 1e6 {
                format!("{x:.4}")
            } else {
                format!("{x:.3e}")
            }
        }
        Some(_) => "∞/NaN".to_string(),
        None => "—".to_string(),
    }
}

/// Render an optional bool as a Yes/No badge.
fn badge_bool(v: Option<&Value>, true_good: bool) -> String {
    match v.and_then(Value::as_bool) {
        Some(b) => {
            let good = b == true_good;
            let cls = if good { "ok" } else { "warn" };
            let label = if b { "yes" } else { "no" };
            format!("<span class=\"badge {cls}\">{label}</span>")
        }
        None => "<span class=\"badge na\">—</span>".to_string(),
    }
}

fn str_field(v: Option<&Value>) -> String {
    v.and_then(Value::as_str).map(esc).unwrap_or_else(|| "—".to_string())
}

/// Build the convergence-dossier header section.
fn convergence_section(conv: &Value) -> String {
    let mut s = String::new();
    let clean = conv.get("converged_clean").and_then(Value::as_bool).unwrap_or(false);
    let clean_badge = if clean {
        "<span class=\"badge ok big\">CONVERGED CLEAN</span>"
    } else {
        "<span class=\"badge warn big\">REVIEW — guard fired or certificate open</span>"
    };
    let _ = write!(s, "<section class=\"card\"><h2>Convergence dossier {clean_badge}</h2>");

    // Optimality certificate.
    if let Some(opt) = conv.get("optimality").filter(|v| !v.is_null()) {
        let _ = write!(
            s,
            "<div class=\"grid\">\
             <div class=\"kv\"><span>optimality passes</span>{}</div>\
             <div class=\"kv\"><span>agreement (rel)</span><b>{}</b></div>\
             <div class=\"kv\"><span>‖∇V‖</span><b>{}</b></div>\
             <div class=\"kv\"><span>well-posed</span>{}</div>\
             <div class=\"kv\"><span>FD directional</span><b>{}</b></div>\
             <div class=\"kv\"><span>analytic directional</span><b>{}</b></div>\
             </div>",
            badge_bool(opt.get("passes"), true),
            num(opt.get("agreement_rel")),
            num(opt.get("grad_norm")),
            badge_bool(opt.get("well_posed"), true),
            num(opt.get("fd_directional")),
            num(opt.get("analytic_directional")),
        );
    } else if let Some(err) = conv.get("optimality_error").and_then(Value::as_str) {
        let _ = write!(s, "<p class=\"muted\">optimality certificate unavailable: {}</p>", esc(err));
    }

    // Guard verdict + probe telemetry.
    if let Some(g) = conv.get("guard_verdict") {
        let _ = write!(
            s,
            "<div class=\"grid\">\
             <div class=\"kv\"><span>returned outer state verbatim</span>{}</div>\
             <div class=\"kv\"><span>seed-basin fallback</span>{}</div>\
             <div class=\"kv\"><span>pristine-seed fallback</span>{}</div>\
             <div class=\"kv\"><span>charts canonicalized</span>{}</div>\
             </div>",
            badge_bool(g.get("returned_outer_state_verbatim"), true),
            badge_bool(g.get("used_seed_basin_fallback"), false),
            badge_bool(g.get("used_pristine_seed_fallback"), false),
            badge_bool(g.get("charts_canonicalized"), false),
        );
    }
    if let Some(p) = conv.get("outer_probe") {
        let _ = write!(
            s,
            "<div class=\"grid\">\
             <div class=\"kv\"><span>criterion calls</span><b>{}</b></div>\
             <div class=\"kv\"><span>FD probe calls</span><b>{}</b></div>\
             <div class=\"kv\"><span>infeasible total</span><b>{}</b></div>\
             <div class=\"kv\"><span>mutating probes (regression if &gt;0)</span><b>{}</b></div>\
             </div>",
            num(p.get("criterion_calls")),
            num(p.get("fd_probe_calls")),
            num(p.get("infeasible_total")),
            num(p.get("mutating_value_probes")),
        );
    }
    if let Some(w) = conv.get("curvature_walk").filter(|v| !v.is_null()) {
        let _ = write!(
            s,
            "<div class=\"grid\">\
             <div class=\"kv\"><span>curvature walk arrived (η=1)</span>{}</div>\
             <div class=\"kv\"><span>clean walk</span>{}</div>\
             <div class=\"kv\"><span>η steps</span><b>{}</b></div>\
             <div class=\"kv\"><span>reseeds</span><b>{}</b></div>\
             </div>",
            badge_bool(w.get("arrived"), true),
            badge_bool(w.get("clean_walk"), true),
            num(w.get("eta_steps")),
            num(w.get("reseeds")),
        );
    }

    // Collapse ledger.
    let empty: Vec<Value> = Vec::new();
    let collapse = conv
        .get("collapse_ledger")
        .and_then(Value::as_array)
        .unwrap_or(&empty);
    if collapse.is_empty() {
        let _ = write!(s, "<p class=\"muted\">collapse ledger: empty (no active-mass breaches)</p>");
    } else {
        let _ = write!(s, "<p class=\"muted\">collapse ledger: {} event(s)</p>", collapse.len());
    }
    s.push_str("</section>");
    s
}

/// Build one atom's card.
fn atom_card(atom: &Value) -> String {
    let mut s = String::new();
    let idx = atom.get("atom_index").and_then(Value::as_u64).unwrap_or(0);
    let name = str_field(atom.get("atom_name"));
    let contested = atom
        .get("topology")
        .and_then(|t| t.get("contested"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let flag = if contested {
        " <span class=\"badge warn\">contested</span>"
    } else {
        ""
    };
    let _ = write!(s, "<div class=\"atom\"><h3>#{idx} · {name}{flag}</h3><div class=\"grid\">");

    if let Some(t) = atom.get("trust").filter(|v| !v.is_null()) {
        let _ = write!(
            s,
            "<div class=\"kv\"><span>trust score</span><b>{}</b></div>\
             <div class=\"kv\"><span>tangent cond</span><b>{}</b></div>\
             <div class=\"kv\"><span>coverage</span><b>{}</b></div>\
             <div class=\"kv\"><span>activation freq</span><b>{}</b></div>",
            num(t.get("trust_score")),
            num(t.get("tangent_condition_score")),
            num(t.get("coverage")),
            num(t.get("activation_frequency")),
        );
    }
    if let Some(l) = atom.get("lens").filter(|v| !v.is_null()) {
        let _ = write!(
            s,
            "<div class=\"kv\"><span>presence</span><b>{}</b></div>\
             <div class=\"kv\"><span>coupling</span><b>{}</b></div>\
             <div class=\"kv\"><span>discrepancy</span><b>{}</b></div>\
             <div class=\"kv\"><span>represented-not-used</span>{}</div>",
            num(l.get("presence")),
            num(l.get("coupling")),
            num(l.get("discrepancy")),
            badge_bool(l.get("represented_not_used"), false),
        );
    }
    if let Some(sh) = atom.get("shape_band").filter(|v| !v.is_null()) {
        let _ = write!(
            s,
            "<div class=\"kv\"><span>mean band sd</span><b>{}</b></div>\
             <div class=\"kv\"><span>max band sd</span><b>{}</b></div>",
            num(sh.get("mean_band_sd")),
            num(sh.get("max_band_sd")),
        );
    }
    if let Some(tp) = atom.get("topology").filter(|v| !v.is_null()) {
        let _ = write!(
            s,
            "<div class=\"kv\"><span>measured Betti</span><b>{}</b></div>\
             <div class=\"kv\"><span>expected Betti</span><b>{}</b></div>\
             <div class=\"kv\"><span>H1 persistence</span><b>{}</b></div>",
            str_field(tp.get("measured_betti")),
            str_field(tp.get("expected_betti")),
            num(tp.get("dominant_h1_persistence")),
        );
        if let Some(nc) = tp.get("null_calibration").filter(|v| !v.is_null()) {
            let _ = write!(
                s,
                "<div class=\"kv\"><span>null p-value</span><b>{}</b></div>\
                 <div class=\"kv\"><span>spike-in power</span><b>{}</b></div>",
                num(nc.get("null_pvalue")),
                num(nc.get("spikein_power")),
            );
        }
    }
    if let Some(inf) = atom.get("inference").filter(|v| !v.is_null()) {
        let _ = write!(
            s,
            "<div class=\"kv\"><span>decoder variation</span><b>{}</b></div>\
             <div class=\"kv\"><span>log-e non-constant</span><b>{}</b></div>",
            num(inf.get("decoder_variation")),
            num(inf.get("log_e_nonconstant")),
        );
    }
    if let Some(st) = atom.get("steering").filter(|v| !v.is_null()) {
        let _ = write!(
            s,
            "<div class=\"kv\"><span>steering nats/Δt</span><b>{}</b></div>\
             <div class=\"kv\"><span>validity radius</span><b>{}</b></div>",
            num(st.get("predicted_nats")),
            num(st.get("validity_radius")),
        );
    }
    if let Some(cf) = atom.get("coordinate_fidelity").filter(|v| !v.is_null()) {
        let _ = write!(
            s,
            "<div class=\"kv\"><span>uniformity p</span><b>{}</b></div>\
             <div class=\"kv\"><span>arclength defect</span><b>{}</b></div>",
            num(cf.get("uniformity_p_value")),
            num(cf.get("arclength_defect")),
        );
    }
    s.push_str("</div>");
    if let Some(cp) = atom.get("contested_probe").filter(|v| !v.is_null()) {
        let _ = write!(
            s,
            "<div class=\"probe\"><b>designed probe:</b> {}<br><span class=\"muted\">{}</span></div>",
            str_field(cp.get("claim")),
            str_field(cp.get("designed_probe")),
        );
    }
    s.push_str("</div>");
    s
}

/// Render a self-contained HTML document for the dossiers.
///
/// `feature` is the App A Certified Feature Dossier JSON; `convergence` is the
/// optional M6 Convergence Dossier JSON. Both are the `serde_json::Value` forms
/// of the strings `gam-sae` produces. Missing / null fields degrade to `—`.
pub fn render_dossier_html(feature: &Value, convergence: Option<&Value>, title: &str) -> String {
    let mut body = String::new();

    if let Some(conv) = convergence {
        body.push_str(&convergence_section(conv));
    }

    // Feature-dossier header.
    let n_atoms = feature.get("n_atoms").and_then(Value::as_u64).unwrap_or(0);
    let _ = write!(body, "<section class=\"card\"><h2>Certified feature dossier · {n_atoms} atoms</h2>");
    if let Some(g) = feature.get("residual_gauge") {
        let _ = write!(
            body,
            "<div class=\"grid\">\
             <div class=\"kv\"><span>gauge group</span><b>{}</b></div>\
             <div class=\"kv\"><span>residual gauge dim</span><b>{}</b></div>\
             <div class=\"kv\"><span>diffeomorphism unpinned</span>{}</div>\
             <div class=\"kv\"><span>metric provenance</span><b>{}</b></div>\
             </div>",
            str_field(g.get("group_signature")),
            num(g.get("residual_gauge_dim")),
            badge_bool(g.get("diffeomorphism_unpinned"), false),
            str_field(g.get("metric_provenance")),
        );
    }
    if let Some(dl) = feature.get("description_length").filter(|v| !v.is_null()) {
        let _ = write!(
            body,
            "<div class=\"grid\">\
             <div class=\"kv\"><span>total bits</span><b>{}</b></div>\
             <div class=\"kv\"><span>bits/token</span><b>{}</b></div>\
             <div class=\"kv\"><span>code bits</span><b>{}</b></div>\
             <div class=\"kv\"><span>selection bits</span><b>{}</b></div>\
             </div>",
            num(dl.get("total_bits")),
            num(dl.get("bits_per_token")),
            num(dl.get("code_bits")),
            num(dl.get("selection_bits")),
        );
    }
    body.push_str("</section>");

    // Per-atom cards.
    let empty: Vec<Value> = Vec::new();
    let atoms = feature.get("atoms").and_then(Value::as_array).unwrap_or(&empty);
    body.push_str("<section class=\"card\"><h2>Atoms</h2>");
    for atom in atoms {
        body.push_str(&atom_card(atom));
    }
    body.push_str("</section>");

    format!(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">\
         <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\
         <title>{title}</title><style>{css}</style></head>\
         <body><header><h1>{title}</h1></header><main>{body}</main></body></html>",
        title = esc(title),
        css = DOSSIER_CSS,
        body = body,
    )
}

const DOSSIER_CSS: &str = "\
:root{color-scheme:light dark}\
*{box-sizing:border-box}\
body{font-family:Inter,system-ui,-apple-system,sans-serif;margin:0;background:#0f1115;color:#e6e8eb;line-height:1.5}\
header{padding:24px 32px;border-bottom:1px solid #262a33}\
h1{font-size:20px;margin:0}\
h2{font-size:15px;margin:0 0 12px;color:#c8ccd4}\
h3{font-size:14px;margin:0 0 8px;color:#e6e8eb}\
main{padding:24px 32px;max-width:1100px;margin:0 auto}\
.card{background:#171a21;border:1px solid #262a33;border-radius:10px;padding:18px 20px;margin-bottom:18px}\
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:8px 18px;margin:8px 0}\
.kv{display:flex;justify-content:space-between;gap:12px;font-size:13px;border-bottom:1px dotted #262a33;padding:3px 0}\
.kv span{color:#8a90a0}\
.kv b{font-variant-numeric:tabular-nums;font-weight:600}\
.atom{border:1px solid #262a33;border-radius:8px;padding:12px 14px;margin:10px 0;background:#12151b}\
.probe{margin-top:10px;padding:10px;border-left:3px solid #d19a3c;background:#1c1a14;font-size:13px;border-radius:4px}\
.muted{color:#8a90a0;font-size:13px}\
.badge{display:inline-block;padding:1px 8px;border-radius:999px;font-size:12px;font-weight:600}\
.badge.ok{background:#12351f;color:#5cc98a}\
.badge.warn{background:#3a2412;color:#e0a35c}\
.badge.na{background:#23262e;color:#8a90a0}\
.badge.big{font-size:12px;vertical-align:middle;margin-left:8px}\
@media(prefers-color-scheme:light){\
body{background:#f7f8fa;color:#1e2230}\
header{border-color:#e2e5ea}\
.card{background:#fff;border-color:#e2e5ea}\
.atom{background:#fafbfc;border-color:#e2e5ea}\
.kv{border-color:#eef0f3}.kv span{color:#6b7280}\
h2{color:#374151}h3{color:#1e2230}.muted{color:#6b7280}\
}\
";

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn renders_self_contained_dossier_html() {
        let convergence = json!({
            "schema": "gam-sae/convergence-dossier/v1",
            "converged_clean": true,
            "optimality": {"passes": true, "agreement_rel": 1.2e-5, "grad_norm": 3.1e-4,
                           "well_posed": true, "fd_directional": 0.5, "analytic_directional": 0.5},
            "guard_verdict": {"returned_outer_state_verbatim": true,
                              "used_seed_basin_fallback": false,
                              "used_pristine_seed_fallback": false,
                              "charts_canonicalized": false},
            "outer_probe": {"criterion_calls": 42, "fd_probe_calls": 8,
                            "infeasible_total": 0, "mutating_value_probes": 0},
            "curvature_walk": {"arrived": true, "clean_walk": true, "eta_steps": 7, "reseeds": 0},
            "collapse_ledger": []
        });
        let feature = json!({
            "schema": "gam-sae/certified-feature-dossier/v1",
            "n_atoms": 2,
            "residual_gauge": {"group_signature": "trivial", "residual_gauge_dim": 0,
                               "diffeomorphism_unpinned": false,
                               "metric_provenance": "OutputFisher"},
            "description_length": {"total_bits": 1234.5, "bits_per_token": 7.7,
                                   "code_bits": 1000.0, "selection_bits": 200.0},
            "atoms": [
                {"atom_index": 0, "atom_name": "circle_a",
                 "trust": {"trust_score": 0.8, "tangent_condition_score": 0.9,
                           "coverage": 0.95, "activation_frequency": 0.3},
                 "lens": {"presence": 0.7, "coupling": 0.6, "discrepancy": 0.1,
                          "represented_not_used": false},
                 "topology": {"measured_betti": "(b0=1, b1=1)", "expected_betti": "(b0=1, b1=1)",
                              "dominant_h1_persistence": 0.4, "contested": false,
                              "null_calibration": {"null_pvalue": 0.001, "spikein_power": 0.9}},
                 "steering": {"predicted_nats": 2.5, "validity_radius": 1.1}},
                {"atom_index": 1, "atom_name": "circle_b",
                 "topology": {"measured_betti": "(b0=2, b1=0)", "expected_betti": "(b0=1, b1=1)",
                              "dominant_h1_persistence": 0.05, "contested": true},
                 "contested_probe": {"claim": "measured topology disagrees",
                                     "designed_probe": "resample and re-run persistence"}}
            ]
        });

        let html = render_dossier_html(&feature, Some(&convergence), "OLMo PCA-64 dossier");
        assert!(html.starts_with("<!doctype html>"), "self-contained document");
        assert!(html.contains("CONVERGED CLEAN"), "convergence headline rendered");
        assert!(html.contains("circle_a") && html.contains("circle_b"), "atoms rendered");
        assert!(html.contains("contested"), "contested flag surfaced");
        assert!(html.contains("designed probe"), "designed probe surfaced");
        assert!(!html.contains("http://") && !html.contains("https://"), "no external assets");
    }
}
