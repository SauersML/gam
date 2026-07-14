pub mod sparkline;

use std::fs;
use std::path::{Path, PathBuf};

// Data structs: plain data only, no gam library types.
// main.rs is responsible for all model/fit computation and populating these.

pub struct ReportInput {
    pub model_path: String,
    pub family_name: String,
    pub model_class: String,
    pub formula: String,
    pub n_obs: Option<usize>,
    pub deviance: f64,
    pub reml_score: f64,
    pub iterations: usize,
    /// Human-readable P-IRLS / outer convergence status (e.g. "Converged",
    /// "Max iterations reached"). Plain text so report.rs stays free of gam
    /// library types; main.rs supplies `PirlsStatus::label()`.
    pub convergence_status: String,
    /// Whether the fit cleanly converged. Drives the visual flag on the
    /// convergence line — any non-converged state is highlighted.
    pub converged: bool,
    /// Final outer-objective gradient norm at the recorded solution, when the
    /// outer loop measured it (`None` for cache-hit / gradient-free exits).
    pub outer_gradient_norm: Option<f64>,
    /// First-order optimality certificate (#934), when the fit recorded one.
    /// Plain data so report.rs stays free of gam library types; main.rs maps
    /// the fit's `OuterCriterionCertificate` into this row.
    pub criterion_certificate: Option<CriterionCertificateRow>,
    /// Optional smoothing-forensics rows (#1892). These are diagnostic-only
    /// snapshots aligned to fitted terms/blocks and expose the quantities that
    /// distinguish over-smoothing mechanisms without changing any fit math.
    pub smoothing_forensics: Vec<SmoothingForensicsRow>,
    pub edf_total: f64,
    pub r_squared: Option<f64>,
    pub coefficients: Vec<CoefficientRow>,
    pub edf_blocks: Vec<EdfBlockRow>,
    pub continuous_order: Vec<ContinuousOrderRow>,
    pub anisotropic_scales: Vec<AnisotropicScalesRow>,
    pub measure_jet_spectra: Vec<MeasureJetSpectrumRow>,
    pub diagnostics: Option<DiagnosticsInput>,
    pub smooth_plots: Vec<SmoothPlotData>,
    pub alo: Option<AloData>,
    pub notes: Vec<String>,
}

pub struct EdfBlockRow {
    pub index: usize,
    pub edf: f64,
    pub role: Option<String>,
}

/// One row of the on-demand smoothing-forensics report (#1892).
///
/// All fields are plain renderer data. Paths are recorded as the sequence the
/// fitting layer can prove it used; saved models often only retain the final
/// value, in which case the path is a one-element vector rather than an
/// invented history. EDF is split into the value consumed by the model/report
/// criterion and the assembly/influence side when both channels are available.
pub struct SmoothingForensicsRow {
    pub term: String,
    pub lambda_path: Vec<f64>,
    pub sigma2_path: Vec<f64>,
    pub edf_criterion: Option<f64>,
    pub edf_assembly: Option<f64>,
    pub double_penalty_range: Option<f64>,
    pub double_penalty_null_space: Option<f64>,
    pub seed_screening: Vec<String>,
}

/// First-order optimality evidence carried into a rendered report.
///
/// Matrix-free EFS fits are certified by explicit root-equivalent fixed-point
/// equations rather than a gradient. Keeping the proof kind in the report row
/// prevents those residuals from being mislabeled as gradient evidence.
pub enum CriterionStationarityRow {
    AnalyticGradient {
        grad_norm: f64,
        projected_grad_norm: f64,
        bound: f64,
    },
    FixedPoint {
        residual_inf_norm: f64,
        projected_residual_inf_norm: f64,
        bound: f64,
        covered_coordinates: usize,
    },
}

impl CriterionStationarityRow {
    fn rendered_metrics(&self, stationary: bool) -> String {
        let relation = if stationary { "\u{2264}" } else { ">" };
        match self {
            Self::AnalyticGradient {
                grad_norm,
                projected_grad_norm,
                bound,
            } => format!(
                "|g|={grad_norm:.3e}, |Pg|={projected_grad_norm:.3e} {relation} bound={bound:.3e}"
            ),
            Self::FixedPoint {
                residual_inf_norm,
                projected_residual_inf_norm,
                bound,
                covered_coordinates,
            } => format!(
                "fixed-point |r|\u{221e}={residual_inf_norm:.3e}, |Pr|\u{221e}={projected_residual_inf_norm:.3e} {relation} bound={bound:.3e}, coordinates={covered_coordinates}"
            ),
        }
    }
}

/// Optimality certificate row (#934, post-FD-purge): the fit's analytic KKT
/// self-audit at the returned optimum, plus curvature-definiteness and λ-rail
/// facts. `stationary` / `clean` are precomputed verdicts so the renderer never
/// re-derives policy.
pub struct CriterionCertificateRow {
    pub stationarity: CriterionStationarityRow,
    pub hessian_psd: Option<bool>,
    pub lambdas_railed: Vec<usize>,
    pub stationary: bool,
    pub clean: bool,
}

pub struct CoefficientRow {
    pub index: usize,
    pub estimate: f64,
    pub std_error: Option<f64>,
}

pub struct ContinuousOrderRow {
    pub name: String,
    pub lambda0: f64,
    pub lambda1: f64,
    pub lambda2: f64,
    pub r_ratio: Option<f64>,
    pub nu: Option<f64>,
    pub kappa2: Option<f64>,
    pub status: String,
}

pub struct AnisotropicScalesRow {
    pub term_name: String,
    pub global_length_scale: Option<f64>,
    /// Per-axis: (axis_index, eta, per_axis_length_scale, per_axis_kappa)
    pub axes: Vec<(usize, f64, Option<f64>, Option<f64>)>,
}

/// Measure-jet scale spectrum row: the realized multiscale band of one
/// measure-jet term, plus — in per-scale-candidate mode — the fitted
/// physical λ̂_ℓ per scale and the implied continuous order
/// ŝ = −½ · (least-squares slope of ln λ̂_ℓ on ln ε_ℓ). `per_scale` empty
/// means the term carries a single fused jet-energy penalty, so only the
/// band and the spec's order are shown. main.rs computes everything
/// (`measure_jet_implied_order` derives ŝ); this row stays plain data like
/// the rest of the file.
pub struct MeasureJetSpectrumRow {
    pub term_name: String,
    pub eps_min: f64,
    pub eps_max: f64,
    pub n_scales: usize,
    pub length_scale: f64,
    pub spec_order_s: f64,
    /// Per-scale (ε_ℓ, physical λ̂_ℓ) pairs, ascending in ε; empty = fused.
    pub per_scale: Vec<(f64, f64)>,
    pub implied_order: Option<f64>,
}

pub struct DiagnosticsInput {
    /// Family-appropriate residuals on the N(0,1) scale, or `None` when the
    /// family has no implemented residual distribution — the residual plots
    /// are then omitted instead of rendered against a false normal reference.
    pub residuals: Option<ResidualDiagnostics>,
    pub y_observed: Vec<f64>,
    pub y_predicted: Vec<f64>,
    pub calibration: Option<CalibrationData>,
}

/// Residuals whose null distribution is standard normal under a correct
/// model (randomized-quantile / deviance, per family). main.rs computes the
/// definition; this struct is plain plotting data. `values` is aligned with
/// `DiagnosticsInput::y_observed`; `sorted` and `theoretical_quantiles` are
/// the Q-Q pairing at the same plotting positions.
pub struct ResidualDiagnostics {
    pub values: Vec<f64>,
    pub sorted: Vec<f64>,
    pub theoretical_quantiles: Vec<f64>,
    /// Axis label naming the residual definition, e.g.
    /// "Randomized Quantile Residual".
    pub label: String,
}

/// Equal-count calibration bins (true deciles when n ≥ 10): observations are
/// sorted by predicted probability and split into groups of equal size, so
/// every point summarises the same number of observations.
pub struct CalibrationData {
    pub mean_predicted: Vec<f64>,
    pub observed_rate: Vec<f64>,
}

pub struct AloData {
    /// Affine fitted-likelihood coordinates, aligned with every row vector.
    pub coordinate_names: Vec<String>,
    pub rows: Vec<AloRow>,
}

pub struct AloRow {
    pub index: usize,
    pub leverage: f64,
    pub eta_tilde: Vec<f64>,
    pub standard_errors: Vec<f64>,
    pub cook_distance: f64,
}

pub struct SmoothPlotData {
    pub name: String,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
}

/// Write the report HTML to the output path, returning the path written.
pub fn write_report(
    input: &ReportInput,
    out: Option<&Path>,
    model_path: &Path,
) -> Result<PathBuf, String> {
    let out = out.map(PathBuf::from).unwrap_or_else(|| {
        let stem = model_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model");
        PathBuf::from(format!("{stem}.report.html"))
    });
    let html = render_html(input)?;
    fs::write(&out, html)
        .map_err(|e| format!("failed to write report '{}': {e}", out.display()))?;
    Ok(out)
}

pub fn render_html(input: &ReportInput) -> Result<String, String> {
    let json = |v: &[f64]| serde_json::to_string(v).map_err(|e| e.to_string());

    let mut scripts = Vec::new();
    let plot_cfg =
        "responsive:true,displaylogo:false,modeBarButtonsToRemove:['lasso2d','select2d']";
    let plot_style = |title: &str, xtitle: &str, ytitle: &str| {
        format!(
            "{{margin:{{t:44,b:48,l:56,r:24}},\
             font:{{family:'Inter,system-ui,sans-serif'}},\
             title:{{text:'{title}',font:{{size:14,color:'#1e293b'}}}},\
             xaxis:{{title:'{xtitle}',gridcolor:'#f0f0f0',zeroline:false}},\
             yaxis:{{title:'{ytitle}',gridcolor:'#f0f0f0',zeroline:false}},\
             plot_bgcolor:'#fafafa',paper_bgcolor:'white',hoverlabel:{{font:{{size:12}}}}}}"
        )
    };
    let marker = "marker:{color:'#6366f1',size:4,opacity:0.6}";

    if let Some(diag) = &input.diagnostics {
        if let Some(resid) = &diag.residuals {
            // QQ plot: residuals are already on the N(0,1) scale, so the
            // standard-normal reference line is the calibration target for
            // every family, not just Gaussian.
            scripts.push(format!(
                "Plotly.newPlot('qq_plot',\
                 [{{x:{theo},y:{res},mode:'markers',type:'scattergl',{marker}}},\
                 {{x:{theo},y:{theo},mode:'lines',showlegend:false,\
                 line:{{color:'#cbd5e1',width:1,dash:'dash'}}}}],\
                 {layout},{{{cfg}}});",
                theo = json(&resid.theoretical_quantiles)?,
                res = json(&resid.sorted)?,
                marker = marker,
                layout = plot_style(
                    "Normal Q-Q",
                    "Theoretical Quantile",
                    &format!("Sample Quantile ({})", js_escape(&resid.label))
                ),
                cfg = plot_cfg,
            ));

            // Residuals vs fitted
            let fit_min = diag
                .y_predicted
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min);
            let fit_max = diag
                .y_predicted
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            scripts.push(format!(
                "Plotly.newPlot('resid_fitted',\
                 [{{x:{fitted},y:{resid},mode:'markers',type:'scattergl',{marker}}}],\
                 Object.assign({layout},{{shapes:[{{type:'line',x0:{fit_min},x1:{fit_max},\
                 y0:0,y1:0,line:{{color:'#cbd5e1',width:1,dash:'dash'}}}}]}}),\
                 {{{cfg}}});",
                fitted = json(&diag.y_predicted)?,
                resid = json(&resid.values)?,
                marker = marker,
                layout = plot_style(
                    "Residuals vs Fitted",
                    "Fitted Value",
                    &js_escape(&resid.label)
                ),
                fit_min = fit_min,
                fit_max = fit_max,
                cfg = plot_cfg,
            ));
        }

        // Observed vs Predicted
        let range_min = diag
            .y_observed
            .iter()
            .chain(diag.y_predicted.iter())
            .copied()
            .fold(f64::INFINITY, f64::min);
        let range_max = diag
            .y_observed
            .iter()
            .chain(diag.y_predicted.iter())
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        scripts.push(format!(
            "Plotly.newPlot('obs_pred',\
             [{{x:{pred},y:{obs},mode:'markers',type:'scattergl',{marker}}},\
             {{x:[{lo},{hi}],y:[{lo},{hi}],mode:'lines',\
             line:{{color:'#cbd5e1',width:1,dash:'dash'}},showlegend:false}}],\
             {layout},{{{cfg}}});",
            pred = json(&diag.y_predicted)?,
            obs = json(&diag.y_observed)?,
            marker = marker,
            lo = range_min,
            hi = range_max,
            layout = plot_style("Observed vs Predicted", "Predicted", "Observed"),
            cfg = plot_cfg,
        ));

        if let Some(resid) = &diag.residuals {
            // Residual histogram
            scripts.push(format!(
                "Plotly.newPlot('resid_hist',\
                 [{{x:{resid},type:'histogram',\
                 marker:{{color:'#6366f1',line:{{color:'#4f46e5',width:0.5}}}},opacity:0.85}}],\
                 {layout},{{{cfg}}});",
                resid = json(&resid.values)?,
                layout = plot_style(
                    "Residual Distribution",
                    &js_escape(&resid.label),
                    "Frequency"
                ),
                cfg = plot_cfg,
            ));

            // Scale-location plot: sqrt(|residual|) vs fitted. The residuals
            // are already standardized to unit scale by their family-specific
            // definition, so no further sample-SD rescaling is applied — a
            // trend here is genuine scale misfit, not the variance function.
            let sqrt_abs = resid
                .values
                .iter()
                .map(|r| r.abs().sqrt())
                .collect::<Vec<_>>();
            scripts.push(format!(
                "Plotly.newPlot('scale_loc',\
                 [{{x:{fitted},y:{sqrt_abs},mode:'markers',type:'scattergl',{marker}}}],\
                 {layout},{{{cfg}}});",
                fitted = json(&diag.y_predicted)?,
                sqrt_abs = json(&sqrt_abs)?,
                marker = marker,
                layout = plot_style(
                    "Scale-Location",
                    "Fitted Value",
                    &format!("&radic;|{}|", js_escape(&resid.label))
                ),
                cfg = plot_cfg,
            ));
        }

        // Calibration (binary only)
        if let Some(cal) = &diag.calibration {
            scripts.push(format!(
                "Plotly.newPlot('cal_plot',\
                 [{{x:{x},y:{y},mode:'markers+lines',type:'scatter',\
                 marker:{{color:'#6366f1',size:7}},line:{{color:'#6366f1',width:2}}}},\
                 {{x:[0,1],y:[0,1],mode:'lines',showlegend:false,\
                 line:{{color:'#cbd5e1',width:1,dash:'dash'}}}}],\
                 Object.assign({layout},\
                 {{xaxis:{{title:'Mean Predicted',range:[-0.02,1.02],gridcolor:'#f0f0f0'}},\
                 yaxis:{{title:'Observed Rate',range:[-0.02,1.02],gridcolor:'#f0f0f0'}}}}),\
                 {{{cfg}}});",
                x = json(&cal.mean_predicted)?,
                y = json(&cal.observed_rate)?,
                layout = plot_style("Calibration (Deciles)", "Mean Predicted", "Observed Rate"),
                cfg = plot_cfg,
            ));
        }
    }

    // Smooth term partial-effect plots
    for sp in &input.smooth_plots {
        let div_id = format!("smooth_{}", to_html_id(&sp.name));
        scripts.push(format!(
            "Plotly.newPlot('{div_id}',\
             [{{x:{x},y:{y},mode:'lines',type:'scatter',\
             line:{{color:'#6366f1',width:2.5}}}}],\
             {layout},{{{cfg}}});",
            div_id = div_id,
            x = json(&sp.x)?,
            y = json(&sp.y)?,
            layout = plot_style(
                &format!("s({})", js_escape(&sp.name)),
                &js_escape(&sp.name),
                "Partial Effect",
            ),
            cfg = plot_cfg,
        ));
    }

    // HTML sections
    let notes_html = if input.notes.is_empty() {
        String::new()
    } else {
        format!(
            "<div class=\"alert\">{}</div>",
            input
                .notes
                .iter()
                .map(|n| esc(n))
                .collect::<Vec<_>>()
                .join("<br/>")
        )
    };

    // Summary card: key-value pairs, 2-column grid
    let mut summary_pairs: Vec<(&str, String)> = vec![
        ("Family", esc(&input.family_name)),
        ("Model Class", esc(&input.model_class)),
    ];
    if let Some(n) = input.n_obs {
        summary_pairs.push(("Observations", format!("{}", n)));
    }
    summary_pairs.push(("Deviance", fmt_num(input.deviance)));
    summary_pairs.push(("REML / LAML", fmt_num(input.reml_score)));
    if let Some(r2) = input.r_squared {
        summary_pairs.push(("R-squared", format!("{:.6}", r2)));
    }
    summary_pairs.push(("EDF (total)", format!("{:.4}", input.edf_total)));
    // Outer iterations, annotated with the cap when the solver did not
    // converge cleanly so "47" cannot be misread as "converged at 47".
    let iter_value = if input.converged {
        format!("{}", input.iterations)
    } else {
        format!(
            "{} <span class=\"conv-warn\">(did not converge)</span>",
            input.iterations
        )
    };
    summary_pairs.push(("Outer Iterations", iter_value));
    // Convergence status: always shown, visually flagged when not `Converged`,
    // so a reader can immediately tell a healthy fit from one that hit the
    // iteration cap, exhausted the LM step search, or went unstable.
    let conv_value = if input.converged {
        format!(
            "<span class=\"conv-ok\">{}</span>",
            esc(&input.convergence_status)
        )
    } else {
        format!(
            "<span class=\"conv-warn\">\u{26A0} {}</span>",
            esc(&input.convergence_status)
        )
    };
    summary_pairs.push(("Convergence", conv_value));
    if let Some(g) = input.outer_gradient_norm {
        summary_pairs.push(("Outer Gradient Norm", format!("{g:.3e}")));
    }
    // Optimality certificate (#934): the fit's analytic KKT self-audit at the
    // optimum. A stationarity flag here names the broken criterion the moment
    // it is introduced — surface it as loudly as non-convergence.
    if let Some(cert) = &input.criterion_certificate {
        let cert_value = if cert.clean {
            format!(
                "<span class=\"conv-ok\">stationary</span> ({})",
                cert.stationarity.rendered_metrics(true)
            )
        } else {
            let mut flags = Vec::new();
            if !cert.stationary {
                flags.push(format!(
                    "non-stationary ({})",
                    cert.stationarity.rendered_metrics(false)
                ));
            }
            if cert.hessian_psd == Some(false) {
                flags.push("outer Hessian not positive semidefinite".to_string());
            }
            if !cert.lambdas_railed.is_empty() {
                flags.push(format!(
                    "\u{03BB} railed at bound: {:?}",
                    cert.lambdas_railed
                ));
            }
            format!(
                "<span class=\"conv-warn\">\u{26A0} {}</span>",
                esc(&flags.join("; "))
            )
        };
        summary_pairs.push(("Optimality Certificate", cert_value));
    }

    let summary_items = summary_pairs
        .iter()
        .map(|(k, v)| format!("<div class=\"stat-item\"><span class=\"stat-label\">{k}</span><span class=\"stat-value\">{v}</span></div>"))
        .collect::<Vec<_>>()
        .join("\n");

    let formula_html = format!("<code class=\"formula\">{}</code>", esc(&input.formula));

    // Coefficients table — collapsible if > 20 rows
    let n_coef = input.coefficients.len();
    let coef_rows = input
        .coefficients
        .iter()
        .map(|c| {
            let se_str = c.std_error.map(|v| format!("{v:.6e}")).unwrap_or_else(|| "\u{2014}".to_string());
            let z_str = c.std_error
                .filter(|&se| se.abs() > 1e-15)
                .map(|se| format!("{:.3}", c.estimate / se))
                .unwrap_or_else(|| "\u{2014}".to_string());
            format!(
                "<tr><td class=\"mono\">{}</td><td class=\"num\">{:.6e}</td><td class=\"num\">{}</td><td class=\"num\">{}</td></tr>",
                c.index, c.estimate, se_str, z_str
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let coef_table = format!(
        "<div class=\"table-wrap\"><table>\n\
         <thead><tr><th>#</th><th>Estimate</th><th>Std. Error</th><th>z</th></tr></thead>\n\
         <tbody>\n{coef_rows}\n</tbody>\n</table></div>"
    );
    let coef_body = if n_coef > 20 {
        format!(
            "<details><summary class=\"toggle\">{n_coef} coefficients (click to expand)</summary>\n{coef_table}\n</details>"
        )
    } else {
        format!("<p class=\"muted\">{n_coef} parameters</p>\n{coef_table}")
    };

    // EDF blocks
    let has_roles = input.edf_blocks.iter().any(|b| b.role.is_some());
    let edf_rows = input
        .edf_blocks
        .iter()
        .map(|b| {
            if has_roles {
                let role_label = b.role.as_deref().unwrap_or("\u{2014}");
                format!(
                    "<tr><td class=\"mono\">{}</td><td>{}</td><td class=\"num\">{:.4}</td></tr>",
                    b.index,
                    esc(role_label),
                    b.edf
                )
            } else {
                format!(
                    "<tr><td class=\"mono\">{}</td><td class=\"num\">{:.4}</td></tr>",
                    b.index, b.edf
                )
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    let edf_header = if has_roles {
        "<thead><tr><th>Block</th><th>Role</th><th>EDF</th></tr></thead>"
    } else {
        "<thead><tr><th>Block</th><th>EDF</th></tr></thead>"
    };
    let edf_section = format!(
        "<section class=\"card\" id=\"sec-edf\">\n\
         <h2>EDF by Penalty Block</h2>\n\
         <div class=\"table-wrap\"><table>\n\
         {edf_header}\n\
         <tbody>{edf_rows}</tbody>\n\
         </table></div>\n\
         </section>"
    );

    // Continuous smoothness order (only if present)
    let continuous_section = if input.continuous_order.is_empty() {
        String::new()
    } else {
        let rows = input
            .continuous_order
            .iter()
            .map(|c| {
                let f = |v: Option<f64>| {
                    v.map(|x| format!("{x:.4e}"))
                        .unwrap_or_else(|| "\u{2014}".to_string())
                };
                format!(
                    "<tr><td>{}</td><td class=\"num\">{:.4e}</td><td class=\"num\">{:.4e}</td>\
                 <td class=\"num\">{:.4e}</td><td class=\"num\">{}</td><td class=\"num\">{}</td>\
                 <td class=\"num\">{}</td><td class=\"status\">{}</td></tr>",
                    esc(&c.name),
                    c.lambda0,
                    c.lambda1,
                    c.lambda2,
                    f(c.r_ratio),
                    f(c.nu),
                    f(c.kappa2),
                    esc(&c.status),
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            "<section class=\"card\" id=\"sec-cont-order\">\n\
             <h2>Continuous Smoothness Order</h2>\n\
             <div class=\"table-wrap\"><table>\n\
             <thead><tr><th>Term</th><th>&lambda;<sub>0</sub></th><th>&lambda;<sub>1</sub></th>\
             <th>&lambda;<sub>2</sub></th><th>R</th><th>&nu;</th><th>&kappa;&sup2;</th><th>Status</th></tr></thead>\n\
             <tbody>{rows}</tbody>\n</table></div>\n</section>"
        )
    };

    // Anisotropic spatial geometry section
    let aniso_section = if input.anisotropic_scales.is_empty() {
        String::new()
    } else {
        let rows = input
            .anisotropic_scales
            .iter()
            .flat_map(|row| {
                let header = match row.global_length_scale {
                    Some(length_scale) => format!(
                        "{} (global \u{2113}={length_scale:.4})",
                        esc(&row.term_name),
                    ),
                    None => format!(
                        "{} (pure Duchon shape-only anisotropy)",
                        esc(&row.term_name)
                    ),
                };
                let mut out = vec![format!(
                    "<tr><td colspan=\"5\" style=\"font-weight:600\">{header}</td></tr>"
                )];
                for &(axis, eta, length, kappa) in &row.axes {
                    let length = length
                        .map(|value| format!("{value:.4}"))
                        .unwrap_or_else(|| "\u{2014}".to_string());
                    let kappa = kappa
                        .map(|value| format!("{value:.4}"))
                        .unwrap_or_else(|| "\u{2014}".to_string());
                    out.push(format!(
                        "<tr><td style=\"padding-left:2em\">axis {axis}</td>\
                         <td class=\"num\">{eta:+.4}</td>\
                         <td class=\"num\">{length}</td>\
                         <td class=\"num\">{kappa}</td>\
                         <td></td></tr>"
                    ));
                }
                out
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            "<section class=\"card\" id=\"sec-aniso-scales\">\n\
             <h2>Anisotropic Spatial Geometry</h2>\n\
             <p class=\"muted\">Pure Duchon terms are scale-free, so only the centered axis contrasts (&eta;) are reported; \u{2113} and &kappa; are shown only for terms with a global length scale.</p>\n\
             <div class=\"table-wrap\"><table>\n\
             <thead><tr><th>Term / Axis</th><th>&eta;</th><th>\u{2113}</th><th>&kappa;</th><th></th></tr></thead>\n\
             <tbody>{rows}</tbody>\n</table></div>\n</section>"
        )
    };

    // Measure-jet scale spectrum: one compact line per term (only if present).
    // Two realized shapes, matching the single-scale/multiscale opt-in (#1116): a
    // single fused penalty (single-scale mode, the default at any center count
    // unless `multiscale=true` — see `measure_jet_multiscale_mode`) carries an
    // empty `per_scale`, so we print just the band and the spec order and skip the
    // slope readout; the per-scale spectrum (multiscale mode) prints the fitted
    // lambda_l and the implied order.
    let measure_jet_section = if input.measure_jet_spectra.is_empty() {
        String::new()
    } else {
        let lines = input
            .measure_jet_spectra
            .iter()
            .map(|r| {
                let band = format!(
                    "band {}..{} ({} scales, \u{2113}={})",
                    fmt_num(r.eps_min),
                    fmt_num(r.eps_max),
                    r.n_scales,
                    fmt_num(r.length_scale),
                );
                let tail = if r.per_scale.is_empty() {
                    format!("fused penalty, spec order s={:.2}", r.spec_order_s)
                } else {
                    let lams = r
                        .per_scale
                        .iter()
                        .map(|&(_, lam)| format!("{lam:.3e}"))
                        .collect::<Vec<_>>()
                        .join(", ");
                    let implied = r
                        .implied_order
                        .map(|s| format!("implied order s\u{0302}\u{2248}{s:.2}"))
                        .unwrap_or_else(|| "implied order \u{2014}".to_string());
                    format!(
                        "&lambda;<sub>\u{2113}</sub> = [{lams}], {implied} (spec s={:.2})",
                        r.spec_order_s
                    )
                };
                format!(
                    "<p class=\"mono\">{}: measure-jet {band}, {tail}</p>",
                    esc(&r.term_name)
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            "<section class=\"card\" id=\"sec-mjet-spectrum\">\n\
             <h2>Measure-Jet Scale Spectrum</h2>\n\
             {lines}\n</section>"
        )
    };

    // Smoothing forensics (#1892): compact, numeric rows used to distinguish
    // dispersion→λ feedback, criterion/assembly EDF drift, double-penalty
    // null-space shrinkage, and seed-screening failures. Empty means the saved
    // fit did not retain enough diagnostic state; non-empty rows are rendered
    // verbatim without re-deriving fit quantities in the renderer.
    let smoothing_forensics_section = if input.smoothing_forensics.is_empty() {
        String::new()
    } else {
        let fmt_path = |values: &[f64]| {
            if values.is_empty() {
                "—".to_string()
            } else {
                values
                    .iter()
                    .map(|value| fmt_num(*value))
                    .collect::<Vec<_>>()
                    .join(" → ")
            }
        };
        let fmt_opt = |value: Option<f64>| value.map(fmt_num).unwrap_or_else(|| "—".to_string());
        let rows = input
            .smoothing_forensics
            .iter()
            .map(|row| {
                let seeds = if row.seed_screening.is_empty() {
                    "—".to_string()
                } else {
                    esc(&row.seed_screening.join("; "))
                };
                format!(
                    "<tr><td>{}</td><td class=\"mono\">{}</td><td class=\"mono\">{}</td>\
                     <td class=\"num\">{}</td><td class=\"num\">{}</td>\
                     <td class=\"num\">{}</td><td class=\"num\">{}</td><td>{}</td></tr>",
                    esc(&row.term),
                    esc(&fmt_path(&row.lambda_path)),
                    esc(&fmt_path(&row.sigma2_path)),
                    fmt_opt(row.edf_criterion),
                    fmt_opt(row.edf_assembly),
                    fmt_opt(row.double_penalty_range),
                    fmt_opt(row.double_penalty_null_space),
                    seeds,
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            "<section class=\"card\" id=\"sec-smoothing-forensics\">\n\
             <h2>Smoothing Forensics</h2>\n\
             <p class=\"muted\">Diagnostic-only state for over-smoothing investigations: λ and σ² paths, criterion-vs-assembly EDF, double-penalty split, and seed-screening verdicts.</p>\n\
             <div class=\"table-wrap\"><table>\n\
             <thead><tr><th>Term</th><th>λ path</th><th>σ² path</th><th>EDF criterion</th><th>EDF assembly</th><th>Range penalty</th><th>Null-space penalty</th><th>Seed screening</th></tr></thead>\n\
             <tbody>{rows}</tbody>\n</table></div>\n</section>"
        )
    };

    // Diagnostics plots grid: residual plots only exist when a residual
    // definition with a standard-normal null is available for the family.
    let diagnostics_section = if let Some(diag) = &input.diagnostics {
        let resid_divs = if diag.residuals.is_some() {
            "<div id=\"qq_plot\" class=\"plot\"></div>\n\
               <div id=\"resid_fitted\" class=\"plot\"></div>\n\
               <div id=\"resid_hist\" class=\"plot\"></div>\n\
               <div id=\"scale_loc\" class=\"plot\"></div>\n"
        } else {
            ""
        };
        let cal_div = if diag.calibration.is_some() {
            "<div id=\"cal_plot\" class=\"plot\"></div>"
        } else {
            ""
        };
        format!(
            "<section class=\"card\" id=\"sec-diagnostics\">\n\
             <h2>Diagnostics</h2>\n\
             <div class=\"plot-grid\">\n\
               {resid_divs}\
               <div id=\"obs_pred\" class=\"plot\"></div>\n\
               {cal_div}\n\
             </div>\n</section>"
        )
    } else {
        String::new()
    };

    // Smooth term plots
    let smooth_section = if input.smooth_plots.is_empty() {
        String::new()
    } else {
        let divs = input
            .smooth_plots
            .iter()
            .map(|sp| {
                format!(
                    "<div id=\"smooth_{}\" class=\"plot\"></div>",
                    to_html_id(&sp.name)
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            "<section class=\"card\" id=\"sec-smooth\">\n\
             <h2>Smooth Terms</h2>\n\
             <div class=\"plot-grid\">{divs}</div>\n</section>"
        )
    };

    // ALO diagnostics table (only if present)
    let alo_section = if let Some(alo) = &input.alo {
        if alo.coordinate_names.is_empty() {
            return Err("ALO report data requires at least one coordinate name".to_string());
        }
        for row in &alo.rows {
            if row.eta_tilde.len() != alo.coordinate_names.len()
                || row.standard_errors.len() != alo.coordinate_names.len()
            {
                return Err(format!(
                    "ALO report row {} has coordinate lengths eta={}, se={}; expected {}",
                    row.index,
                    row.eta_tilde.len(),
                    row.standard_errors.len(),
                    alo.coordinate_names.len()
                ));
            }
        }
        let max_show = 100;
        let n_show = alo.rows.len().min(max_show);
        let rows = alo.rows[..n_show]
            .iter()
            .map(|r| {
                let coordinates = alo
                    .coordinate_names
                    .iter()
                    .zip(&r.eta_tilde)
                    .map(|(name, value)| format!("{}={value:.6e}", esc(name)))
                    .collect::<Vec<_>>()
                    .join("<br>");
                let standard_errors = alo
                    .coordinate_names
                    .iter()
                    .zip(&r.standard_errors)
                    .map(|(name, value)| format!("{}={value:.6e}", esc(name)))
                    .collect::<Vec<_>>()
                    .join("<br>");
                format!(
                    "<tr><td class=\"mono\">{}</td><td class=\"num\">{:.6e}</td>\
                 <td class=\"num\">{coordinates}</td><td class=\"num\">{standard_errors}</td>\
                 <td class=\"num\">{:.6e}</td></tr>",
                    r.index, r.leverage, r.cook_distance
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        let truncation_note = if alo.rows.len() > max_show {
            format!(
                "<p class=\"muted\">Showing first {n_show} of {} rows</p>",
                alo.rows.len()
            )
        } else {
            String::new()
        };
        format!(
            "<section class=\"card\" id=\"sec-alo\">\n\
             <h2>ALO Diagnostics</h2>\n\
             {truncation_note}\n\
             <div class=\"table-wrap\"><table>\n\
             <thead><tr><th>Row</th><th>Leverage</th><th>Deleted-row coordinates</th><th>ALO SE</th><th>Cook distance</th></tr></thead>\n\
             <tbody>{rows}</tbody>\n</table></div>\n</section>"
        )
    } else {
        String::new()
    };

    // Build nav links for present sections
    let mut nav_items = vec![
        ("sec-summary", "Summary"),
        ("sec-coef", "Coefficients"),
        ("sec-edf", "EDF"),
    ];
    if !input.continuous_order.is_empty() {
        nav_items.push(("sec-cont-order", "Smoothness Order"));
    }
    if !input.anisotropic_scales.is_empty() {
        nav_items.push(("sec-aniso-scales", "Anisotropy"));
    }
    if !input.measure_jet_spectra.is_empty() {
        nav_items.push(("sec-mjet-spectrum", "Measure-Jet"));
    }
    if !input.smoothing_forensics.is_empty() {
        nav_items.push(("sec-smoothing-forensics", "Forensics"));
    }
    if input.diagnostics.is_some() {
        nav_items.push(("sec-diagnostics", "Diagnostics"));
    }
    if !input.smooth_plots.is_empty() {
        nav_items.push(("sec-smooth", "Smooth Terms"));
    }
    if input.alo.is_some() {
        nav_items.push(("sec-alo", "ALO"));
    }
    let nav_links = nav_items
        .iter()
        .map(|(id, label)| format!("<a href=\"#{id}\">{label}</a>"))
        .collect::<Vec<_>>()
        .join("");

    Ok(format!(
        r##"<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>GAM Report</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root {{
  --bg: #f8fafc; --card: #fff; --border: #e2e8f0;
  --accent: #6366f1; --accent-light: #eef2ff; --accent-dark: #4f46e5;
  --text: #1e293b; --text2: #475569; --text3: #94a3b8;
  --font: 'Inter',ui-sans-serif,system-ui,-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
  --mono: 'SF Mono','JetBrains Mono','Fira Code',ui-monospace,monospace;
  --shadow: 0 1px 3px rgba(0,0,0,.06),0 1px 2px rgba(0,0,0,.04);
  --radius: 10px;
}}
*,*::before,*::after {{ margin:0;padding:0;box-sizing:border-box; }}
html {{ scroll-behavior:smooth; }}
body {{ font-family:var(--font);background:var(--bg);color:var(--text);line-height:1.6;-webkit-font-smoothing:antialiased; }}

.header {{
  background:linear-gradient(135deg,#1e293b 0%,#334155 100%);
  color:#f8fafc; padding:28px 32px 20px;
}}
.header h1 {{ font-size:24px;font-weight:700;letter-spacing:-.03em; }}
.header .subtitle {{ color:var(--text3);font-size:13px;margin-top:2px; }}
.nav {{
  display:flex;gap:2px;margin-top:16px;flex-wrap:wrap;
}}
.nav a {{
  color:#cbd5e1;text-decoration:none;font-size:12px;font-weight:500;
  padding:5px 12px;border-radius:6px;transition:all .15s;
}}
.nav a:hover {{ background:rgba(255,255,255,.1);color:#f1f5f9; }}

.container {{ max-width:1200px;margin:0 auto;padding:20px 24px 48px; }}

.card {{
  background:var(--card);border:1px solid var(--border);
  border-radius:var(--radius);padding:24px;margin-bottom:16px;
  box-shadow:var(--shadow);
}}
.card h2 {{
  font-size:15px;font-weight:600;color:var(--text);
  margin-bottom:16px;padding-bottom:10px;
  border-bottom:2px solid var(--accent-light);
  display:flex;align-items:center;gap:8px;
}}

.formula {{
  font-family:var(--mono);font-size:13px;
  background:var(--accent-light);color:var(--accent-dark);
  padding:8px 14px;border-radius:6px;display:block;
  margin-bottom:4px;word-break:break-word;line-height:1.8;
}}

.stat-grid {{
  display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px;
}}
.stat-item {{
  display:flex;flex-direction:column;padding:10px 14px;
  border-radius:8px;background:var(--bg);border:1px solid var(--border);
}}
.stat-label {{ font-size:11px;font-weight:600;color:var(--text3);text-transform:uppercase;letter-spacing:.05em; }}
.stat-value {{ font-size:15px;font-weight:600;color:var(--text);font-variant-numeric:tabular-nums;margin-top:2px; }}
.conv-ok {{ color:#15803d; }}
.conv-warn {{ color:#b45309;font-weight:700; }}

.alert {{
  background:#fffbeb;border:1px solid #fde68a;border-radius:var(--radius);
  padding:12px 16px;margin-bottom:16px;font-size:13px;color:#92400e;line-height:1.5;
}}

.table-wrap {{ overflow-x:auto;border-radius:8px;border:1px solid var(--border); }}
table {{ border-collapse:collapse;width:100%;font-size:13px; }}
thead th {{
  background:var(--accent-light);color:var(--accent-dark);
  font-weight:600;text-align:left;padding:9px 12px;
  border-bottom:2px solid var(--border);white-space:nowrap;
  position:sticky;top:0;
}}
tbody td {{ padding:7px 12px;border-bottom:1px solid #f1f5f9; }}
tbody tr:last-child td {{ border-bottom:none; }}
tbody tr:hover {{ background:#fafbfc; }}
td.num {{ text-align:right;font-variant-numeric:tabular-nums;font-family:var(--mono);font-size:12px; }}
td.mono,.mono {{ font-family:var(--mono);font-size:12px; }}
td.status {{
  font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.04em;
  color:var(--accent);
}}

.plot-grid {{ display:grid;grid-template-columns:repeat(2,1fr);gap:12px; }}
.plot {{
  width:100%;height:340px;border-radius:8px;
  border:1px solid var(--border);background:white;
}}

.toggle {{
  cursor:pointer;font-size:13px;font-weight:500;color:var(--accent);
  padding:8px 0;list-style:none;user-select:none;
}}
.toggle::-webkit-details-marker {{ display:none; }}
.toggle::before {{ content:'\25B6\FE0E  ';font-size:10px;transition:transform .2s; }}
details[open] .toggle::before {{ content:'\25BC\FE0E  '; }}

.muted {{ color:var(--text3);font-size:12px;margin-bottom:8px; }}

.footer {{
  text-align:center;padding:32px 0 8px;color:var(--text3);font-size:11px;
  border-top:1px solid var(--border);margin-top:24px;
}}
.footer strong {{ color:var(--text2);font-weight:600; }}

@media (max-width:768px) {{
  .plot-grid {{ grid-template-columns:1fr; }}
  .stat-grid {{ grid-template-columns:1fr 1fr; }}
  .container {{ padding:12px 12px 24px; }}
  .nav {{ display:none; }}
}}
@media print {{
  .header {{ background:#1e293b !important;-webkit-print-color-adjust:exact;print-color-adjust:exact; }}
  .card {{ break-inside:avoid;box-shadow:none; }}
  .plot {{ height:280px; }}
  .nav {{ display:none; }}
}}
</style></head>
<body>
<div class="header">
  <h1>GAM Report</h1>
  <p class="subtitle">{model_path}</p>
  <nav class="nav">{nav}</nav>
</div>
<div class="container">
{notes}

<section class="card" id="sec-summary">
<h2>Model Summary</h2>
{formula}
<div class="stat-grid">
{summary_items}
</div>
</section>

<section class="card" id="sec-coef">
<h2>Coefficients</h2>
{coef_body}
</section>

{edf_section}

{continuous_section}
{aniso_section}
{measure_jet_section}
{smoothing_forensics_section}
{diagnostics_section}
{smooth_section}
{alo_section}

<div class="footer">
  Generated by <strong>gam</strong>
</div>
</div>
<script>
{scripts}
</script>
</body></html>"##,
        model_path = esc(&input.model_path),
        nav = nav_links,
        notes = notes_html,
        formula = formula_html,
        summary_items = summary_items,
        coef_body = coef_body,
        edf_section = edf_section,
        continuous_section = continuous_section,
        aniso_section = aniso_section,
        measure_jet_section = measure_jet_section,
        smoothing_forensics_section = smoothing_forensics_section,
        diagnostics_section = diagnostics_section,
        smooth_section = smooth_section,
        alo_section = alo_section,
        scripts = scripts.join("\n"),
    ))
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

fn js_escape(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('\'', "\\'")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

fn to_html_id(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn fmt_num(v: f64) -> String {
    if v.abs() < 1e4 && v.abs() > 1e-2 {
        format!("{:.4}", v)
    } else {
        format!("{:.6e}", v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── esc ──────────────────────────────────────────────────────────────────

    #[test]
    fn esc_passthrough_safe_text() {
        assert_eq!(esc("hello world"), "hello world");
    }

    #[test]
    fn esc_empty_string() {
        assert_eq!(esc(""), "");
    }

    #[test]
    fn esc_ampersand() {
        assert_eq!(esc("a&b"), "a&amp;b");
    }

    #[test]
    fn esc_angle_brackets() {
        assert_eq!(esc("<script>"), "&lt;script&gt;");
    }

    #[test]
    fn esc_double_quote() {
        assert_eq!(esc("\"x\""), "&quot;x&quot;");
    }

    #[test]
    fn esc_single_quote() {
        assert_eq!(esc("it's"), "it&#39;s");
    }

    #[test]
    fn esc_all_entities() {
        assert_eq!(esc("a&b<c>d\"e'f"), "a&amp;b&lt;c&gt;d&quot;e&#39;f");
    }

    // ── js_escape ─────────────────────────────────────────────────────────────

    #[test]
    fn js_escape_passthrough_safe_text() {
        assert_eq!(js_escape("hello"), "hello");
    }

    #[test]
    fn js_escape_backslash() {
        assert_eq!(js_escape("a\\b"), "a\\\\b");
    }

    #[test]
    fn js_escape_single_quote() {
        assert_eq!(js_escape("it's"), "it\\'s");
    }

    #[test]
    fn js_escape_double_quote() {
        assert_eq!(js_escape("say \"hi\""), "say \\\"hi\\\"");
    }

    #[test]
    fn js_escape_newline() {
        assert_eq!(js_escape("line\nnext"), "line\\nnext");
    }

    // ── to_html_id ───────────────────────────────────────────────────────────

    #[test]
    fn to_html_id_passthrough_alphanumeric() {
        assert_eq!(to_html_id("hello123"), "hello123");
    }

    #[test]
    fn to_html_id_hyphen_and_underscore_preserved() {
        assert_eq!(to_html_id("x-var_1"), "x-var_1");
    }

    #[test]
    fn to_html_id_dot_becomes_underscore() {
        assert_eq!(to_html_id("x.1"), "x_1");
    }

    #[test]
    fn to_html_id_parens_become_underscore() {
        assert_eq!(to_html_id("s(x)"), "s_x_");
    }

    #[test]
    fn to_html_id_empty_string() {
        assert_eq!(to_html_id(""), "");
    }

    // ── fmt_num ──────────────────────────────────────────────────────────────

    #[test]
    fn fmt_num_normal_range_positive() {
        assert_eq!(fmt_num(1.5), "1.5000");
    }

    #[test]
    fn fmt_num_normal_range_negative() {
        assert_eq!(fmt_num(-3.14), "-3.1400");
    }

    #[test]
    fn fmt_num_just_below_1e4() {
        assert_eq!(fmt_num(9999.0), "9999.0000");
    }

    #[test]
    fn fmt_num_at_1e4_uses_scientific() {
        // Rust `{:.6e}` prints a bare exponent (no zero padding, no plus).
        assert_eq!(fmt_num(10000.0), "1.000000e4");
    }

    #[test]
    fn fmt_num_just_above_0_01_threshold() {
        assert_eq!(fmt_num(0.011), "0.0110");
    }

    #[test]
    fn fmt_num_exactly_0_01_uses_scientific() {
        assert_eq!(fmt_num(0.01), "1.000000e-2");
    }

    #[test]
    fn fmt_num_zero_uses_scientific() {
        assert_eq!(fmt_num(0.0), "0.000000e0");
    }

    // ── render_html smoke test ────────────────────────────────────────────────

    fn minimal_input(formula: &str) -> ReportInput {
        ReportInput {
            model_path: "model.gam".to_string(),
            family_name: "Gaussian".to_string(),
            model_class: "GAM".to_string(),
            formula: formula.to_string(),
            n_obs: Some(100),
            deviance: 42.5,
            reml_score: -17.3,
            iterations: 5,
            convergence_status: "Converged".to_string(),
            converged: true,
            outer_gradient_norm: None,
            criterion_certificate: None,
            smoothing_forensics: vec![],
            edf_total: 3.2,
            r_squared: Some(0.85),
            coefficients: vec![CoefficientRow {
                index: 0,
                estimate: 1.23,
                std_error: Some(0.05),
            }],
            edf_blocks: vec![EdfBlockRow {
                index: 0,
                edf: 3.2,
                role: None,
            }],
            continuous_order: vec![],
            anisotropic_scales: vec![],
            measure_jet_spectra: vec![],
            diagnostics: None,
            smooth_plots: vec![],
            alo: None,
            notes: vec![],
        }
    }

    #[test]
    fn render_html_produces_doctype() {
        let html = render_html(&minimal_input("y ~ s(x)")).unwrap();
        assert!(
            html.starts_with("<!doctype html>"),
            "expected HTML doctype at start"
        );
    }

    #[test]
    fn render_html_contains_formula() {
        let html = render_html(&minimal_input("y ~ s(x)")).unwrap();
        assert!(
            html.contains("y ~ s(x)"),
            "formula not found in rendered HTML"
        );
    }

    #[test]
    fn render_html_names_analytic_gradient_certificate_as_gradient() {
        let mut input = minimal_input("y ~ s(x)");
        input.criterion_certificate = Some(CriterionCertificateRow {
            stationarity: CriterionStationarityRow::AnalyticGradient {
                grad_norm: 2.0e-9,
                projected_grad_norm: 1.0e-9,
                bound: 1.0e-8,
            },
            hessian_psd: Some(true),
            lambdas_railed: Vec::new(),
            stationary: true,
            clean: true,
        });
        let html = render_html(&input).unwrap();
        assert!(html.contains("|g|=2.000e-9"));
        assert!(!html.contains("fixed-point"));
    }

    #[test]
    fn render_html_names_fixed_point_certificate_as_residual() {
        let mut input = minimal_input("y ~ s(x)");
        input.criterion_certificate = Some(CriterionCertificateRow {
            stationarity: CriterionStationarityRow::FixedPoint {
                residual_inf_norm: 2.0e-9,
                projected_residual_inf_norm: 1.0e-9,
                bound: 1.0e-8,
                covered_coordinates: 7,
            },
            hessian_psd: None,
            lambdas_railed: Vec::new(),
            stationary: true,
            clean: true,
        });
        let html = render_html(&input).unwrap();
        assert!(html.contains("fixed-point"));
        assert!(html.contains("coordinates=7"));
        assert!(!html.contains("|g|="));
    }

    #[test]
    fn render_html_escapes_formula_special_chars() {
        let html = render_html(&minimal_input("y ~ <bad>")).unwrap();
        assert!(
            html.contains("&lt;bad&gt;"),
            "HTML special chars in formula must be escaped"
        );
        assert!(
            !html.contains("<bad>"),
            "raw unescaped <bad> must not appear"
        );
    }

    #[test]
    fn render_html_notes_are_escaped() {
        let mut input = minimal_input("y ~ s(x)");
        input.notes = vec!["<script>alert('xss')</script>".to_string()];
        let html = render_html(&input).unwrap();
        assert!(
            !html.contains("<script>alert"),
            "raw <script> tag must not appear in output"
        );
        assert!(
            html.contains("&lt;script&gt;"),
            "script tag must be HTML-escaped"
        );
    }

    #[test]
    fn render_html_preserves_multicoordinate_alo_rows() {
        let mut input = minimal_input("y ~ x");
        input.alo = Some(AloData {
            coordinate_names: vec!["mean".to_string(), "<log-scale>".to_string()],
            rows: vec![AloRow {
                index: 4,
                leverage: 0.25,
                eta_tilde: vec![1.5, -0.75],
                standard_errors: vec![0.1, 0.2],
                cook_distance: 0.03125,
            }],
        });
        let html = render_html(&input).expect("render multicoordinate ALO");
        assert!(html.contains("Deleted-row coordinates"));
        assert!(html.contains("mean=1.500000e0"));
        assert!(html.contains("&lt;log-scale&gt;=-7.500000e-1"));
        assert!(html.contains("Cook distance"));
        assert!(!html.contains("<log-scale>"));
    }

    #[test]
    fn render_html_rejects_misaligned_alo_coordinate_vectors() {
        let mut input = minimal_input("y ~ x");
        input.alo = Some(AloData {
            coordinate_names: vec!["mean".to_string(), "log-scale".to_string()],
            rows: vec![AloRow {
                index: 0,
                leverage: 0.1,
                eta_tilde: vec![1.0],
                standard_errors: vec![0.2, 0.3],
                cook_distance: 0.01,
            }],
        });
        let error = render_html(&input).expect_err("misaligned ALO rows must be rejected");
        assert!(error.contains("coordinate lengths"));
    }

    #[test]
    fn render_html_includes_smoothing_forensics_when_present() {
        let mut input = minimal_input("y ~ s(x)");
        input.smoothing_forensics = vec![SmoothingForensicsRow {
            term: "s(x)".to_string(),
            lambda_path: vec![0.1, 0.2],
            sigma2_path: vec![1.5],
            edf_criterion: Some(3.0),
            edf_assembly: Some(3.1),
            double_penalty_range: Some(2.0),
            double_penalty_null_space: Some(1.0),
            seed_screening: vec!["accepted".to_string()],
        }];
        let html = render_html(&input).unwrap();
        assert!(html.contains("Smoothing Forensics"));
        assert!(html.contains("0.1000 → 0.2000"));
        assert!(html.contains("accepted"));
    }

    #[test]
    fn render_html_non_converged_shows_warning() {
        let mut input = minimal_input("y ~ s(x)");
        input.converged = false;
        input.convergence_status = "Max iterations reached".to_string();
        let html = render_html(&input).unwrap();
        assert!(
            html.contains("conv-warn"),
            "non-converged fit must show conv-warn class"
        );
    }
}
