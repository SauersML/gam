use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use ndarray::{Array1, ArrayView1, s};

use gam::alo::compute_alo_diagnostics_from_fit;
use gam::estimate::{ContinuousSmoothnessOrderStatus, FitResult, ModelSummary, SmoothTermSummary};
use gam::families::family_meta::{family_to_link, pretty_familyname};
use gam::inference::model::{FittedModel as SavedModel, PredictModelClass};
use gam::probability::standard_normal_quantile;
use gam::smooth::{SmoothBasisSpec, SmoothTermSpec, TermCollectionSpec};
use gam::types::LikelihoodFamily;

// Public data types — main.rs fills these, report.rs renders them

pub struct ReportInput {
    pub model_path: String,
    pub family_name: String,
    pub model_class: String,
    pub formula: String,
    pub n_obs: Option<usize>,
    pub deviance: f64,
    pub reml_score: f64,
    pub iterations: usize,
    pub edf_total: f64,
    pub coefficients: Vec<CoefficientRow>,
    pub edf_blocks: Vec<(usize, f64)>,
    pub continuous_order: Vec<ContinuousOrderRow>,
    pub diagnostics: Option<DiagnosticsInput>,
    pub smooth_plots: Vec<SmoothPlotData>,
    pub notes: Vec<String>,
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

pub struct DiagnosticsInput {
    pub residuals_sorted: Vec<f64>,
    pub theoretical_quantiles: Vec<f64>,
    pub y_observed: Vec<f64>,
    pub y_predicted: Vec<f64>,
    pub calibration: Option<CalibrationData>,
    pub alo: Option<AloData>,
}

pub struct CalibrationData {
    pub mean_predicted: Vec<f64>,
    pub observed_rate: Vec<f64>,
}

pub struct AloData {
    pub rows: Vec<AloRow>,
}

pub struct AloRow {
    pub index: usize,
    pub leverage: f64,
    pub eta_tilde: f64,
    pub se_sandwich: f64,
}

pub struct SmoothPlotData {
    pub name: String,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
}

// Builder — collects data from main.rs helpers, produces ReportInput

pub fn build_report_input(
    model: &SavedModel,
    fit: &FitResult,
    model_path: &Path,
    data: Option<ReportDataContext<'_>>,
) -> Result<ReportInput, String> {
    let family = model.likelihood();
    let family_name = pretty_familyname(family).to_string();
    let model_class = format!("{:?}", model.predict_model_class());
    let formula = model.formula.clone();

    let beta_se = fit
        .beta_standard_errors_corrected()
        .or(fit.beta_standard_errors());

    let coefficients: Vec<CoefficientRow> = fit
        .beta
        .iter()
        .copied()
        .enumerate()
        .map(|(i, b)| CoefficientRow {
            index: i,
            estimate: b,
            std_error: beta_se.and_then(|s| s.get(i).copied()),
        })
        .collect();

    let edf_blocks: Vec<(usize, f64)> = fit.edf_by_block().iter().copied().enumerate().collect();

    let mut notes = Vec::new();
    let mut continuous_order = Vec::new();
    let mut diagnostics = None;
    let mut smooth_plots = Vec::new();
    let mut n_obs = None;

    if let Some(ctx) = data {
        n_obs = Some(ctx.n_obs);

        // Build continuous smoothness order rows from the model summary
        for st in &ctx.summary.smooth_terms {
            if let Some(ord) = st.continuous_order.as_ref() {
                let status = match ord.status {
                    ContinuousSmoothnessOrderStatus::Ok => "Ok",
                    ContinuousSmoothnessOrderStatus::NonMaternRegime => "Non-Matern",
                    ContinuousSmoothnessOrderStatus::FirstOrderLimit => "1st-Order Limit",
                    ContinuousSmoothnessOrderStatus::IntrinsicLimit => "Intrinsic Limit",
                    ContinuousSmoothnessOrderStatus::UndefinedZeroLambda => "Undef (zero lambda)",
                };
                let finite_or_none = |v: Option<f64>| v.filter(|x| x.is_finite());
                continuous_order.push(ContinuousOrderRow {
                    name: st.name.clone(),
                    lambda0: ord.lambda0,
                    lambda1: ord.lambda1,
                    lambda2: ord.lambda2,
                    r_ratio: finite_or_none(ord.r_ratio),
                    nu: finite_or_none(ord.nu),
                    kappa2: finite_or_none(ord.kappa2),
                    status: status.to_string(),
                });
            }
        }

        // Residual QQ data
        let residuals = &ctx.y - &ctx.predicted;
        let mut residuals_sorted = residuals.to_vec();
        residuals_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = residuals_sorted.len().max(1);
        let theoretical_quantiles = (0..n)
            .map(|i| standard_normal_quantile((i as f64 + 0.5) / n as f64))
            .collect::<Result<Vec<_>, _>>()?;

        // Calibration for binary
        let calibration = if is_binary_response(ctx.y.view()) {
            Some(build_calibration(ctx.y.view(), ctx.predicted.view()))
        } else {
            None
        };

        // ALO diagnostics
        let alo = match model
            .resolved_inverse_link()
            .ok()
            .and_then(|r| r.map(|lk| lk.link_function()))
        {
            Some(link) => match compute_alo_diagnostics_from_fit(fit, ctx.y.view(), link) {
                Ok(alo) => Some(AloData {
                    rows: (0..alo.leverage.len())
                        .map(|i| AloRow {
                            index: i,
                            leverage: alo.leverage[i],
                            eta_tilde: alo.eta_tilde[i],
                            se_sandwich: alo.se_sandwich[i],
                        })
                        .collect(),
                }),
                Err(e) => {
                    notes.push(format!("ALO diagnostics unavailable: {e}"));
                    None
                }
            },
            None => {
                notes.push("ALO diagnostics unavailable: missing inverse link".to_string());
                None
            }
        };

        diagnostics = Some(DiagnosticsInput {
            residuals_sorted,
            theoretical_quantiles,
            y_observed: ctx.y.to_vec(),
            y_predicted: ctx.predicted.to_vec(),
            calibration,
            alo,
        });

        // Smooth term partial-effect plots
        for st in &ctx.spec.smooth_terms {
            if let Some(col) = smooth_term_primary_column(st) {
                if col < ctx.data_ncols {
                    if let Some(design_term) =
                        ctx.design_smooth_terms.iter().find(|t| t.0 == st.name)
                    {
                        let x_col = ctx.data_column(col);
                        let contrib = ctx
                            .design_slice(design_term.1.clone())
                            .dot(&fit.beta.slice(s![design_term.1.clone()]));
                        let mut pairs: Vec<(f64, f64)> =
                            x_col.iter().copied().zip(contrib.iter().copied()).collect();
                        pairs.sort_by(|a, b| {
                            a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
                        });
                        smooth_plots.push(SmoothPlotData {
                            name: st.name.clone(),
                            x: pairs.iter().map(|p| p.0).collect(),
                            y: pairs.iter().map(|p| p.1).collect(),
                        });
                    }
                }
            }
        }
    } else {
        notes.push(
            "No data provided — residual, calibration, and ALO diagnostics are omitted. \
             Pass training data as the second positional argument."
                .to_string(),
        );
    }

    Ok(ReportInput {
        model_path: model_path.display().to_string(),
        family_name,
        model_class,
        formula,
        n_obs,
        deviance,
        reml_score: fit.reml_score,
        iterations: fit.iterations,
        edf_total: fit.edf_total().unwrap_or(0.0),
        coefficients,
        edf_blocks,
        continuous_order,
        diagnostics,
        smooth_plots,
        notes,
    })
}

/// Context passed from main.rs when data is available for diagnostics.
pub struct ReportDataContext<'a> {
    pub n_obs: usize,
    pub y: Array1<f64>,
    pub predicted: Array1<f64>,
    pub summary: ModelSummary,
    pub spec: &'a TermCollectionSpec,
    pub data_ncols: usize,
    /// (name, coeff_range) for each smooth design term
    pub design_smooth_terms: Vec<(String, std::ops::Range<usize>)>,
    /// Full design matrix view
    design_matrix: ndarray::Array2<f64>,
    /// Raw data matrix view
    data_matrix: ndarray::Array2<f64>,
}

impl<'a> ReportDataContext<'a> {
    pub fn new(
        y: Array1<f64>,
        predicted: Array1<f64>,
        summary: ModelSummary,
        spec: &'a TermCollectionSpec,
        design_matrix: ndarray::Array2<f64>,
        data_matrix: ndarray::Array2<f64>,
        design_smooth_terms: Vec<(String, std::ops::Range<usize>)>,
    ) -> Self {
        let n_obs = y.len();
        let data_ncols = data_matrix.ncols();
        Self {
            n_obs,
            y,
            predicted,
            summary,
            spec,
            data_ncols,
            design_smooth_terms,
            design_matrix,
            data_matrix,
        }
    }

    fn data_column(&self, col: usize) -> Array1<f64> {
        self.data_matrix.column(col).to_owned()
    }

    fn design_slice(&self, range: std::ops::Range<usize>) -> ndarray::ArrayView2<'_, f64> {
        self.design_matrix.slice(s![.., range])
    }
}

// HTML rendering

pub fn render_html(input: &ReportInput) -> Result<String, String> {
    let mut scripts = Vec::new();

    // --- Diagnostics plots ---
    if let Some(diag) = &input.diagnostics {
        // QQ plot
        scripts.push(format!(
            "Plotly.newPlot('qq_plot',\
             [{{x:{theo},y:{res},mode:'markers',type:'scatter',\
             marker:{{color:'#6366f1',size:5,opacity:0.7}}}}],\
             {{margin:{{t:40,b:50,l:60,r:20}},\
             title:{{text:'Residual QQ Plot',font:{{size:14}}}},\
             xaxis:{{title:'Theoretical Quantile',gridcolor:'#f0f0f0'}},\
             yaxis:{{title:'Sample Quantile',gridcolor:'#f0f0f0'}},\
             plot_bgcolor:'#fafafa',paper_bgcolor:'white'}},\
             {{responsive:true}});",
            theo = serde_json::to_string(&diag.theoretical_quantiles).map_err(|e| e.to_string())?,
            res = serde_json::to_string(&diag.residuals_sorted).map_err(|e| e.to_string())?,
        ));

        // Residuals vs fitted
        scripts.push(format!(
            "Plotly.newPlot('resid_fitted',\
             [{{x:{fitted},y:{resid},mode:'markers',type:'scatter',\
             marker:{{color:'#6366f1',size:5,opacity:0.7}}}}],\
             {{margin:{{t:40,b:50,l:60,r:20}},\
             title:{{text:'Residuals vs Fitted',font:{{size:14}}}},\
             xaxis:{{title:'Fitted Value',gridcolor:'#f0f0f0'}},\
             yaxis:{{title:'Residual',gridcolor:'#f0f0f0'}},\
             plot_bgcolor:'#fafafa',paper_bgcolor:'white',\
             shapes:[{{type:'line',x0:{fit_min},x1:{fit_max},y0:0,y1:0,\
             line:{{color:'#94a3b8',width:1,dash:'dash'}}}}]}},\
             {{responsive:true}});",
            fitted = serde_json::to_string(&diag.y_predicted).map_err(|e| e.to_string())?,
            resid = serde_json::to_string(
                &diag
                    .y_observed
                    .iter()
                    .zip(diag.y_predicted.iter())
                    .map(|(o, p)| o - p)
                    .collect::<Vec<_>>()
            )
            .map_err(|e| e.to_string())?,
            fit_min = diag
                .y_predicted
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min),
            fit_max = diag
                .y_predicted
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max),
        ));

        // Observed vs Predicted
        let obs_pred_min = diag
            .y_observed
            .iter()
            .chain(diag.y_predicted.iter())
            .copied()
            .fold(f64::INFINITY, f64::min);
        let obs_pred_max = diag
            .y_observed
            .iter()
            .chain(diag.y_predicted.iter())
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        scripts.push(format!(
            "Plotly.newPlot('obs_pred',\
             [{{x:{pred},y:{obs},mode:'markers',type:'scatter',name:'Data',\
             marker:{{color:'#6366f1',size:5,opacity:0.7}}}},\
             {{x:[{lo},{hi}],y:[{lo},{hi}],mode:'lines',name:'y=x',\
             line:{{color:'#94a3b8',width:1,dash:'dash'}}}}],\
             {{margin:{{t:40,b:50,l:60,r:20}},\
             title:{{text:'Observed vs Predicted',font:{{size:14}}}},\
             xaxis:{{title:'Predicted',gridcolor:'#f0f0f0'}},\
             yaxis:{{title:'Observed',gridcolor:'#f0f0f0'}},\
             showlegend:false,\
             plot_bgcolor:'#fafafa',paper_bgcolor:'white'}},\
             {{responsive:true}});",
            pred = serde_json::to_string(&diag.y_predicted).map_err(|e| e.to_string())?,
            obs = serde_json::to_string(&diag.y_observed).map_err(|e| e.to_string())?,
            lo = obs_pred_min,
            hi = obs_pred_max,
        ));

        // Histogram of residuals
        let residuals_unsorted: Vec<f64> = diag
            .y_observed
            .iter()
            .zip(diag.y_predicted.iter())
            .map(|(o, p)| o - p)
            .collect();
        scripts.push(format!(
            "Plotly.newPlot('resid_hist',\
             [{{x:{resid},type:'histogram',\
             marker:{{color:'#6366f1',line:{{color:'#4f46e5',width:1}}}},\
             opacity:0.8}}],\
             {{margin:{{t:40,b:50,l:60,r:20}},\
             title:{{text:'Residual Distribution',font:{{size:14}}}},\
             xaxis:{{title:'Residual',gridcolor:'#f0f0f0'}},\
             yaxis:{{title:'Count',gridcolor:'#f0f0f0'}},\
             plot_bgcolor:'#fafafa',paper_bgcolor:'white'}},\
             {{responsive:true}});",
            resid = serde_json::to_string(&residuals_unsorted).map_err(|e| e.to_string())?,
        ));

        // Calibration plot (binary only)
        if let Some(cal) = &diag.calibration {
            scripts.push(format!(
                "Plotly.newPlot('cal_plot',\
                 [{{x:{x},y:{y},mode:'markers+lines',type:'scatter',name:'Calibration',\
                 marker:{{color:'#6366f1',size:8}},line:{{color:'#6366f1'}}}},\
                 {{x:[0,1],y:[0,1],mode:'lines',name:'Ideal',\
                 line:{{color:'#94a3b8',width:1,dash:'dash'}}}}],\
                 {{margin:{{t:40,b:50,l:60,r:20}},\
                 title:{{text:'Calibration (Deciles)',font:{{size:14}}}},\
                 xaxis:{{title:'Mean Predicted',gridcolor:'#f0f0f0',range:[0,1]}},\
                 yaxis:{{title:'Observed Rate',gridcolor:'#f0f0f0',range:[0,1]}},\
                 showlegend:false,\
                 plot_bgcolor:'#fafafa',paper_bgcolor:'white'}},\
                 {{responsive:true}});",
                x = serde_json::to_string(&cal.mean_predicted).map_err(|e| e.to_string())?,
                y = serde_json::to_string(&cal.observed_rate).map_err(|e| e.to_string())?,
            ));
        }
    }

    // --- Smooth term plots ---
    for sp in &input.smooth_plots {
        let div_id = format!("smooth_{}", to_html_id(&sp.name));
        scripts.push(format!(
            "Plotly.newPlot('{div_id}',\
             [{{x:{x},y:{y},mode:'lines',type:'scatter',\
             line:{{color:'#6366f1',width:2}}}}],\
             {{margin:{{t:40,b:50,l:60,r:20}},\
             title:{{text:'s({name})',font:{{size:14}}}},\
             xaxis:{{title:'{name}',gridcolor:'#f0f0f0'}},\
             yaxis:{{title:'Partial Effect',gridcolor:'#f0f0f0'}},\
             plot_bgcolor:'#fafafa',paper_bgcolor:'white'}},\
             {{responsive:true}});",
            div_id = div_id,
            x = serde_json::to_string(&sp.x).map_err(|e| e.to_string())?,
            y = serde_json::to_string(&sp.y).map_err(|e| e.to_string())?,
            name = js_escape(&sp.name),
        ));
    }

    // --- Build HTML sections ---
    let notes_html = if input.notes.is_empty() {
        String::new()
    } else {
        format!(
            "<div class=\"alert\">{}</div>",
            input
                .notes
                .iter()
                .map(|n| escape(n))
                .collect::<Vec<_>>()
                .join("<br/>")
        )
    };

    let summary_rows = [
        ("Family", input.family_name.clone()),
        ("Model Class", input.model_class.clone()),
        (
            "Formula",
            format!("<code>{}</code>", escape(&input.formula)),
        ),
        (
            "Observations",
            input
                .n_obs
                .map(|n| format!("{n}"))
                .unwrap_or_else(|| "—".to_string()),
        ),
        ("Deviance", format!("{:.6}", input.deviance)),
        ("REML / LAML", format!("{:.6}", input.reml_score)),
        ("Iterations", format!("{}", input.iterations)),
        ("EDF (total)", format!("{:.4}", input.edf_total)),
    ]
    .iter()
    .map(|(k, v)| format!("<tr><th>{k}</th><td>{v}</td></tr>"))
    .collect::<Vec<_>>()
    .join("\n");

    let coef_rows = input
        .coefficients
        .iter()
        .map(|c| {
            let se_str = c
                .std_error
                .map(|v| format!("{v:.6e}"))
                .unwrap_or_else(|| "—".to_string());
            let z_str = c
                .std_error
                .filter(|&se| se.abs() > 1e-15)
                .map(|se| format!("{:.3}", c.estimate / se))
                .unwrap_or_else(|| "—".to_string());
            format!(
                "<tr><td class=\"mono\">{}</td><td class=\"num\">{:.6e}</td>\
                 <td class=\"num\">{}</td><td class=\"num\">{}</td></tr>",
                c.index, c.estimate, se_str, z_str
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    let edf_rows = input
        .edf_blocks
        .iter()
        .map(|(i, edf)| {
            format!("<tr><td class=\"mono\">{i}</td><td class=\"num\">{edf:.4}</td></tr>")
        })
        .collect::<Vec<_>>()
        .join("\n");

    let continuous_section = if input.continuous_order.is_empty() {
        String::new()
    } else {
        let rows = input
            .continuous_order
            .iter()
            .map(|c| {
                let fmt = |v: Option<f64>| {
                    v.map(|x| format!("{x:.4e}"))
                        .unwrap_or_else(|| "—".to_string())
                };
                format!(
                    "<tr><td>{}</td><td class=\"num\">{:.4e}</td>\
                     <td class=\"num\">{:.4e}</td><td class=\"num\">{:.4e}</td>\
                     <td class=\"num\">{}</td><td class=\"num\">{}</td>\
                     <td class=\"num\">{}</td><td>{}</td></tr>",
                    escape(&c.name),
                    c.lambda0,
                    c.lambda1,
                    c.lambda2,
                    fmt(c.r_ratio),
                    fmt(c.nu),
                    fmt(c.kappa2),
                    escape(&c.status),
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            r#"<div class="card">
<h2>Continuous Smoothness Order</h2>
<div class="table-wrap"><table>
<thead><tr><th>Term</th><th>&lambda;<sub>0</sub></th><th>&lambda;<sub>1</sub></th>
<th>&lambda;<sub>2</sub></th><th>R</th><th>&nu;</th><th>&kappa;&sup2;</th><th>Status</th></tr></thead>
<tbody>{rows}</tbody>
</table></div></div>"#,
            rows = rows
        )
    };

    let diagnostics_section = if input.diagnostics.is_some() {
        let cal_div = if input
            .diagnostics
            .as_ref()
            .and_then(|d| d.calibration.as_ref())
            .is_some()
        {
            "<div id=\"cal_plot\" class=\"plot\"></div>"
        } else {
            ""
        };
        format!(
            r#"<div class="card">
<h2>Diagnostics</h2>
<div class="plot-grid">
  <div id="qq_plot" class="plot"></div>
  <div id="resid_fitted" class="plot"></div>
  <div id="obs_pred" class="plot"></div>
  <div id="resid_hist" class="plot"></div>
  {cal_div}
</div></div>"#,
            cal_div = cal_div
        )
    } else {
        String::new()
    };

    let alo_section = if let Some(diag) = &input.diagnostics {
        if let Some(alo) = &diag.alo {
            let n_show = alo.rows.len().min(50);
            let rows = alo.rows[..n_show]
                .iter()
                .map(|r| {
                    format!(
                        "<tr><td class=\"mono\">{}</td><td class=\"num\">{:.6e}</td>\
                         <td class=\"num\">{:.6e}</td><td class=\"num\">{:.6e}</td></tr>",
                        r.index, r.leverage, r.eta_tilde, r.se_sandwich
                    )
                })
                .collect::<Vec<_>>()
                .join("\n");
            let note = if alo.rows.len() > n_show {
                format!(
                    "<p class=\"muted\">Showing first {n_show} of {} rows</p>",
                    alo.rows.len()
                )
            } else {
                String::new()
            };
            format!(
                r#"<div class="card">
<h2>ALO Diagnostics</h2>
{note}
<div class="table-wrap"><table>
<thead><tr><th>Row</th><th>Leverage</th><th>&eta;&#x0303;</th><th>SE (sandwich)</th></tr></thead>
<tbody>{rows}</tbody>
</table></div></div>"#,
                note = note,
                rows = rows
            )
        } else {
            String::new()
        }
    } else {
        String::new()
    };

    let smooth_divs = input
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

    let smooth_section = if input.smooth_plots.is_empty() {
        String::new()
    } else {
        format!(
            r#"<div class="card">
<h2>Smooth Terms</h2>
<div class="plot-grid">{smooth_divs}</div>
</div>"#,
            smooth_divs = smooth_divs
        )
    };

    Ok(format!(
        r##"<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>GAM Report &mdash; {model_path}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root {{
  --bg: #f8fafc;
  --card: #ffffff;
  --border: #e2e8f0;
  --accent: #6366f1;
  --accent-light: #eef2ff;
  --text: #1e293b;
  --text-muted: #64748b;
  --header-bg: #1e293b;
  --header-fg: #f8fafc;
  --font: 'Inter', ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
  --mono: 'JetBrains Mono', 'SF Mono', 'Fira Code', 'Cascadia Code', ui-monospace, monospace;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: var(--font); background: var(--bg); color: var(--text); line-height: 1.6; }}
.header {{
  background: var(--header-bg); color: var(--header-fg);
  padding: 24px 32px; margin-bottom: 24px;
}}
.header h1 {{ font-size: 22px; font-weight: 700; letter-spacing: -0.02em; }}
.header p {{ color: #94a3b8; font-size: 13px; margin-top: 4px; }}
.container {{ max-width: 1200px; margin: 0 auto; padding: 0 24px 48px; }}
.card {{
  background: var(--card); border: 1px solid var(--border);
  border-radius: 8px; padding: 24px; margin-bottom: 20px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}}
.card h2 {{
  font-size: 16px; font-weight: 600; color: var(--text);
  margin-bottom: 16px; padding-bottom: 8px; border-bottom: 2px solid var(--accent-light);
}}
.alert {{
  background: #fffbeb; border: 1px solid #fde68a; border-radius: 8px;
  padding: 12px 16px; margin-bottom: 20px; font-size: 13px; color: #92400e;
}}
.table-wrap {{ overflow-x: auto; }}
table {{
  border-collapse: collapse; width: 100%; font-size: 13px;
}}
thead th {{
  background: var(--accent-light); color: var(--accent);
  font-weight: 600; text-align: left;
  padding: 8px 12px; border-bottom: 2px solid var(--border);
  white-space: nowrap;
}}
tbody td {{
  padding: 6px 12px; border-bottom: 1px solid var(--border);
}}
tbody tr:last-child td {{ border-bottom: none; }}
tbody tr:hover {{ background: #f8fafc; }}
td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
td.mono, .mono {{ font-family: var(--mono); font-size: 12px; }}
.summary-grid {{
  display: grid; grid-template-columns: 1fr 1fr; gap: 0;
}}
.summary-grid th {{ width: 140px; white-space: nowrap; background: transparent; color: var(--text-muted); font-weight: 500; border-bottom: 1px solid var(--border); }}
.summary-grid td {{ border-bottom: 1px solid var(--border); }}
.plot-grid {{
  display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px;
}}
.plot {{
  width: 100%; height: 340px; border-radius: 6px;
  border: 1px solid var(--border); background: white;
}}
.muted {{ color: var(--text-muted); font-size: 12px; margin-bottom: 8px; }}
code {{
  font-family: var(--mono); font-size: 12px;
  background: var(--accent-light); color: var(--accent);
  padding: 2px 6px; border-radius: 4px;
}}
@media (max-width: 768px) {{
  .plot-grid {{ grid-template-columns: 1fr; }}
  .summary-grid {{ grid-template-columns: 1fr; }}
  .container {{ padding: 0 12px 24px; }}
}}
</style></head>
<body>
<div class="header">
  <h1>GAM Report</h1>
  <p>{model_path}</p>
</div>
<div class="container">
{notes}

<div class="card">
<h2>Model Summary</h2>
<table class="summary-grid">
{summary_rows}
</table>
</div>

<div class="card">
<h2>Coefficients</h2>
<p class="muted">{n_coef} parameters</p>
<div class="table-wrap"><table>
<thead><tr><th>Index</th><th>Estimate</th><th>Std. Error</th><th>z-value</th></tr></thead>
<tbody>
{coef_rows}
</tbody>
</table></div>
</div>

<div class="card">
<h2>EDF by Penalty Block</h2>
<div class="table-wrap"><table>
<thead><tr><th>Block</th><th>EDF</th></tr></thead>
<tbody>{edf_rows}</tbody>
</table></div>
</div>

{continuous_section}
{diagnostics_section}
{smooth_section}
{alo_section}

<div style="text-align:center;padding:24px 0;color:var(--text-muted);font-size:12px;">
  Generated by <strong>gam</strong>
</div>
</div>
<script>
{scripts}
</script>
</body></html>"##,
        model_path = escape(&input.model_path),
        notes = notes_html,
        summary_rows = summary_rows,
        n_coef = input.coefficients.len(),
        coef_rows = coef_rows,
        edf_rows = edf_rows,
        continuous_section = continuous_section,
        diagnostics_section = diagnostics_section,
        smooth_section = smooth_section,
        alo_section = alo_section,
        scripts = scripts.join("\n"),
    ))
}

/// Write the report HTML to the output path, returning the path used.
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

// Private helpers

fn escape(s: &str) -> String {
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

fn smooth_term_primary_column(term: &SmoothTermSpec) -> Option<usize> {
    match &term.basis {
        SmoothBasisSpec::BSpline1D { feature_col, .. } => Some(*feature_col),
        SmoothBasisSpec::ThinPlate { feature_cols, .. }
        | SmoothBasisSpec::Matern { feature_cols, .. }
        | SmoothBasisSpec::Duchon { feature_cols, .. }
        | SmoothBasisSpec::TensorBSpline { feature_cols, .. } => {
            if feature_cols.len() == 1 {
                Some(feature_cols[0])
            } else {
                None
            }
        }
    }
}

fn is_binary_response(y: ArrayView1<'_, f64>) -> bool {
    if y.is_empty() {
        return false;
    }
    y.iter()
        .all(|v| (*v - 0.0).abs() < 1e-12 || (*v - 1.0).abs() < 1e-12)
}

fn build_calibration(y: ArrayView1<'_, f64>, predicted: ArrayView1<'_, f64>) -> CalibrationData {
    let mut bin_sum_pred = [0.0f64; 10];
    let mut bin_sum_obs = [0.0f64; 10];
    let mut counts = [0usize; 10];
    for i in 0..y.len() {
        let p = predicted[i].clamp(0.0, 1.0);
        let b = ((p * 10.0).floor() as usize).min(9);
        bin_sum_pred[b] += p;
        bin_sum_obs[b] += y[i];
        counts[b] += 1;
    }
    let mut mean_predicted = Vec::new();
    let mut observed_rate = Vec::new();
    for b in 0..10 {
        if counts[b] == 0 {
            continue;
        }
        mean_predicted.push(bin_sum_pred[b] / counts[b] as f64);
        observed_rate.push((bin_sum_obs[b] / counts[b] as f64).clamp(0.0, 1.0));
    }
    CalibrationData {
        mean_predicted,
        observed_rate,
    }
}
