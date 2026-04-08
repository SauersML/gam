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
    pub edf_total: f64,
    pub r_squared: Option<f64>,
    pub coefficients: Vec<CoefficientRow>,
    pub edf_blocks: Vec<EdfBlockRow>,
    pub continuous_order: Vec<ContinuousOrderRow>,
    pub anisotropic_scales: Vec<AnisotropicScalesRow>,
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

pub struct DiagnosticsInput {
    pub residuals_sorted: Vec<f64>,
    pub theoretical_quantiles: Vec<f64>,
    pub y_observed: Vec<f64>,
    pub y_predicted: Vec<f64>,
    pub calibration: Option<CalibrationData>,
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
        let residuals: Vec<f64> = diag
            .y_observed
            .iter()
            .zip(diag.y_predicted.iter())
            .map(|(o, p)| o - p)
            .collect();

        // QQ plot
        scripts.push(format!(
            "Plotly.newPlot('qq_plot',\
             [{{x:{theo},y:{res},mode:'markers',type:'scattergl',{marker}}}],\
             {layout},{{{cfg}}});",
            theo = json(&diag.theoretical_quantiles)?,
            res = json(&diag.residuals_sorted)?,
            marker = marker,
            layout = plot_style("Normal Q-Q", "Theoretical Quantile", "Sample Quantile"),
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
            resid = json(&residuals)?,
            marker = marker,
            layout = plot_style("Residuals vs Fitted", "Fitted Value", "Residual"),
            fit_min = fit_min,
            fit_max = fit_max,
            cfg = plot_cfg,
        ));

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

        // Residual histogram
        scripts.push(format!(
            "Plotly.newPlot('resid_hist',\
             [{{x:{resid},type:'histogram',\
             marker:{{color:'#6366f1',line:{{color:'#4f46e5',width:0.5}}}},opacity:0.85}}],\
             {layout},{{{cfg}}});",
            resid = json(&residuals)?,
            layout = plot_style("Residual Distribution", "Residual", "Frequency"),
            cfg = plot_cfg,
        ));

        // Scale-location plot: sqrt(|standardized residuals|) vs fitted
        let resid_std = {
            let n = residuals.len() as f64;
            let mean = residuals.iter().sum::<f64>() / n.max(1.0);
            let var =
                residuals.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
            let sd = var.sqrt().max(1e-15);
            residuals
                .iter()
                .map(|r| (r / sd).abs().sqrt())
                .collect::<Vec<_>>()
        };
        scripts.push(format!(
            "Plotly.newPlot('scale_loc',\
             [{{x:{fitted},y:{sqrt_abs},mode:'markers',type:'scattergl',{marker}}}],\
             {layout},{{{cfg}}});",
            fitted = json(&diag.y_predicted)?,
            sqrt_abs = json(&resid_std)?,
            marker = marker,
            layout = plot_style(
                "Scale-Location",
                "Fitted Value",
                "&radic;|Standardized Residual|"
            ),
            cfg = plot_cfg,
        ));

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
    summary_pairs.push(("Outer Iterations", format!("{}", input.iterations)));

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

    // Diagnostics plots grid
    let diagnostics_section = if input.diagnostics.is_some() {
        let has_cal = input
            .diagnostics
            .as_ref()
            .and_then(|d| d.calibration.as_ref())
            .is_some();
        let cal_div = if has_cal {
            "<div id=\"cal_plot\" class=\"plot\"></div>"
        } else {
            ""
        };
        format!(
            "<section class=\"card\" id=\"sec-diagnostics\">\n\
             <h2>Diagnostics</h2>\n\
             <div class=\"plot-grid\">\n\
               <div id=\"qq_plot\" class=\"plot\"></div>\n\
               <div id=\"resid_fitted\" class=\"plot\"></div>\n\
               <div id=\"obs_pred\" class=\"plot\"></div>\n\
               <div id=\"resid_hist\" class=\"plot\"></div>\n\
               <div id=\"scale_loc\" class=\"plot\"></div>\n\
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
        let max_show = 100;
        let n_show = alo.rows.len().min(max_show);
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
             <thead><tr><th>Row</th><th>Leverage</th><th>\u{03B7}\u{0303}</th><th>SE (sandwich)</th></tr></thead>\n\
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
