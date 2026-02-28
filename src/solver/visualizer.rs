use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::prelude::*;
use ratatui::widgets::{Axis, Block, Borders, Chart, Dataset, GraphType, Paragraph, Wrap};
use std::io::{self, IsTerminal, Stdout};
use std::sync::Mutex;
use std::time::{Duration, Instant};

static VISUALIZER: Mutex<Option<OptimizationVisualizer>> = Mutex::new(None);

pub struct VisualizerGuard {
    active: bool,
}

impl Drop for VisualizerGuard {
    fn drop(&mut self) {
        if self.active {
            teardown();
        }
    }
}

impl VisualizerGuard {
    pub fn is_active(&self) -> bool {
        self.active
    }
}

pub fn init_guard(enabled: bool) -> VisualizerGuard {
    let active = enabled && init();
    VisualizerGuard { active }
}

pub struct OptimizationVisualizer {
    terminal: Terminal<CrosstermBackend<Stdout>>,
    history_cost_accepted: Vec<(f64, f64)>,
    history_cost_trial: Vec<(f64, f64)>,
    history_grad_log: Vec<(f64, f64)>,
    start_time: Instant,
    current_iter: f64,
    best_cost: f64,
    current_status: String,
    current_stage: String,
    current_detail: String,
    current_eval_state: String,
    current_cost: f64,
    current_grad: f64,
    last_draw: Instant,
    progress_label: String,
    progress_current: usize,
    progress_total: Option<usize>,
    edf_terms: Vec<(String, f64, f64)>,
    diagnostics_lines: Vec<String>,
    diagnostics_condition: Option<f64>,
    diagnostics_step_size: Option<f64>,
    diagnostics_ridge: Option<f64>,
}

impl OptimizationVisualizer {
    fn new() -> io::Result<Self> {
        if !io::stdout().is_terminal() {
            return Err(io::Error::other("stdout is not a terminal"));
        }

        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(stdout);
        let terminal = Terminal::new(backend)?;

        Ok(Self {
            terminal,
            history_cost_accepted: Vec::new(),
            history_cost_trial: Vec::new(),
            history_grad_log: Vec::new(),
            start_time: Instant::now(),
            current_iter: 0.0,
            best_cost: f64::INFINITY,
            current_status: "Initializing...".to_string(),
            current_stage: "init".to_string(),
            current_detail: String::new(),
            current_eval_state: String::new(),
            current_cost: f64::NAN,
            current_grad: f64::NAN,
            last_draw: Instant::now(),
            progress_label: String::new(),
            progress_current: 0,
            progress_total: None,
            edf_terms: Vec::new(),
            diagnostics_lines: Vec::new(),
            diagnostics_condition: None,
            diagnostics_step_size: None,
            diagnostics_ridge: None,
        })
    }

    fn draw(&mut self) -> io::Result<()> {
        let cost_accepted = self.history_cost_accepted.clone();
        let cost_trial = self.history_cost_trial.clone();
        let grad_data = self.history_grad_log.clone();
        let status = self.current_status.clone();
        let stage = self.current_stage.clone();
        let detail = self.current_detail.clone();
        let eval_state = self.current_eval_state.clone();
        let iter = self.current_iter;
        let best = self.best_cost;
        let elapsed = self.start_time.elapsed().as_secs();
        let current_cost = self.current_cost;
        let current_grad = self.current_grad;
        let progress_label = self.progress_label.clone();
        let progress_current = self.progress_current;
        let progress_total = self.progress_total;
        let edf_terms = self.edf_terms.clone();
        let diagnostics_lines = self.diagnostics_lines.clone();
        let diagnostics_condition = self.diagnostics_condition;
        let diagnostics_step_size = self.diagnostics_step_size;
        let diagnostics_ridge = self.diagnostics_ridge;

        self.terminal.draw(|f| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Percentage(72), Constraint::Percentage(28)])
                .split(f.area());
            let top_chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(62), Constraint::Percentage(38)])
                .split(chunks[0]);

            let (min_y, max_y) = if cost_accepted.is_empty() && cost_trial.is_empty() {
                (0.0, 1.0)
            } else {
                let min_val = cost_accepted
                    .iter()
                    .chain(cost_trial.iter())
                    .map(|(_, y)| *y)
                    .fold(f64::INFINITY, f64::min);
                let max_val = cost_accepted
                    .iter()
                    .chain(cost_trial.iter())
                    .map(|(_, y)| *y)
                    .fold(f64::NEG_INFINITY, f64::max);
                (min_val, max_val)
            };
            let window = (max_y - min_y).max(1.0);

            let datasets = vec![
                Dataset::default()
                    .name("Accepted")
                    .marker(symbols::Marker::Braille)
                    .graph_type(GraphType::Line)
                    .style(Style::default().fg(Color::Cyan))
                    .data(&cost_accepted),
                Dataset::default()
                    .name("Trial")
                    .marker(symbols::Marker::Dot)
                    .graph_type(GraphType::Scatter)
                    .style(Style::default().fg(Color::LightBlue))
                    .data(&cost_trial),
            ];

            let chart = Chart::new(datasets)
                .block(
                    Block::default()
                        .title("Objective (LAML/REML)")
                        .borders(Borders::ALL),
                )
                .x_axis(
                    Axis::default()
                        .title("Outer Iteration")
                        .bounds([0.0, iter.max(10.0)])
                        .labels(vec![
                            Line::from("0"),
                            Line::from(format!("{:.0}", iter.max(10.0))),
                        ]),
                )
                .y_axis(
                    Axis::default()
                        .title("Objective Value")
                        .bounds([min_y - window * 0.1, max_y + window * 0.1])
                        .labels(vec![
                            Line::from(format!("{:.2}", min_y)),
                            Line::from(format!("{:.2}", max_y)),
                        ]),
                );
            f.render_widget(chart, top_chunks[0]);

            let mut model_lines = Vec::<String>::new();
            model_lines.push(format!("Stage: {stage}"));
            model_lines.push(format!("Detail: {detail}"));
            model_lines.push(format!("Evaluation: {:.0} {eval_state}", iter));
            model_lines.push(format!("Status: {status}"));
            model_lines.push(format!("Best Objective: {:.6}", best));
            model_lines.push(format!("Current Objective: {:.6}", current_cost));
            model_lines.push(format!("Gradient Norm: {:.3e}", current_grad));
            model_lines.push(format!("Elapsed Time: {elapsed}s"));
            model_lines.push(String::new());
            model_lines.push("Effective Degrees of Freedom by Term".to_string());

            if edf_terms.is_empty() {
                model_lines.push("  (awaiting EDF updates...)".to_string());
            } else {
                let name_w = edf_terms
                    .iter()
                    .map(|(name, _, _)| name.len())
                    .max()
                    .unwrap_or(8)
                    .max(8);
                for (name, edf, ref_df) in edf_terms {
                    let ratio = if ref_df > 0.0 {
                        (edf / ref_df).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    let filled = (ratio * 20.0).round() as usize;
                    let bar = format!(
                        "{}{}",
                        "█".repeat(filled),
                        "·".repeat(20usize.saturating_sub(filled))
                    );
                    model_lines.push(format!(
                        "{:<name_w$} {:>6.2} / {:>6.2}  {}",
                        name,
                        edf,
                        ref_df,
                        bar,
                        name_w = name_w
                    ));
                }
            }

            if let Some(total) = progress_total {
                model_lines.push(String::new());
                model_lines.push(format!(
                    "Progress: {} {}/{}",
                    progress_label, progress_current, total
                ));
            } else if !progress_label.is_empty() {
                model_lines.push(String::new());
                model_lines.push(format!("Progress: {} ({})", progress_label, progress_current));
            }

            let model_panel = Paragraph::new(model_lines.join("\n"))
                .block(
                    Block::default()
                        .title("Model Complexity")
                        .borders(Borders::ALL),
                )
                .style(Style::default().fg(Color::White))
                .wrap(Wrap { trim: true });
            f.render_widget(model_panel, top_chunks[1]);

            let mut diagnostics = Vec::<String>::new();
            diagnostics.push(format!(
                "Condition Number: {}",
                diagnostics_condition
                    .map(|v| format!("{v:.3e}"))
                    .unwrap_or_else(|| "n/a".to_string())
            ));
            diagnostics.push(format!(
                "Step Size (norm delta rho): {}",
                diagnostics_step_size
                    .map(|v| format!("{v:.3e}"))
                    .unwrap_or_else(|| "n/a".to_string())
            ));
            diagnostics.push(format!(
                "Stabilization Ridge: {}",
                diagnostics_ridge
                    .map(|v| format!("{v:.3e}"))
                    .unwrap_or_else(|| "n/a".to_string())
            ));
            diagnostics.push(format!(
                "Convergence Signal (log10|gradient|): {:.3}",
                grad_data.last().map(|(_, y)| *y).unwrap_or(0.0)
            ));
            diagnostics.push(String::new());
            diagnostics.push("Recent Diagnostics and Warnings:".to_string());
            if diagnostics_lines.is_empty() {
                diagnostics.push("  (no diagnostics yet)".to_string());
            } else {
                for line in diagnostics_lines {
                    diagnostics.push(format!("  {line}"));
                }
            }
            diagnostics.push(String::new());
            diagnostics.push("Press Ctrl+C to abort".to_string());

            let diagnostics_panel = Paragraph::new(diagnostics.join("\n"))
                .block(Block::default().title("Diagnostics").borders(Borders::ALL))
                .style(Style::default().fg(Color::White))
                .wrap(Wrap { trim: true });
            f.render_widget(diagnostics_panel, chunks[1]);
        })?;
        Ok(())
    }
}

pub fn init() -> bool {
    let mut guard = VISUALIZER.lock().unwrap();
    if guard.is_some() {
        return true;
    }
    match OptimizationVisualizer::new() {
        Ok(vis) => {
            *guard = Some(vis);
            true
        }
        Err(_) => false,
    }
}

pub fn update(cost: f64, grad_norm: f64, status_msg: &str, iter: f64, eval_state: &str) {
    let mut guard = VISUALIZER.lock().unwrap();
    if let Some(vis) = guard.as_mut() {
        vis.current_iter = iter;
        vis.current_cost = cost;
        vis.current_grad = grad_norm;
        vis.current_eval_state = eval_state.to_string();

        if cost.is_finite() && cost.abs() < 1e15 {
            let target_series = if eval_state == "trial" {
                &mut vis.history_cost_trial
            } else {
                &mut vis.history_cost_accepted
            };
            push_sample(target_series, (iter, cost));
            if cost < vis.best_cost {
                vis.best_cost = cost;
            }
        }

        if grad_norm.is_finite() {
            let grad_log = grad_norm.max(1e-12).log10();
            push_sample(&mut vis.history_grad_log, (iter, grad_log));
        }

        vis.current_status = status_msg.to_string();

        if vis.last_draw.elapsed() >= Duration::from_millis(40) {
            let _ = vis.draw();
            vis.last_draw = Instant::now();
        }
    }
}

pub fn set_stage(stage: &str, detail: &str) {
    let mut guard = VISUALIZER.lock().unwrap();
    if let Some(vis) = guard.as_mut() {
        vis.current_stage = stage.to_string();
        vis.current_detail = detail.to_string();
        if vis.last_draw.elapsed() >= Duration::from_millis(40) {
            let _ = vis.draw();
            vis.last_draw = Instant::now();
        }
    }
}

pub fn set_progress(label: &str, current: usize, total: Option<usize>) {
    let mut guard = VISUALIZER.lock().unwrap();
    if let Some(vis) = guard.as_mut() {
        vis.progress_label = label.to_string();
        vis.progress_current = current;
        vis.progress_total = total;
        if vis.last_draw.elapsed() >= Duration::from_millis(40) {
            let _ = vis.draw();
            vis.last_draw = Instant::now();
        }
    }
}

pub fn set_edf_terms(terms: &[(String, f64, f64)]) {
    let mut guard = VISUALIZER.lock().unwrap();
    if let Some(vis) = guard.as_mut() {
        vis.edf_terms = terms.to_vec();
        if vis.last_draw.elapsed() >= Duration::from_millis(40) {
            let _ = vis.draw();
            vis.last_draw = Instant::now();
        }
    }
}

pub fn set_diagnostics(condition_number: Option<f64>, step_size: Option<f64>, ridge: Option<f64>) {
    let mut guard = VISUALIZER.lock().unwrap();
    if let Some(vis) = guard.as_mut() {
        vis.diagnostics_condition = condition_number;
        vis.diagnostics_step_size = step_size;
        vis.diagnostics_ridge = ridge;
        if vis.last_draw.elapsed() >= Duration::from_millis(40) {
            let _ = vis.draw();
            vis.last_draw = Instant::now();
        }
    }
}

pub fn push_diagnostic(message: &str) {
    let mut guard = VISUALIZER.lock().unwrap();
    if let Some(vis) = guard.as_mut() {
        vis.diagnostics_lines.push(message.to_string());
        const MAX_LINES: usize = 10;
        if vis.diagnostics_lines.len() > MAX_LINES {
            let overflow = vis.diagnostics_lines.len() - MAX_LINES;
            vis.diagnostics_lines.drain(0..overflow);
        }
        if vis.last_draw.elapsed() >= Duration::from_millis(40) {
            let _ = vis.draw();
            vis.last_draw = Instant::now();
        }
    }
}

pub fn teardown() {
    let mut guard = VISUALIZER.lock().unwrap();
    if guard.take().is_some() {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen);
    }
}

fn push_sample(series: &mut Vec<(f64, f64)>, sample: (f64, f64)) {
    const MAX_POINTS: usize = 1200;
    series.push(sample);
    if series.len() > MAX_POINTS {
        let mut compacted = Vec::with_capacity(series.len() / 2 + 1);
        for (idx, point) in series.iter().enumerate() {
            if idx % 2 == 0 {
                compacted.push(*point);
            }
        }
        *series = compacted;
    }
}
