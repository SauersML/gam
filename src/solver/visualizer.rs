use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use log::{Level, LevelFilter, Log, Metadata, Record};
use ratatui::prelude::*;
use ratatui::text::{Line as TextLine, Span};
use ratatui::widgets::{Axis, Block, Borders, Chart, Dataset, GraphType, Paragraph, Wrap};
use std::io::{self, IsTerminal, Stdout, Write};
use std::env;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

const INTERACTIVE_DRAW_INTERVAL: Duration = Duration::from_millis(40);
const DUMB_DRAW_INTERVAL: Duration = Duration::from_secs(1);
const MAX_HISTORY_POINTS: usize = 1200;
const MAX_DIAGNOSTIC_LINES: usize = 10;

static LOGGER: ProgressLogger = ProgressLogger;
static ACTIVE_MULTIPROGRESS: OnceLock<Mutex<Option<MultiProgress>>> = OnceLock::new();

fn active_multiprogress() -> &'static Mutex<Option<MultiProgress>> {
    ACTIVE_MULTIPROGRESS.get_or_init(|| Mutex::new(None))
}

struct ProgressLogger;

impl Log for ProgressLogger {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        metadata.level() <= log::max_level()
    }

    fn log(&self, record: &Record<'_>) {
        if !self.enabled(record.metadata()) {
            return;
        }
        let line = format_log_record(record);
        if let Ok(guard) = active_multiprogress().lock()
            && let Some(mp) = guard.as_ref()
        {
            let _ = mp.println(line);
            return;
        }
        let _ = writeln!(io::stderr(), "{line}");
    }

    fn flush(&self) {}
}

fn format_log_record(record: &Record<'_>) -> String {
    let tag = match record.level() {
        Level::Error => "ERROR",
        Level::Warn => "WARN",
        Level::Info => "INFO",
        Level::Debug => "DEBUG",
        Level::Trace => "TRACE",
    };
    format!("[{tag}] {}", record.args())
}

pub fn init_logging() {
    if log::set_logger(&LOGGER).is_ok() {
        log::set_max_level(LevelFilter::Info);
    }
}

fn install_multiprogress(mp: Option<MultiProgress>) {
    if let Ok(mut guard) = active_multiprogress().lock() {
        *guard = mp;
    }
}

#[derive(Clone, Debug, Default)]
struct LaneState {
    label: String,
    current: usize,
    total: Option<usize>,
    done: bool,
}

#[derive(Clone, Debug)]
struct VisualizerModel {
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
    primary_lane: LaneState,
    secondary_lane: LaneState,
    edf_terms: Vec<(String, f64, f64)>,
    diagnostics_lines: Vec<String>,
    diagnostics_condition: Option<f64>,
    diagnostics_step_size: Option<f64>,
    diagnostics_ridge: Option<f64>,
}

impl Default for VisualizerModel {
    fn default() -> Self {
        Self {
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
            primary_lane: LaneState::default(),
            secondary_lane: LaneState::default(),
            edf_terms: Vec::new(),
            diagnostics_lines: Vec::new(),
            diagnostics_condition: None,
            diagnostics_step_size: None,
            diagnostics_ridge: None,
        }
    }
}

enum VisualizerState {
    Disabled,
    Interactive(InteractiveVisualizer),
    Dumb(DumbVisualizer),
}

pub struct VisualizerSession {
    state: VisualizerState,
    model: VisualizerModel,
}

struct InteractiveVisualizer {
    terminal: Terminal<CrosstermBackend<Stdout>>,
    last_draw: Instant,
}

struct DumbVisualizer {
    multi: Option<MultiProgress>,
    primary_bar: ProgressBar,
    secondary_bar: ProgressBar,
    last_draw: Instant,
    text_only: bool,
    last_lines: Vec<String>,
}

impl InteractiveVisualizer {
    fn new() -> io::Result<Self> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(stdout);
        let terminal = Terminal::new(backend)?;
        Ok(Self {
            terminal,
            last_draw: Instant::now() - INTERACTIVE_DRAW_INTERVAL,
        })
    }

    fn maybe_draw(&mut self, model: &VisualizerModel, force: bool) {
        if !force && self.last_draw.elapsed() < INTERACTIVE_DRAW_INTERVAL {
            return;
        }
        let _ = self.draw(model);
        self.last_draw = Instant::now();
    }

    fn draw(&mut self, model: &VisualizerModel) -> io::Result<()> {
        let cost_accepted = model.history_cost_accepted.clone();
        let cost_trial = model.history_cost_trial.clone();
        let grad_data = model.history_grad_log.clone();
        let status = model.current_status.clone();
        let stage = model.current_stage.clone();
        let detail = model.current_detail.clone();
        let eval_state = model.current_eval_state.clone();
        let iter = model.current_iter;
        let best = model.best_cost;
        let elapsed = model.start_time.elapsed().as_secs();
        let current_cost = model.current_cost;
        let current_grad = model.current_grad;
        let primary_lane = model.primary_lane.clone();
        let secondary_lane = model.secondary_lane.clone();
        let edf_terms = model.edf_terms.clone();
        let diagnostics_lines = model.diagnostics_lines.clone();
        let diagnostics_condition = model.diagnostics_condition;
        let diagnostics_step_size = model.diagnostics_step_size;
        let diagnostics_ridge = model.diagnostics_ridge;
        let status_class = classify_model_status(model);
        let accent = status_color(status_class);
        let primary_class = classify_lane_status(&primary_lane, status_class);
        let secondary_class = classify_lane_status(&secondary_lane, status_class);

        self.terminal.draw(|f| {
            let area = f.area();
            let compact = area.height < 30 || area.width < 120;
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints(if compact {
                    [Constraint::Percentage(68), Constraint::Percentage(32)]
                } else {
                    [Constraint::Percentage(72), Constraint::Percentage(28)]
                })
                .split(area);
            let top_chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints(if compact {
                    [Constraint::Percentage(58), Constraint::Percentage(42)]
                } else {
                    [Constraint::Percentage(62), Constraint::Percentage(38)]
                })
                .split(chunks[0]);

            let (min_y, max_y) = if cost_accepted.is_empty() && cost_trial.is_empty() {
                (0.0, 1.0)
            } else {
                let minval = cost_accepted
                    .iter()
                    .chain(cost_trial.iter())
                    .map(|(_, y)| *y)
                    .fold(f64::INFINITY, f64::min);
                let maxval = cost_accepted
                    .iter()
                    .chain(cost_trial.iter())
                    .map(|(_, y)| *y)
                    .fold(f64::NEG_INFINITY, f64::max);
                (minval, maxval)
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
                        .border_style(Style::default().fg(accent))
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

            let right_chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints(if compact {
                    [Constraint::Percentage(36), Constraint::Percentage(64)]
                } else {
                    [Constraint::Percentage(34), Constraint::Percentage(66)]
                })
                .split(top_chunks[1]);

            let summary_lines = summary_panel_lines(
                &stage,
                &detail,
                &status,
                elapsed,
                &primary_lane,
                &secondary_lane,
                primary_class,
                secondary_class,
            );
            let summary_panel = Paragraph::new(summary_lines)
                .block(
                    Block::default()
                        .title("Session")
                        .border_style(Style::default().fg(accent))
                        .borders(Borders::ALL),
                )
                .wrap(Wrap { trim: true });
            f.render_widget(summary_panel, right_chunks[0]);

            let mut model_lines = Vec::<String>::new();
            if edf_terms.is_empty() {
                model_lines
                    .push("  Waiting for effective degrees of freedom updates...".to_string());
            } else {
                let maxrows = right_chunks[1]
                    .height
                    .saturating_sub(if compact { 3 } else { 4 })
                    as usize;
                let mut shown_terms = edf_terms.clone();
                shown_terms
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                if shown_terms.len() > maxrows.saturating_sub(1) && maxrows > 1 {
                    shown_terms.truncate(maxrows - 1);
                    model_lines.push(format!("showing top {} terms by EDF", shown_terms.len()));
                }
                let namew = shown_terms
                    .iter()
                    .map(|(name, _, _)| name.len())
                    .max()
                    .unwrap_or(8)
                    .min(if compact { 14 } else { 20 })
                    .max(6);
                for (name, edf, ref_df) in shown_terms {
                    let ratio = if ref_df > 0.0 {
                        (edf / ref_df).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    let barwidth = if compact { 12 } else { 20 };
                    let bar = unicode_bar(ratio, barwidth);
                    let displayname = if name.len() > namew {
                        let keep = namew.saturating_sub(1);
                        format!("{}…", &name[..keep])
                    } else {
                        name
                    };
                    model_lines.push(format!(
                        "{:<namew$} effective {:>6.2} of {:>6.2} reference  {}",
                        displayname,
                        edf,
                        ref_df,
                        bar,
                        namew = namew
                    ));
                }
            }

            let model_panel = Paragraph::new(model_lines.join("\n"))
                .block(
                    Block::default()
                        .title("Effective Degrees of Freedom")
                        .border_style(Style::default().fg(Color::LightCyan))
                        .borders(Borders::ALL),
                )
                .style(Style::default().fg(Color::White))
                .wrap(Wrap { trim: true });
            f.render_widget(model_panel, right_chunks[1]);

            let mut diagnostics = Vec::<String>::new();
            diagnostics.push(format!("Stage: {stage}"));
            diagnostics.push(format!("Detail: {detail}"));
            diagnostics.push(format!("Evaluation: {:.0} {eval_state}", iter));
            diagnostics.push(format!("Status: {status}"));
            diagnostics.push(format!("Best Objective: {:.6}", best));
            diagnostics.push(format!("Current Objective: {:.6}", current_cost));
            diagnostics.push(format!("Gradient Norm: {:.3e}", current_grad));
            diagnostics.push(format!("Elapsed Time: {elapsed}s"));
            diagnostics.push(String::new());
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
                "Convergence signal (base 10 log gradient norm): {:.3}",
                grad_data.last().map(|(_, y)| *y).unwrap_or(0.0)
            ));
            diagnostics.push("Recent Diagnostics and Warnings:".to_string());

            let base_reserved = diagnostics.len() + 1;
            let availrows = chunks[1].height.saturating_sub(2) as usize;
            let max_diag_lines = availrows.saturating_sub(base_reserved).max(1);
            if diagnostics_lines.is_empty() {
                diagnostics.push("  (no diagnostics yet)".to_string());
            } else {
                let start = diagnostics_lines.len().saturating_sub(max_diag_lines);
                for line in diagnostics_lines.into_iter().skip(start) {
                    diagnostics.push(format!("  {line}"));
                }
            }
            diagnostics.push("Press Ctrl+C to abort".to_string());

            let diagnostics_panel = Paragraph::new(diagnostics.join("\n"))
                .block(
                    Block::default()
                        .title("Diagnostics")
                        .border_style(Style::default().fg(accent))
                        .borders(Borders::ALL),
                )
                .style(Style::default().fg(Color::White))
                .wrap(Wrap { trim: true });
            f.render_widget(diagnostics_panel, chunks[1]);
        })?;
        Ok(())
    }

    fn teardown(&mut self, model: &VisualizerModel) {
        self.maybe_draw(model, true);
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen);
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StatusClass {
    Running,
    Success,
    Warning,
    Error,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LaneTone {
    Primary,
    Secondary,
}

impl DumbVisualizer {
    fn new(text_only: bool) -> Self {
        let (multi, primary_bar, secondary_bar) = if text_only {
            (None, ProgressBar::hidden(), ProgressBar::hidden())
        } else {
            let multi = MultiProgress::with_draw_target(ProgressDrawTarget::stderr());
            let primary_bar = multi.add(ProgressBar::hidden());
            let secondary_bar = multi.add(ProgressBar::hidden());
            primary_bar.set_style(primary_style());
            secondary_bar.set_style(secondary_style());
            install_multiprogress(Some(multi.clone()));
            (Some(multi), primary_bar, secondary_bar)
        };
        Self {
            multi,
            primary_bar,
            secondary_bar,
            last_draw: Instant::now() - DUMB_DRAW_INTERVAL,
            text_only,
            last_lines: Vec::new(),
        }
    }

    fn maybe_draw(&mut self, model: &VisualizerModel, force: bool) {
        if !self.text_only {
            self.sync_bars(model);
        }
        if !force && self.last_draw.elapsed() < DUMB_DRAW_INTERVAL {
            return;
        }
        self.draw(model, force);
        self.last_draw = Instant::now();
    }

    fn sync_bars(&self, model: &VisualizerModel) {
        sync_bar(&self.primary_bar, &model.primary_lane, &model);
        sync_bar(&self.secondary_bar, &model.secondary_lane, &model);
    }

    fn draw(&mut self, model: &VisualizerModel, force: bool) {
        let lines = dumb_render_lines(model);
        if !force && lines == self.last_lines {
            return;
        }
        self.last_lines = lines.clone();
        if let Some(multi) = &self.multi {
            for line in lines {
                let _ = multi.println(line);
            }
        } else {
            for line in lines {
                let _ = writeln!(io::stderr(), "{line}");
            }
        }
    }

    fn teardown(&mut self, model: &VisualizerModel) {
        self.maybe_draw(model, true);
        if let Some(multi) = &self.multi {
            let _ = multi.clear();
        }
        install_multiprogress(None);
    }
}

fn primary_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "{spinner:.cyan} [{elapsed_precise}] ▕{bar:40.cyan/blue}▏ {percent:>3}% | ETA: {eta_precise} | {msg}",
    )
    .unwrap_or_else(|_| ProgressStyle::default_bar())
    .progress_chars("█▉▊▋▌▍▎▏ ")
}

fn secondary_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "{spinner:.magenta} [{elapsed_precise}] ▕{bar:40.magenta/blue}▏ {percent:>3}% | ETA: {eta_precise} | {msg}",
    )
    .unwrap_or_else(|_| ProgressStyle::default_bar())
    .progress_chars("█▉▊▋▌▍▎▏ ")
}

fn sync_bar(bar: &ProgressBar, lane: &LaneState, model: &VisualizerModel) {
    if lane.label.is_empty() {
        bar.set_draw_target(ProgressDrawTarget::hidden());
        return;
    }
    bar.set_draw_target(ProgressDrawTarget::stderr());
    match lane.total {
        Some(total) if total > 0 => {
            bar.set_length(total as u64);
            bar.set_position(lane.current.min(total) as u64);
        }
        _ => {
            bar.enable_steady_tick(Duration::from_millis(120));
        }
    }
    let mut msg = format!(
        "{} | LAML: {} | |grad|: {} | Ridge: {}",
        lane.label,
        format_metric(model.current_cost, "{:.4}"),
        format_metric(model.current_grad, "{:.3e}"),
        model
            .diagnostics_ridge
            .map(|v| format!("{v:.1e}"))
            .unwrap_or_else(|| "n/a".to_string()),
    );
    if lane.done {
        msg.push_str(" | converged");
    }
    bar.set_style(style_for_lane(
        classify_lane_status(lane, classify_model_status(model)),
        if lane.label.contains("Warmup")
            || lane.label.contains("Sample")
            || lane.label.contains("Inner")
        {
            LaneTone::Secondary
        } else {
            LaneTone::Primary
        },
    ));
    bar.set_message(msg);
    if lane.done {
        bar.finish_and_clear();
    }
}

impl Default for VisualizerSession {
    fn default() -> Self {
        Self {
            state: VisualizerState::Disabled,
            model: VisualizerModel::default(),
        }
    }
}

impl VisualizerSession {
    pub fn new(enabled: bool) -> Self {
        init_logging();
        if !enabled {
            return Self::default();
        }
        let notebook_or_noninteractive = should_use_text_only_progress();
        let interactive = !notebook_or_noninteractive
            && io::stdout().is_terminal()
            && io::stderr().is_terminal();
        let state = if interactive {
            match InteractiveVisualizer::new() {
                Ok(v) => VisualizerState::Interactive(v),
                Err(_) => VisualizerState::Dumb(DumbVisualizer::new(true)),
            }
        } else {
            VisualizerState::Dumb(DumbVisualizer::new(true))
        };
        Self {
            state,
            model: VisualizerModel::default(),
        }
    }

    pub fn is_active(&self) -> bool {
        !matches!(self.state, VisualizerState::Disabled)
    }

    pub fn is_interactive(&self) -> bool {
        matches!(self.state, VisualizerState::Interactive(_))
    }

    pub fn update(
        &mut self,
        cost: f64,
        grad_norm: f64,
        status_msg: &str,
        iter: f64,
        eval_state: &str,
        total_steps: Option<usize>,
    ) {
        self.model.current_iter = iter;
        self.model.current_cost = cost;
        self.model.current_grad = grad_norm;
        self.model.current_eval_state = eval_state.to_string();
        self.model.current_status = status_msg.to_string();

        if let Some(total) = total_steps {
            self.model.primary_lane.total = Some(total);
            self.model.primary_lane.current = iter.max(0.0).round() as usize;
        }

        if cost.is_finite() && cost.abs() < 1e15 {
            let target_series = if eval_state == "trial" {
                &mut self.model.history_cost_trial
            } else {
                &mut self.model.history_cost_accepted
            };
            push_sample(target_series, (iter, cost));
            if cost < self.model.best_cost {
                self.model.best_cost = cost;
            }
        }

        if grad_norm.is_finite() {
            let grad_log = grad_norm.max(1e-12).log10();
            push_sample(&mut self.model.history_grad_log, (iter, grad_log));
        }

        self.redraw(false);
    }

    pub fn set_stage(&mut self, stage: &str, detail: &str) {
        self.model.current_stage = stage.to_string();
        self.model.current_detail = detail.to_string();
        self.model.current_status = detail.to_string();
        self.redraw(true);
    }

    pub fn start_workflow(&mut self, label: &str, total: usize) {
        self.model.primary_lane = started_lane(label, total);
        self.redraw(true);
    }

    pub fn advance_workflow(&mut self, current: usize) {
        if self.model.primary_lane.label.is_empty() {
            return;
        }
        advance_lane(&mut self.model.primary_lane, current);
        self.redraw(false);
    }

    pub fn start_secondary_workflow(&mut self, label: &str, total: usize) {
        self.model.secondary_lane = started_lane(label, total);
        self.redraw(true);
    }

    pub fn advance_secondary_workflow(&mut self, current: usize) {
        if self.model.secondary_lane.label.is_empty() {
            return;
        }
        advance_lane(&mut self.model.secondary_lane, current);
        self.redraw(false);
    }

    pub fn finish_secondary_progress(&mut self, message: &str) {
        if self.model.secondary_lane.label.is_empty() {
            return;
        }
        if let Some(total) = self.model.secondary_lane.total {
            self.model.secondary_lane.current = total;
        }
        self.model.secondary_lane.done = true;
        self.push_diagnostic(message);
        self.redraw(true);
        self.model.secondary_lane = LaneState::default();
    }

    pub fn finish_progress(&mut self, message: &str) {
        if self.model.primary_lane.label.is_empty() {
            return;
        }
        if let Some(total) = self.model.primary_lane.total {
            self.model.primary_lane.current = total;
        }
        self.model.primary_lane.done = true;
        self.push_diagnostic(message);
        self.redraw(true);
    }

    pub fn set_edf_terms(&mut self, terms: &[(String, f64, f64)]) {
        self.model.edf_terms = terms.to_vec();
        self.redraw(false);
    }

    pub fn set_diagnostics(
        &mut self,
        condition_number: Option<f64>,
        step_size: Option<f64>,
        ridge: Option<f64>,
    ) {
        self.model.diagnostics_condition = condition_number;
        self.model.diagnostics_step_size = step_size;
        self.model.diagnostics_ridge = ridge;
        self.redraw(false);
    }

    pub fn push_diagnostic(&mut self, message: &str) {
        self.model.diagnostics_lines.push(message.to_string());
        if self.model.diagnostics_lines.len() > MAX_DIAGNOSTIC_LINES {
            let overflow = self.model.diagnostics_lines.len() - MAX_DIAGNOSTIC_LINES;
            self.model.diagnostics_lines.drain(0..overflow);
        }
        self.redraw(true);
    }

    pub fn teardown(&mut self) {
        match &mut self.state {
            VisualizerState::Disabled => {}
            VisualizerState::Interactive(vis) => vis.teardown(&self.model),
            VisualizerState::Dumb(vis) => vis.teardown(&self.model),
        }
        self.state = VisualizerState::Disabled;
    }

    fn redraw(&mut self, force: bool) {
        match &mut self.state {
            VisualizerState::Disabled => {}
            VisualizerState::Interactive(vis) => vis.maybe_draw(&self.model, force),
            VisualizerState::Dumb(vis) => vis.maybe_draw(&self.model, force),
        }
    }
}

impl Drop for VisualizerSession {
    fn drop(&mut self) {
        self.teardown();
    }
}

fn push_sample(series: &mut Vec<(f64, f64)>, sample: (f64, f64)) {
    series.push(sample);
    if series.len() > MAX_HISTORY_POINTS {
        let mut compacted = Vec::with_capacity(series.len() / 2 + 1);
        for (idx, point) in series.iter().enumerate() {
            if idx % 2 == 0 {
                compacted.push(*point);
            }
        }
        *series = compacted;
    }
}

fn classify_model_status(model: &VisualizerModel) -> StatusClass {
    let status = model.current_status.to_ascii_lowercase();
    let detail = model.current_detail.to_ascii_lowercase();
    let recent_diag = model
        .diagnostics_lines
        .last()
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();
    if contains_error_signal(&status) || contains_error_signal(&detail) || contains_error_signal(&recent_diag) {
        StatusClass::Error
    } else if contains_warning_signal(&status)
        || contains_warning_signal(&detail)
        || contains_warning_signal(&recent_diag)
    {
        StatusClass::Warning
    } else if contains_success_signal(&status) || contains_success_signal(&detail) {
        StatusClass::Success
    } else {
        StatusClass::Running
    }
}

fn classify_lane_status(lane: &LaneState, model_class: StatusClass) -> StatusClass {
    if lane.done {
        StatusClass::Success
    } else {
        model_class
    }
}

fn contains_error_signal(text: &str) -> bool {
    ["error", "failed", "indefinite", "abort"].iter().any(|needle| text.contains(needle))
}

fn contains_warning_signal(text: &str) -> bool {
    ["warning", "stalled", "max iterations", "non-finite"].iter().any(|needle| text.contains(needle))
}

fn contains_success_signal(text: &str) -> bool {
    ["complete", "converged", "finished", "done"].iter().any(|needle| text.contains(needle))
}

fn status_color(class: StatusClass) -> Color {
    match class {
        StatusClass::Running => Color::Cyan,
        StatusClass::Success => Color::Green,
        StatusClass::Warning => Color::Yellow,
        StatusClass::Error => Color::Red,
    }
}

fn status_label(class: StatusClass) -> &'static str {
    match class {
        StatusClass::Running => "RUNNING",
        StatusClass::Success => "CONVERGED",
        StatusClass::Warning => "WARNING",
        StatusClass::Error => "ERROR",
    }
}

fn style_for_lane(class: StatusClass, tone: LaneTone) -> ProgressStyle {
    let template = match (tone, class) {
        (LaneTone::Primary, StatusClass::Running) => {
            "{spinner:.cyan} [{elapsed_precise}] ▕{bar:40.cyan/blue}▏ {percent:>3}% | ETA: {eta_precise} | {msg}"
        }
        (LaneTone::Primary, StatusClass::Success) => {
            "{spinner:.green} [{elapsed_precise}] ▕{bar:40.green/blue}▏ {percent:>3}% | ETA: {eta_precise} | {msg}"
        }
        (LaneTone::Primary, StatusClass::Warning) => {
            "{spinner:.yellow} [{elapsed_precise}] ▕{bar:40.yellow/red}▏ {percent:>3}% | ETA: {eta_precise} | {msg}"
        }
        (LaneTone::Primary, StatusClass::Error) => {
            "{spinner:.red} [{elapsed_precise}] ▕{bar:40.red/yellow}▏ {percent:>3}% | ETA: {eta_precise} | {msg}"
        }
        (LaneTone::Secondary, StatusClass::Running) => {
            "{spinner:.magenta} [{elapsed_precise}] ▕{bar:40.magenta/blue}▏ {percent:>3}% | ETA: {eta_precise} | {msg}"
        }
        (LaneTone::Secondary, StatusClass::Success) => {
            "{spinner:.green} [{elapsed_precise}] ▕{bar:40.green/magenta}▏ {percent:>3}% | ETA: {eta_precise} | {msg}"
        }
        (LaneTone::Secondary, StatusClass::Warning) => {
            "{spinner:.yellow} [{elapsed_precise}] ▕{bar:40.yellow/magenta}▏ {percent:>3}% | ETA: {eta_precise} | {msg}"
        }
        (LaneTone::Secondary, StatusClass::Error) => {
            "{spinner:.red} [{elapsed_precise}] ▕{bar:40.red/magenta}▏ {percent:>3}% | ETA: {eta_precise} | {msg}"
        }
    };
    ProgressStyle::with_template(template)
        .unwrap_or_else(|_| ProgressStyle::default_bar())
        .progress_chars("█▉▊▋▌▍▎▏ ")
}

fn summary_panel_lines(
    stage: &str,
    detail: &str,
    status: &str,
    elapsed: u64,
    primary_lane: &LaneState,
    secondary_lane: &LaneState,
    primary_class: StatusClass,
    secondary_class: StatusClass,
) -> Vec<TextLine<'static>> {
    let model_class = if secondary_class == StatusClass::Error || primary_class == StatusClass::Error {
        StatusClass::Error
    } else if secondary_class == StatusClass::Warning || primary_class == StatusClass::Warning {
        StatusClass::Warning
    } else if primary_class == StatusClass::Success && (secondary_lane.label.is_empty() || secondary_class == StatusClass::Success) {
        StatusClass::Success
    } else {
        StatusClass::Running
    };
    let mut lines = vec![
        TextLine::from(vec![
            Span::styled(
                format!(" {} ", status_label(model_class)),
                Style::default()
                    .fg(Color::Black)
                    .bg(status_color(model_class))
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("  "),
            Span::styled(stage.to_ascii_uppercase(), Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
        ]),
        TextLine::from(Span::styled(
            detail.to_string(),
            Style::default().fg(Color::Gray),
        )),
        TextLine::from(Span::styled(
            format!("Elapsed {}s", elapsed),
            Style::default().fg(Color::DarkGray),
        )),
        lane_line("Outer", primary_lane, primary_class),
    ];
    if !secondary_lane.label.is_empty() {
        lines.push(lane_line("Inner", secondary_lane, secondary_class));
    }
    lines.push(TextLine::from(Span::styled(
        status.to_string(),
        Style::default().fg(status_color(model_class)),
    )));
    lines
}

fn lane_line(name: &str, lane: &LaneState, class: StatusClass) -> TextLine<'static> {
    let color = status_color(class);
    let label = if lane.label.is_empty() {
        format!("{name}: idle")
    } else {
        match lane.total {
            Some(total) if total > 0 => {
                let ratio = lane.current.min(total) as f64 / total as f64;
                format!(
                    "{name}: {} ▕{}▏ {}/{}",
                    lane.label,
                    unicode_bar(ratio, 12),
                    lane.current.min(total),
                    total
                )
            }
            _ => format!("{name}: {} step {}", lane.label, lane.current),
        }
    };
    TextLine::from(Span::styled(label, Style::default().fg(color)))
}

fn dumb_render_lines(model: &VisualizerModel) -> Vec<String> {
    let mut lines = Vec::new();
    let elapsed = model.start_time.elapsed().as_secs();
    let status_class = classify_model_status(model);
    lines.push(format!(
        "[{}] {} | {} | elapsed={}s | status={}",
        status_label(status_class),
        model.current_stage,
        model.current_detail,
        elapsed,
        model.current_status
    ));
    if !model.primary_lane.label.is_empty() {
        lines.push(render_dumb_lane("Outer Opt", &model.primary_lane, model));
    }
    if !model.secondary_lane.label.is_empty() {
        lines.push(render_dumb_lane("Inner PIRLS", &model.secondary_lane, model));
    }
    if let Some(last) = model.diagnostics_lines.last() {
        lines.push(format!("Note: {last}"));
    }
    lines
}

fn render_dumb_lane(prefix: &str, lane: &LaneState, model: &VisualizerModel) -> String {
    match lane.total {
        Some(total) if total > 0 => {
            let ratio = lane.current.min(total) as f64 / total as f64;
            format!(
                "{prefix}: {:>3}% ▕{}▏ ETA: {} | {} | LAML: {} | |grad|: {}",
                (ratio * 100.0).round() as usize,
                unicode_bar(ratio, 16),
                estimate_eta(model, lane),
                lane.label,
                format_metric(model.current_cost, "{:.4}"),
                format_metric(model.current_grad, "{:.3e}"),
            )
        }
        _ => format!(
            "{prefix}: {} | step={} | LAML: {} | |grad|: {}",
            lane.label,
            lane.current,
            format_metric(model.current_cost, "{:.4}"),
            format_metric(model.current_grad, "{:.3e}"),
        ),
    }
}

fn estimate_eta(model: &VisualizerModel, lane: &LaneState) -> String {
    let Some(total) = lane.total else {
        return "n/a".to_string();
    };
    if lane.current == 0 {
        return "n/a".to_string();
    }
    let elapsed = model.start_time.elapsed().as_secs_f64();
    let rate = lane.current as f64 / elapsed.max(1e-6);
    let remaining = total.saturating_sub(lane.current) as f64 / rate.max(1e-6);
    format!("{:.0}s", remaining.max(0.0))
}

fn format_metric(value: f64, fmt: &str) -> String {
    if value.is_finite() {
        if fmt == "{:.3e}" {
            format!("{value:.3e}")
        } else {
            format!("{value:.4}")
        }
    } else {
        "n/a".to_string()
    }
}

fn unicode_bar(ratio: f64, width: usize) -> String {
    const STEPS: [char; 9] = [' ', '▏', '▎', '▍', '▌', '▋', '▊', '▉', '█'];
    let clamped = ratio.clamp(0.0, 1.0);
    let filled = clamped * width as f64;
    let full = filled.floor() as usize;
    let partial = ((filled - full as f64) * 8.0).round() as usize;
    let mut out = String::with_capacity(width);
    for idx in 0..width {
        if idx < full {
            out.push('█');
        } else if idx == full && partial > 0 {
            out.push(STEPS[partial.min(8)]);
        } else {
            out.push(' ');
        }
    }
    out
}

fn normalize_total(total: usize) -> usize {
    total.max(1)
}

fn should_use_text_only_progress() -> bool {
    if !io::stdout().is_terminal() || !io::stderr().is_terminal() {
        return true;
    }
    [
        "JPY_PARENT_PID",
        "IPYKERNEL_CELL_NAME",
        "COLAB_RELEASE_TAG",
        "GITHUB_ACTIONS",
    ]
    .iter()
    .any(|key| env::var_os(key).is_some())
}

fn started_lane(label: &str, total: usize) -> LaneState {
    LaneState {
        label: label.to_string(),
        current: 0,
        total: Some(normalize_total(total)),
        done: false,
    }
}

fn advance_lane(lane: &mut LaneState, current: usize) {
    let next = match lane.total {
        Some(total) => current.min(total),
        None => current,
    };
    lane.current = lane.current.max(next);
    lane.done = false;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unicode_bar_uses_partial_blocks() {
        let bar = unicode_bar(0.53, 8);
        assert!(bar.contains('█'));
        assert!(bar.contains('▎') || bar.contains('▍') || bar.contains('▌') || bar.contains('▋'));
    }

    #[test]
    fn dumb_renderer_throttles_to_seconds_and_formats_eta() {
        let mut session = VisualizerSession::new(false);
        session.model.start_time = Instant::now() - Duration::from_secs(4);
        session.set_stage("fit", "optimizing");
        session.start_workflow("REML", 10);
        session.advance_workflow(4);
        session.update(42.0, 1e-3, "optimizing", 4.0, "accepted", Some(10));
        let lines = dumb_render_lines(&session.model);
        assert!(lines.iter().any(|line| line.contains("Outer Opt:")));
        assert!(lines.iter().any(|line| line.contains("ETA:")));
    }

    #[test]
    fn diagnostics_ring_buffer_drops_old_messages() {
        let mut session = VisualizerSession::new(false);
        for idx in 0..20 {
            session.push_diagnostic(&format!("line {idx}"));
        }
        assert_eq!(session.model.diagnostics_lines.len(), MAX_DIAGNOSTIC_LINES);
        assert_eq!(session.model.diagnostics_lines.first().map(String::as_str), Some("line 10"));
    }

    #[test]
    fn finishing_secondary_progress_clears_inner_lane() {
        let mut session = VisualizerSession::new(false);
        session.start_secondary_workflow("inner", 5);
        session.advance_secondary_workflow(2);
        assert_eq!(session.model.secondary_lane.label, "inner");
        session.finish_secondary_progress("done");
        assert!(session.model.secondary_lane.label.is_empty());
        assert_eq!(
            session.model.diagnostics_lines.last().map(String::as_str),
            Some("done")
        );
    }

    #[test]
    fn enabled_session_uses_dumb_mode_when_not_attached_to_tty() {
        let session = VisualizerSession::new(true);
        assert!(session.is_active());
        assert!(!session.is_interactive());
    }

    #[test]
    fn finish_progress_keeps_completion_diagnostic() {
        let mut session = VisualizerSession::new(false);
        session.start_workflow("outer", 10);
        session.advance_workflow(5);
        session.finish_progress("outer complete");
        assert_eq!(
            session.model.diagnostics_lines.last().map(String::as_str),
            Some("outer complete")
        );
        assert!(session.model.primary_lane.done);
        assert_eq!(session.model.primary_lane.current, 10);
    }

    #[test]
    fn workflow_normalizes_zero_total_and_never_goes_backwards() {
        let mut session = VisualizerSession::new(false);
        session.start_workflow("outer", 0);
        assert_eq!(session.model.primary_lane.total, Some(1));
        session.advance_workflow(1);
        session.advance_workflow(0);
        assert_eq!(session.model.primary_lane.current, 1);
    }

    #[test]
    fn secondary_workflow_ignores_advances_before_start() {
        let mut session = VisualizerSession::new(false);
        session.advance_secondary_workflow(3);
        assert!(session.model.secondary_lane.label.is_empty());
        session.start_secondary_workflow("inner", 2);
        session.advance_secondary_workflow(5);
        assert_eq!(session.model.secondary_lane.current, 2);
    }

    #[test]
    fn warning_status_is_reflected_in_dumb_render() {
        let mut session = VisualizerSession::new(false);
        session.set_stage("fit", "optimizing");
        session.push_diagnostic("warning: matrix ill-conditioned");
        let lines = dumb_render_lines(&session.model);
        assert!(lines.first().is_some_and(|line| line.contains("[WARNING]")));
    }
}
