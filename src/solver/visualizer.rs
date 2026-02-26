use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::prelude::*;
use ratatui::widgets::{Axis, Block, Borders, Chart, Dataset, Gauge, GraphType, Paragraph};
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
}

impl OptimizationVisualizer {
    fn new() -> io::Result<Self> {
        if !io::stdout().is_terminal() {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "stdout is not a terminal",
            ));
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
            current_detail: "".to_string(),
            current_eval_state: "".to_string(),
            current_cost: f64::NAN,
            current_grad: f64::NAN,
            last_draw: Instant::now(),
            progress_label: "".to_string(),
            progress_current: 0,
            progress_total: None,
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

        self.terminal.draw(|f| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
                .split(f.area());

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
                .block(Block::default().title("Optimization Trajectory").borders(Borders::ALL))
                .x_axis(
                    Axis::default()
                        .title("Iterations")
                        .bounds([0.0, iter.max(10.0)])
                        .labels(vec![
                            Line::from("0"),
                            Line::from(format!("{:.0}", iter.max(10.0))),
                        ]),
                )
                .y_axis(
                    Axis::default()
                        .title("Cost")
                        .bounds([min_y - window * 0.1, max_y + window * 0.1])
                        .labels(vec![
                            Line::from(format!("{:.2}", min_y)),
                            Line::from(format!("{:.2}", max_y)),
                        ]),
                );
            f.render_widget(chart, chunks[0]);

            let bottom_chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
                .split(chunks[1]);

            let left_chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
                .split(bottom_chunks[0]);

            let info_text = format!(
                "Stage: {}\nDetail: {}\nEval: {:.0} {}\nStatus: {}\nBest Cost: {:.6}\nCurrent Cost: {:.6}\nGrad Norm: {:.3e}\nElapsed: {}s\n\nPress Ctrl+C to abort",
                stage,
                detail,
                iter,
                eval_state,
                status,
                best,
                current_cost,
                current_grad,
                elapsed
            );
            let info = Paragraph::new(info_text)
                .block(Block::default().title("Statistics").borders(Borders::ALL))
                .style(Style::default().fg(Color::White));
            f.render_widget(info, left_chunks[0]);

            if let Some(total) = progress_total {
                let ratio = if total == 0 {
                    0.0
                } else {
                    (progress_current as f64 / total as f64).clamp(0.0, 1.0)
                };
                let label = format!("{}: {}/{}", progress_label, progress_current, total);
                let gauge = Gauge::default()
                    .block(Block::default().title("Progress").borders(Borders::ALL))
                    .gauge_style(Style::default().fg(Color::Green))
                    .ratio(ratio)
                    .label(label);
                f.render_widget(gauge, left_chunks[1]);
            } else {
                let label = if progress_label.is_empty() {
                    "Progress: --".to_string()
                } else {
                    format!("Progress: {} ({})", progress_label, progress_current)
                };
                let progress = Paragraph::new(label)
                    .block(Block::default().title("Progress").borders(Borders::ALL))
                    .style(Style::default().fg(Color::DarkGray));
                f.render_widget(progress, left_chunks[1]);
            }

            let max_g = grad_data
                .iter()
                .map(|(_, y)| *y)
                .fold(0.0, f64::max);
            let grad_sets = vec![Dataset::default()
                .name("log10(|grad|)")
                .marker(symbols::Marker::Dot)
                .style(Style::default().fg(Color::Yellow))
                .data(&grad_data)];
            let grad_chart = Chart::new(grad_sets)
                .block(Block::default().title("Convergence (log10 |grad|)").borders(Borders::ALL))
                .x_axis(Axis::default().bounds([0.0, iter.max(10.0)]))
                .y_axis(
                    Axis::default()
                        .bounds([max_g.min(0.0) - 0.5, max_g.max(0.0) + 0.5])
                        .labels(vec![
                            Line::from(format!("{:.1}", max_g.min(0.0))),
                            Line::from(format!("{:.1}", max_g.max(0.0))),
                        ]),
                );
            f.render_widget(grad_chart, bottom_chunks[1]);
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
