//! Notes a fit records about how it read the request.
//!
//! Two kinds, because they ask different things of the reader:
//!
//! * an **advisory** says the fitted model differs from what was literally
//!   requested, or that its meaning needs a second look — a `k` capped to the
//!   covariate's distinct values, a basis degraded to a line, a scalar term
//!   dropped as unidentifiable, a smooth whose basis failed its adequacy check.
//!   Front ends surface these where the user will see them at fit time (the
//!   CLI prints them, gamfit raises a `GamInferenceWarning`).
//! * an **informational** note records a default the engine chose on the
//!   user's behalf — the internal-knot count of a default B-spline, per-margin
//!   tensor sizes, how an interaction was wired. It is part of the fit's
//!   record (the saved model, `Model.notes`, the summary) but is not a
//!   warning: a default fit is not something the user must act on.

/// Where the term builders record their notes.
///
/// [`FitNotes`] keeps the two kinds apart for the fit pipeline; a plain
/// `Vec<String>` collects both into one list, for a caller that only reads the
/// notes back.
pub trait FitNoteSink {
    /// Record that the model differs from the literal request.
    fn advise(&mut self, note: String);
    /// Record a default choice made on the user's behalf.
    fn inform(&mut self, note: String);
}

/// A fit's notes, split by kind.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FitNotes {
    pub advisories: Vec<String>,
    pub informational: Vec<String>,
}

impl FitNotes {
    /// Every note, advisories first.
    pub fn iter(&self) -> impl Iterator<Item = &String> {
        self.advisories.iter().chain(&self.informational)
    }

    pub fn is_empty(&self) -> bool {
        self.advisories.is_empty() && self.informational.is_empty()
    }
}

impl FitNoteSink for FitNotes {
    fn advise(&mut self, note: String) {
        self.advisories.push(note);
    }

    fn inform(&mut self, note: String) {
        self.informational.push(note);
    }
}

impl FitNoteSink for Vec<String> {
    fn advise(&mut self, note: String) {
        self.push(note);
    }

    fn inform(&mut self, note: String) {
        self.push(note);
    }
}
