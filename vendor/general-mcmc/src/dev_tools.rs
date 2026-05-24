use std::time::Instant;

/// A simple timer for logging elapsed durations.
///
/// The [`Timer`] struct is designed to measure and log the time elapsed between successive events or operations.
/// It can be used to benchmark or trace code execution by printing the elapsed time along with a provided message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Timer {
    last: Instant,
}

impl Timer {
    /// Creates a new [`Timer`] instance.
    ///
    /// This initializes the timer by capturing the current instant.
    pub fn new() -> Self {
        Self {
            last: Instant::now(),
        }
    }

    /// Logs a message along with the elapsed time since the last log.
    ///
    /// This method computes the duration since the last time it was called (or since creation),
    /// prints that elapsed duration along with the provided message, and then updates the timer's
    /// internal timestamp to the current instant.
    ///
    /// # Parameters
    ///
    /// * `msg`: A message to log. The message can be of any type that implements [`std::fmt::Debug`].
    pub fn log<T: std::fmt::Debug>(&mut self, msg: T) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last);
        self.last = now;
        println!("[{:?}] {:?}", elapsed, msg);
    }
}
