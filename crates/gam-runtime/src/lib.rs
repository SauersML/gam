mod cgroup_memory;

pub mod loop_progress;
pub mod parallel;
pub mod process_monitor;
pub mod resource;
pub mod span;
pub mod test_support;
pub mod warm_start;

/// The message carried by a panic payload, for the two payload shapes
/// `std::panic!` / `unwrap` / `expect` actually produce (`&'static str` and
/// `String`); anything else is reported as a non-string payload.
pub fn panic_payload_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&'static str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "non-string panic payload".to_string()
    }
}
