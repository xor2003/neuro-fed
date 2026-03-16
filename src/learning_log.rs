use std::{fs::OpenOptions, io::Write};

/// Appends the provided learning details to `detail.log`, creating the file if it does not exist.
pub fn append_learning_detail(entry: &str) {
    if let Ok(mut file) = OpenOptions::new()
        .create(true)
        .append(true)
        .open("detail.log")
    {
        if let Err(err) = writeln!(file, "{}\n---", entry) {
            tracing::warn!(error = %err, "Failed to write detail.log entry");
        }
    } else {
        tracing::warn!("Unable to open detail.log for appending");
    }
}
