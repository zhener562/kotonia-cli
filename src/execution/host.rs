//! Host-side bash executor for the kotonia-cli agent.
//!
//! Runs shell commands on the operator's machine inside a fixed cwd (typically
//! a git worktree the agent owns). Captures stdout + stderr together so the
//! agent observation matches what a human would see.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::time::timeout;

const DEFAULT_TIMEOUT_SECS: u64 = 300;
const MAX_CAPTURE_BYTES: usize = 256 * 1024;

#[derive(Debug)]
pub enum HostExecutorError {
    Spawn(std::io::Error),
    Io(std::io::Error),
}

impl std::fmt::Display for HostExecutorError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            HostExecutorError::Spawn(e) => write!(f, "failed to spawn bash: {e}"),
            HostExecutorError::Io(e) => write!(f, "host executor I/O error: {e}"),
        }
    }
}

impl std::error::Error for HostExecutorError {}

#[derive(Debug, Clone)]
pub struct ExecutionResult {
    pub exit_code: i32,
    pub timed_out: bool,
    pub truncated: bool,
    /// stdout + stderr interleaved, in arrival order. This is what the agent sees.
    pub combined: String,
}

impl ExecutionResult {
    /// Compact summary suitable for embedding in a ReAct observation.
    pub fn as_observation(&self) -> String {
        let mut header = if self.timed_out {
            format!("[timed out, exit {}]", self.exit_code)
        } else {
            format!("[exit {}]", self.exit_code)
        };
        if self.truncated {
            header.push_str(" [output truncated]");
        }
        if self.combined.is_empty() {
            header
        } else {
            format!("{header}\n{}", self.combined)
        }
    }
}

pub struct HostExecutor {
    cwd: PathBuf,
    timeout: Duration,
}

impl HostExecutor {
    pub fn new<P: Into<PathBuf>>(cwd: P) -> Self {
        Self {
            cwd: cwd.into(),
            timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
        }
    }

    pub fn with_timeout(mut self, d: Duration) -> Self {
        self.timeout = d;
        self
    }

    pub fn cwd(&self) -> &Path {
        &self.cwd
    }

    /// Run a single bash command. The agent always invokes through `bash -c`
    /// so it can use pipes, redirections, and shell built-ins without
    /// worrying about which command runner is in front.
    pub async fn bash(&self, command: &str) -> Result<ExecutionResult, HostExecutorError> {
        let mut child = Command::new("bash")
            .arg("-c")
            .arg(command)
            .current_dir(&self.cwd)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(HostExecutorError::Spawn)?;

        let stdout = child.stdout.take().expect("piped stdout");
        let stderr = child.stderr.take().expect("piped stderr");
        let mut stdout_lines = BufReader::new(stdout).lines();
        let mut stderr_lines = BufReader::new(stderr).lines();

        let mut combined = String::new();
        let mut truncated = false;

        let drain = async {
            loop {
                tokio::select! {
                    biased;
                    line = stdout_lines.next_line() => {
                        match line {
                            Ok(Some(l)) => append_capped(&mut combined, &l, &mut truncated),
                            Ok(None) => break,
                            Err(e) => return Err(HostExecutorError::Io(e)),
                        }
                    }
                    line = stderr_lines.next_line() => {
                        match line {
                            Ok(Some(l)) => append_capped(&mut combined, &l, &mut truncated),
                            Ok(None) => break,
                            Err(e) => return Err(HostExecutorError::Io(e)),
                        }
                    }
                }
            }
            // Drain whichever stream is still open.
            while let Ok(Some(l)) = stdout_lines.next_line().await {
                append_capped(&mut combined, &l, &mut truncated);
            }
            while let Ok(Some(l)) = stderr_lines.next_line().await {
                append_capped(&mut combined, &l, &mut truncated);
            }
            Ok::<(), HostExecutorError>(())
        };

        let mut timed_out = false;
        let exit_status = match timeout(self.timeout, async {
            drain.await?;
            child.wait().await.map_err(HostExecutorError::Io)
        })
        .await
        {
            Ok(res) => res?,
            Err(_) => {
                timed_out = true;
                let _ = child.start_kill();
                child.wait().await.map_err(HostExecutorError::Io)?
            }
        };

        let exit_code = exit_status.code().unwrap_or(-1);
        Ok(ExecutionResult {
            exit_code,
            timed_out,
            truncated,
            combined,
        })
    }
}

fn append_capped(buf: &mut String, line: &str, truncated: &mut bool) {
    if *truncated {
        return;
    }
    let remaining = MAX_CAPTURE_BYTES.saturating_sub(buf.len());
    if remaining == 0 {
        *truncated = true;
        return;
    }
    if line.len() + 1 > remaining {
        let mut end = remaining.saturating_sub(1);
        while !line.is_char_boundary(end) && end > 0 {
            end -= 1;
        }
        buf.push_str(&line[..end]);
        buf.push('\n');
        *truncated = true;
    } else {
        buf.push_str(line);
        buf.push('\n');
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn echoes_stdout() {
        let cwd = std::env::temp_dir();
        let exec = HostExecutor::new(cwd);
        let r = exec.bash("echo hello").await.unwrap();
        assert_eq!(r.exit_code, 0);
        assert!(r.combined.contains("hello"));
        assert!(!r.timed_out);
    }

    #[tokio::test]
    async fn captures_stderr() {
        let exec = HostExecutor::new(std::env::temp_dir());
        let r = exec.bash("echo oops 1>&2; exit 3").await.unwrap();
        assert_eq!(r.exit_code, 3);
        assert!(r.combined.contains("oops"));
    }

    #[tokio::test]
    async fn enforces_timeout() {
        let exec = HostExecutor::new(std::env::temp_dir()).with_timeout(Duration::from_millis(200));
        let r = exec.bash("sleep 5").await.unwrap();
        assert!(r.timed_out);
    }

    #[tokio::test]
    async fn respects_cwd() {
        let exec = HostExecutor::new("/tmp");
        let r = exec.bash("pwd").await.unwrap();
        assert!(r.combined.contains("/tmp"));
    }
}
