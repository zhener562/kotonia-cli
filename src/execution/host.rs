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

/// Initial line buffer size for each stream. Larger lines just grow the Vec.
const LINE_BUF_INITIAL: usize = 4096;

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
        // `BufReader::lines()` would be tighter, but it does strict UTF-8
        // decoding and dies when a shell command (e.g. `head -c N file.txt`)
        // chops a multibyte char in the middle. Read raw bytes and decode
        // lossily so the agent always sees something.
        let mut stdout = BufReader::new(stdout);
        let mut stderr = BufReader::new(stderr);

        let mut combined = String::new();
        let mut truncated = false;

        let drain = async {
            let mut buf_out: Vec<u8> = Vec::with_capacity(LINE_BUF_INITIAL);
            let mut buf_err: Vec<u8> = Vec::with_capacity(LINE_BUF_INITIAL);
            let mut stdout_done = false;
            let mut stderr_done = false;
            while !(stdout_done && stderr_done) {
                tokio::select! {
                    biased;
                    res = stdout.read_until(b'\n', &mut buf_out), if !stdout_done => {
                        match res {
                            Ok(0) => stdout_done = true,
                            Ok(_) => {
                                push_lossy_line(&mut combined, &buf_out, &mut truncated);
                                buf_out.clear();
                            }
                            Err(e) => return Err(HostExecutorError::Io(e)),
                        }
                    }
                    res = stderr.read_until(b'\n', &mut buf_err), if !stderr_done => {
                        match res {
                            Ok(0) => stderr_done = true,
                            Ok(_) => {
                                push_lossy_line(&mut combined, &buf_err, &mut truncated);
                                buf_err.clear();
                            }
                            Err(e) => return Err(HostExecutorError::Io(e)),
                        }
                    }
                }
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

fn push_lossy_line(combined: &mut String, raw: &[u8], truncated: &mut bool) {
    // Trim the trailing '\n' (kept by read_until) so append_capped only
    // appends its own. CR stays for now — agents often want raw output.
    let raw = raw.strip_suffix(b"\n").unwrap_or(raw);
    let line = String::from_utf8_lossy(raw);
    append_capped(combined, &line, truncated);
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

    #[tokio::test]
    async fn survives_invalid_utf8_split() {
        // 「あ」 is 3 bytes in UTF-8 (0xE3 0x81 0x82). Cutting at 1 byte
        // splits the multibyte char and the strict decoder would error;
        // we want the lossy path to keep going and emit a replacement.
        let exec = HostExecutor::new(std::env::temp_dir());
        let r = exec
            .bash("printf '\\xe3\\x81\\x82\\xe3\\x81\\x82' | head -c 1")
            .await
            .unwrap();
        assert_eq!(r.exit_code, 0);
        // No assertion on the exact replacement string — just that it
        // doesn't fail and produces *some* output.
        assert!(!r.combined.is_empty(), "lossy decode produced empty output");
    }
}
