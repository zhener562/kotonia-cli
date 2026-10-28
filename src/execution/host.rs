//! Host-side bash executor for the kotonia-cli agent.
//!
//! Runs shell commands on the operator's machine inside a fixed cwd (typically
//! a git worktree the agent owns). Captures stdout + stderr together so the
//! agent observation matches what a human would see.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use serde::Serialize;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::time::timeout;

/// Initial line buffer size for each stream. Larger lines just grow the Vec.
const LINE_BUF_INITIAL: usize = 4096;

const DEFAULT_TIMEOUT_SECS: u64 = 300;
const MAX_CAPTURE_BYTES: usize = 256 * 1024;

/// What the *model* sees of one command's output. `combined` keeps up to
/// `MAX_CAPTURE_BYTES` for the UI; the observation is shaped down to this,
/// because every observation is resent on every later request in the
/// session. Measured on the local session logs: the 5 largest observations
/// were 63% of all observation bytes, and the top two (262 KB of base64
/// from a curl, 248 KB from a `grep | head -20` that hit one minified-JSON
/// line) each filled the context on their own.
const OBS_MAX_BYTES: usize = 16 * 1024;
/// Head/tail split when an observation is over `OBS_MAX_BYTES`. The tail
/// gets less, but it is where errors and exit summaries land.
const OBS_HEAD_BYTES: usize = 10 * 1024;
const OBS_TAIL_BYTES: usize = 5 * 1024;
/// `head -n` limits lines, not bytes, so one long line slips through it.
const OBS_LINE_MAX_BYTES: usize = 2000;
/// A line at least this long made only of base64 / base64url characters is
/// data, not something the model can read.
const OBS_BASE64_MIN_BYTES: usize = 256;

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

/// Program used for `bash -c`. On unix this is simply `bash` from PATH.
#[cfg(not(windows))]
fn bash_program() -> Result<PathBuf, HostExecutorError> {
    Ok(PathBuf::from("bash"))
}

/// On Windows, plain `Command::new("bash")` is a trap: PATH usually resolves
/// to `C:\Windows\System32\bash.exe`, the WSL launcher, so agent commands
/// would silently run inside a Linux distro (different filesystem view, none
/// of the env vars set below propagate without WSLENV) — or fail outright
/// when WSL has no distro installed. Resolve a win32 bash (Git Bash / MSYS2)
/// instead, skipping the WSL launcher explicitly.
#[cfg(windows)]
fn bash_program() -> Result<PathBuf, HostExecutorError> {
    static BASH: std::sync::OnceLock<Option<PathBuf>> = std::sync::OnceLock::new();
    BASH.get_or_init(find_win32_bash).clone().ok_or_else(|| {
        HostExecutorError::Spawn(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "no win32 bash.exe found (System32's bash.exe is the WSL launcher and is \
             skipped on purpose). Install Git for Windows (https://gitforwindows.org) \
             or point KOTONIA_BASH at a bash.exe",
        ))
    })
}

#[cfg(windows)]
fn find_win32_bash() -> Option<PathBuf> {
    // Explicit override first — also the escape hatch for whoever really
    // does want the WSL launcher.
    if let Some(p) = std::env::var_os("KOTONIA_BASH") {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Some(p);
        }
    }
    let path_var = std::env::var_os("PATH")?;
    // A bash.exe already on PATH (Git\bin or MSYS2 setups), minus the WSL
    // launcher (System32) and the Microsoft Store alias stub (WindowsApps).
    for dir in std::env::split_paths(&path_var) {
        let cand = dir.join("bash.exe");
        if cand.is_file() && !is_windows_bash_launcher_dir(&dir) {
            return Some(cand);
        }
    }
    // Derive from git.exe on PATH: <root>\cmd\git.exe → <root>\bin\bash.exe.
    for dir in std::env::split_paths(&path_var) {
        if !dir.join("git.exe").is_file() {
            continue;
        }
        if let Some(root) = dir.parent() {
            for rel in [
                ["bin", "bash.exe"].as_slice(),
                ["usr", "bin", "bash.exe"].as_slice(),
            ] {
                let cand = rel.iter().fold(root.to_path_buf(), |p, s| p.join(s));
                if cand.is_file() {
                    return Some(cand);
                }
            }
        }
    }
    // Standard Git for Windows install locations, for PATH-less setups.
    let roots = [
        std::env::var_os("ProgramFiles").map(|v| PathBuf::from(v).join("Git")),
        std::env::var_os("ProgramFiles(x86)").map(|v| PathBuf::from(v).join("Git")),
        std::env::var_os("LOCALAPPDATA").map(|v| PathBuf::from(v).join("Programs").join("Git")),
    ];
    for root in roots.into_iter().flatten() {
        let cand = root.join("bin").join("bash.exe");
        if cand.is_file() {
            return Some(cand);
        }
    }
    None
}

/// `System32` hosts the WSL launcher; `WindowsApps` hosts the Store's
/// app-execution-alias stubs. Neither is a bash that runs win32-side.
#[cfg(windows)]
fn is_windows_bash_launcher_dir(dir: &Path) -> bool {
    let lower = dir.to_string_lossy().replace('/', "\\").to_lowercase();
    lower.contains("\\system32") || lower.contains("\\windowsapps")
}

/// The desktop app is a GUI-subsystem process; without CREATE_NO_WINDOW a
/// console-subsystem child (bash.exe) gets a fresh visible console window
/// flashing up for every agent command.
#[cfg(windows)]
fn suppress_console_window(cmd: &mut Command) {
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    cmd.creation_flags(CREATE_NO_WINDOW);
}

#[cfg(not(windows))]
fn suppress_console_window(_cmd: &mut Command) {}

#[derive(Debug, Clone, Serialize)]
pub struct ExecutionResult {
    pub exit_code: i32,
    pub timed_out: bool,
    pub truncated: bool,
    /// stdout + stderr interleaved, in arrival order. Shown as-is to the
    /// operator; the model sees `as_observation()`, which shapes it down.
    pub combined: String,
    /// Where the full `combined` was written when the observation had to
    /// cut something, so the model can grep/sed it instead of re-running.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub full_output_path: Option<PathBuf>,
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
            return header;
        }
        let body = shape_observation(&self.combined);
        if body.elided {
            match &self.full_output_path {
                Some(p) => header.push_str(&format!(
                    " [shortened for context; full output ({} bytes) saved to {}]",
                    self.combined.len(),
                    p.display()
                )),
                None => header.push_str(" [shortened for context]"),
            }
        }
        format!("{header}\n{}", body.text)
    }
}

struct ShapedObservation {
    text: String,
    /// Whether anything was cut (a long line, base64, or the middle).
    elided: bool,
}

fn shape_observation(combined: &str) -> ShapedObservation {
    let mut elided = false;
    let mut text = String::with_capacity(combined.len().min(OBS_MAX_BYTES + 256));
    for line in combined.split_inclusive('\n') {
        let (body, nl) = match line.strip_suffix('\n') {
            Some(b) => (b, "\n"),
            None => (line, ""),
        };
        if looks_like_base64(body) {
            text.push_str(&format!("[base64-like data: {} bytes omitted]", body.len()));
            elided = true;
        } else if body.len() > OBS_LINE_MAX_BYTES {
            let end = floor_char_boundary(body, OBS_LINE_MAX_BYTES);
            text.push_str(&body[..end]);
            text.push_str(&format!(" … [line cut: +{} bytes]", body.len() - end));
            elided = true;
        } else {
            text.push_str(body);
        }
        text.push_str(nl);
    }
    if text.len() > OBS_MAX_BYTES {
        text = keep_head_and_tail(&text);
        elided = true;
    }
    ShapedObservation { text, elided }
}

/// Keep whole lines from the start up to `OBS_HEAD_BYTES` and from the end
/// up to `OBS_TAIL_BYTES`, with a marker saying how much was dropped.
fn keep_head_and_tail(text: &str) -> String {
    let mut head_end = 0;
    for (i, _) in text.match_indices('\n') {
        if i + 1 > OBS_HEAD_BYTES {
            break;
        }
        head_end = i + 1;
    }
    let mut tail_start = text.len();
    for (i, _) in text.rmatch_indices('\n') {
        // `i` is the newline ending the previous line; the tail starts after it.
        if i + 1 >= text.len() {
            continue;
        }
        if text.len() - (i + 1) > OBS_TAIL_BYTES {
            break;
        }
        tail_start = i + 1;
    }
    // Lines can only be OBS_LINE_MAX_BYTES long by now, so these only fall
    // back to a char cut if a single line is wider than the whole window.
    if head_end == 0 {
        head_end = floor_char_boundary(text, OBS_HEAD_BYTES);
    }
    if tail_start <= head_end {
        tail_start = head_end;
    }
    let omitted = &text[head_end..tail_start];
    format!(
        "{}[… {} lines / {} bytes omitted …]\n{}",
        &text[..head_end],
        omitted.lines().count(),
        omitted.len(),
        &text[tail_start..]
    )
}

fn looks_like_base64(line: &str) -> bool {
    let t = line.trim();
    t.len() >= OBS_BASE64_MIN_BYTES
        && t.bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'+' | b'/' | b'=' | b'-' | b'_'))
}

fn floor_char_boundary(s: &str, mut i: usize) -> usize {
    if i >= s.len() {
        return s.len();
    }
    while !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// Write the full output where the model's next bash call can read it. Lives
/// under the OS temp dir rather than the worktree so it never shows up in
/// `git status` or a commit. Best-effort: on failure the observation just
/// says it was shortened.
fn save_full_output(combined: &str) -> Option<PathBuf> {
    let dir = std::env::temp_dir().join("kotonia-cli");
    std::fs::create_dir_all(&dir).ok()?;
    let path = dir.join(format!("obs-{}.txt", uuid::Uuid::new_v4()));
    std::fs::write(&path, combined).ok()?;
    Some(path)
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
        let mut cmd = Command::new(bash_program()?);
        cmd.arg("-c")
            .arg(command)
            .current_dir(&self.cwd)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            // Non-interactive defaults for the tools that check env vars
            // before deciding whether to prompt. Without these, `git commit`
            // (no `-m`) launches `$EDITOR`, `git log`/`diff` launch `less`,
            // etc. — with the tty detachment below they'd still fail, just
            // less legibly (a raw ENXIO) than the clear messages these vars
            // produce ("Aborting commit due to empty commit message", etc).
            .env("GIT_TERMINAL_PROMPT", "0")
            .env("GIT_EDITOR", "true")
            .env("EDITOR", "true")
            .env("VISUAL", "true")
            .env("GIT_PAGER", "cat")
            .env("PAGER", "cat")
            .env("DEBIAN_FRONTEND", "noninteractive");
        detach_controlling_terminal(&mut cmd);
        suppress_console_window(&mut cmd);
        let mut child = cmd.spawn().map_err(HostExecutorError::Spawn)?;

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
        let full_output_path = if shape_observation(&combined).elided {
            save_full_output(&combined)
        } else {
            None
        };
        Ok(ExecutionResult {
            exit_code,
            timed_out,
            truncated,
            combined,
            full_output_path,
        })
    }
}

/// Detach the spawned child into its own session with no controlling
/// terminal. Interactive programs (`sudo`, `ssh`, `vim`/`nano`, `less`/
/// `more`, `passwd`, ...) don't read prompts from stdin — they open
/// `/dev/tty` directly, which resolves via the *session's* controlling
/// terminal, independent of whatever stdin/stdout are redirected to. Left
/// alone, the child inherits kotonia-cli's own controlling terminal (if any)
/// and that open+read genuinely blocks, for up to `self.timeout`, since
/// nothing is ever going to type into it.
///
/// `setsid()` makes the child a new session leader with no controlling
/// terminal at all, so that same `open("/dev/tty")` fails immediately with
/// `ENXIO` instead of blocking — the same technique `nohup`/daemonization
/// tooling uses. This is a structural fix (no command-name pattern
/// matching): every tty-seeking program fails fast, not just ones we
/// happened to enumerate.
#[cfg(unix)]
fn detach_controlling_terminal(cmd: &mut Command) {
    // SAFETY: the closure only calls the async-signal-safe libc::setsid()
    // between fork and exec, per `pre_exec`'s contract — no allocation, no
    // locking, nothing else.
    unsafe {
        cmd.pre_exec(|| {
            if libc::setsid() == -1 {
                let err = std::io::Error::last_os_error();
                // EPERM means the child is already a process group leader
                // (so it can't start a *new* session) — it still has no
                // controlling terminal assigned as a result, which is the
                // property we actually want, so treat this as success
                // rather than failing the spawn.
                if err.raw_os_error() != Some(libc::EPERM) {
                    return Err(err);
                }
            }
            Ok(())
        });
    }
}

#[cfg(not(unix))]
fn detach_controlling_terminal(_cmd: &mut Command) {}

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

    #[test]
    fn short_output_is_passed_through() {
        let shaped = shape_observation("a\nb\n");
        assert_eq!(shaped.text, "a\nb\n");
        assert!(!shaped.elided);
    }

    #[test]
    fn a_long_line_is_cut_even_when_line_count_is_small() {
        // The `grep ... | head -20` case: few lines, one of them huge.
        let line = "{\"k\":\"v\"},".repeat(5_000);
        let shaped = shape_observation(&format!("ok\n{line}\nend\n"));
        assert!(shaped.elided);
        assert!(shaped.text.len() < OBS_LINE_MAX_BYTES + 200);
        assert!(shaped.text.contains("[line cut: +48000 bytes]"));
        assert!(shaped.text.ends_with("end\n"));
    }

    #[test]
    fn base64_is_replaced_by_its_size() {
        let b64 = "QUJD".repeat(1000);
        let shaped = shape_observation(&format!("{{\"image\":\n{b64}\n}}\n"));
        assert!(shaped
            .text
            .contains("[base64-like data: 4000 bytes omitted]"));
        assert!(!shaped.text.contains("QUJDQUJD"));
    }

    #[test]
    fn many_lines_keep_head_and_tail() {
        let body: String = (0..5000).map(|i| format!("line {i}\n")).collect();
        let shaped = shape_observation(&body);
        assert!(shaped.elided);
        assert!(shaped.text.len() <= OBS_HEAD_BYTES + OBS_TAIL_BYTES + 100);
        assert!(shaped.text.starts_with("line 0\n"));
        assert!(shaped.text.ends_with("line 4999\n"));
        assert!(shaped.text.contains("lines /"));
    }

    #[test]
    fn multibyte_lines_are_cut_on_char_boundaries() {
        let line = "あ".repeat(5000);
        let shaped = shape_observation(&line);
        assert!(shaped.elided);
        assert!(shaped.text.starts_with("あ"));
    }

    #[tokio::test]
    async fn oversized_output_is_saved_and_pointed_to() {
        let exec = HostExecutor::new(std::env::temp_dir());
        let r = exec
            .bash("head -c 100000 /dev/zero | tr '\\0' a; echo; echo done")
            .await
            .unwrap();
        let path = r.full_output_path.clone().expect("full output saved");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), r.combined);
        let obs = r.as_observation();
        assert!(obs.contains(&path.display().to_string()));
        assert!(obs.len() < OBS_LINE_MAX_BYTES + 500);
        assert!(obs.trim_end().ends_with("done"));
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn small_output_writes_no_file() {
        let exec = HostExecutor::new(std::env::temp_dir());
        let r = exec.bash("echo hi").await.unwrap();
        assert!(r.full_output_path.is_none());
        assert_eq!(r.as_observation(), "[exit 0]\nhi\n");
    }

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

    #[tokio::test]
    async fn tty_open_fails_fast_instead_of_hanging() {
        // Without the setsid detachment, this would inherit whatever
        // controlling terminal the test runner has (if any) and block
        // opening /dev/tty waiting for input nobody will supply. With
        // detachment, the child has no controlling terminal at all, so the
        // open fails immediately (ENXIO) and bash's `exec` redirection
        // fails right away instead of hanging for the executor timeout.
        let exec = HostExecutor::new(std::env::temp_dir()).with_timeout(Duration::from_secs(5));
        let r = exec.bash("exec 3< /dev/tty").await.unwrap();
        assert!(
            !r.timed_out,
            "opening /dev/tty blocked instead of failing fast"
        );
        assert_ne!(r.exit_code, 0);
    }

    #[tokio::test]
    async fn git_commit_without_message_fails_fast() {
        // `git commit` with no `-m` normally launches `$EDITOR` and blocks
        // waiting for a human to write a commit message. `GIT_EDITOR=true`
        // makes that a no-op, so git sees an unmodified (comments-only)
        // message and aborts immediately instead of hanging.
        let dir = tempfile::tempdir().unwrap();
        let exec = HostExecutor::new(dir.path()).with_timeout(Duration::from_secs(10));
        let setup = exec
            .bash(
                "git init -q && git config user.email a@b.c && \
                 git config user.name t && echo hi > f.txt && git add f.txt",
            )
            .await
            .unwrap();
        assert_eq!(setup.exit_code, 0, "setup failed: {}", setup.combined);

        let r = exec.bash("git commit").await.unwrap();
        assert!(
            !r.timed_out,
            "git commit without -m hung waiting for an editor"
        );
    }

    // Windows-only: the whole point of the win32 bash resolver is to run
    // commands through a real Git Bash / MSYS2 bash rather than System32's
    // WSL launcher. These prove the resolver picks a real bash and that it
    // actually executes win32-side (a /c/... MSYS path, never /mnt/c WSL).
    #[cfg(windows)]
    #[test]
    fn win32_bash_launcher_dir_classification() {
        assert!(is_windows_bash_launcher_dir(Path::new(
            r"C:\Windows\System32"
        )));
        assert!(is_windows_bash_launcher_dir(Path::new(
            r"C:\Users\x\AppData\Local\Microsoft\WindowsApps"
        )));
        assert!(!is_windows_bash_launcher_dir(Path::new(
            r"C:\Program Files\Git\bin"
        )));
    }

    #[cfg(windows)]
    #[test]
    fn win32_bash_resolves_and_is_not_wsl_launcher() {
        // GitHub's windows runners (and any Git-for-Windows host) ship a real
        // bash, so resolution must succeed and must not be the WSL launcher.
        let bash = bash_program().expect("win32 bash should resolve on a Git-for-Windows host");
        assert!(
            bash.is_file(),
            "resolved bash must exist: {}",
            bash.display()
        );
        assert!(
            !is_windows_bash_launcher_dir(bash.parent().unwrap()),
            "resolver must skip the WSL launcher, got: {}",
            bash.display()
        );
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn win32_bash_runs_win_side_not_wsl() {
        // End-to-end through the resolved bash: given a Windows cwd, `pwd`
        // reports a win32 MSYS path (`/c/...`). WSL's launcher would report
        // a `/mnt/c/...` path — its absence proves we're not on WSL.
        let exec = HostExecutor::new(std::env::temp_dir());
        let r = exec.bash("pwd").await.unwrap();
        assert_eq!(r.exit_code, 0, "bash did not run cleanly: {}", r.combined);
        assert!(
            !r.combined.contains("/mnt/"),
            "resolved bash ran under WSL, not win32: {}",
            r.combined
        );
    }
}
