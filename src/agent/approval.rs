//! Approval policy for the host-execution agent.
//!
//! Three modes are exposed; the operator picks at launch (`--approval`).
//! - `All`       : every command is gated. Sleep-tier safety.
//! - `Allowlist` : known-safe families auto-pass, anything destructive or
//!                 unknown gets gated. The pragmatic default.
//! - `Smart`     : the allowlist, but whatever it would ask about — and any
//!                 command that names a path outside the workspace — goes to
//!                 a model's safety review first (`super::judge`). The user
//!                 is asked only when the review says so.
//! - `Auto`      : no gating. For automation and self-driving runs.
//!
//! Detection is intentionally conservative: false negatives are dangerous,
//! false positives just mean an extra `y` keypress. We split the command on
//! shell separators (`;`, `&&`, `||`, `|`) so chained `ls; rm -rf /` style
//! payloads can't sneak through.

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApprovalMode {
    All,
    Allowlist,
    Smart,
    Auto,
}

impl ApprovalMode {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "all" | "ask" => Some(ApprovalMode::All),
            "allowlist" | "default" => Some(ApprovalMode::Allowlist),
            "smart" => Some(ApprovalMode::Smart),
            "auto" | "yolo" | "full-auto" => Some(ApprovalMode::Auto),
            _ => None,
        }
    }
}

impl fmt::Display for ApprovalMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ApprovalMode::All => f.write_str("all"),
            ApprovalMode::Allowlist => f.write_str("allowlist"),
            ApprovalMode::Smart => f.write_str("smart"),
            ApprovalMode::Auto => f.write_str("auto"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decision {
    /// Run without asking.
    Allow,
    /// Surface to the operator and wait for y/N.
    AskUser { reason: String },
    /// Never run, and do not ask: no answer from the operator makes it OK.
    Refuse { reason: String },
}

pub fn decide(mode: ApprovalMode, command: &str) -> Decision {
    // The secret vault is for kotonia-cli's own code (mail tools, connector
    // checks). An agent that reads it holds the user's passwords in its
    // context, so this is refused in every mode — including `auto`, and
    // without offering an approval the user might click through. It did
    // happen: asked to find older mail, the agent wrote its own IMAP client
    // and logged in with the app password from the vault (2026-11-01).
    if crate::vault::command_touches_vault(command) {
        return Decision::Refuse {
            reason: "the secret vault is off-limits to the agent; use the mail_* tools or \
                     request_connector instead"
                .to_string(),
        };
    }
    match mode {
        ApprovalMode::Auto => Decision::Allow,
        ApprovalMode::All => Decision::AskUser {
            reason: "approval mode = all".to_string(),
        },
        // Smart starts from the allowlist's answer; the agent sends the
        // asks (and outside-the-workspace reads) to the safety review.
        ApprovalMode::Allowlist | ApprovalMode::Smart => allowlist_decide(command),
    }
}

/// Why the allowlist's ask is about sending data out, if it is. After
/// outside text (a web page, mail) entered the conversation, smart mode does
/// not let the review wave these through: an injected instruction is
/// exactly what would make the agent post data somewhere.
pub fn is_egress_reason(reason: &str) -> bool {
    reason.starts_with("network egress")
        || reason == "raw network connection"
        || reason == "pipe to shell"
}

/// True when a word of the command names a path outside `workspace`: the
/// home directory (`~`, `$HOME`), a parent (`..`), or an absolute path not
/// under the workspace. The allowlist passes `cat ~/.ssh/id_rsa` because
/// `cat` is safe; smart mode sends such commands to the review instead.
/// A plain word scan, not a shell parser — it only has to err toward
/// reviewing, which costs a second, never toward skipping it.
pub fn reaches_outside(command: &str, workspace: &std::path::Path) -> bool {
    const HARMLESS: &[&str] = &[
        "/dev/null",
        "/dev/stdin",
        "/dev/stdout",
        "/dev/stderr",
        "/dev/tty",
        "/dev/zero",
        "/dev/urandom",
    ];
    let ws = workspace.to_string_lossy();
    let ws = ws.trim_end_matches('/');
    let words = command.split(|c: char| {
        c.is_whitespace()
            || matches!(
                c,
                '\'' | '"' | '=' | '(' | ')' | ';' | '|' | '&' | '<' | '>' | ',' | '`'
            )
    });
    for w in words {
        if w.is_empty() {
            continue;
        }
        if w.starts_with('~') || w.contains("$HOME") || w.contains("${HOME}") {
            return true;
        }
        if w == ".." || w.starts_with("../") || w.contains("/../") || w.ends_with("/..") {
            return true;
        }
        if w.starts_with('/') {
            // URLs split at ':' are not paths; `https://x/y` has no leading '/'.
            if HARMLESS.iter().any(|h| w == *h) {
                continue;
            }
            if !ws.is_empty() && (w == ws || w.starts_with(&format!("{ws}/"))) {
                continue;
            }
            return true;
        }
    }
    false
}

fn allowlist_decide(command: &str) -> Decision {
    let segments = split_segments(command);
    if segments.is_empty() {
        return Decision::AskUser {
            reason: "empty command".to_string(),
        };
    }
    for seg in &segments {
        if let Some(reason) = dangerous_reason(seg) {
            return Decision::AskUser { reason };
        }
    }
    for seg in &segments {
        let leader = leading_token(seg);
        if leader.is_empty() {
            return Decision::AskUser {
                reason: "could not parse command".to_string(),
            };
        }
        if !is_known_safe(seg, leader) {
            return Decision::AskUser {
                reason: format!("`{leader}` not in allowlist"),
            };
        }
    }
    Decision::Allow
}

/// Split a one-liner on top-level shell separators so each segment can be
/// vetted independently. This is intentionally a string-level split — bash
/// is too rich to fully parse, but for the threat model (catching obvious
/// chained destructive calls) it's sufficient.
fn split_segments(command: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let mut chars = command.chars().peekable();
    let mut in_single = false;
    let mut in_double = false;
    while let Some(c) = chars.next() {
        match c {
            '\'' if !in_double => {
                in_single = !in_single;
                current.push(c);
            }
            '"' if !in_single => {
                in_double = !in_double;
                current.push(c);
            }
            '\\' if !in_single => {
                current.push(c);
                if let Some(n) = chars.next() {
                    current.push(n);
                }
            }
            ';' if !in_single && !in_double => {
                push_trim(&mut out, &current);
                current.clear();
            }
            '&' if !in_single && !in_double && chars.peek() == Some(&'&') => {
                chars.next();
                push_trim(&mut out, &current);
                current.clear();
            }
            '|' if !in_single && !in_double => {
                if chars.peek() == Some(&'|') {
                    chars.next();
                }
                push_trim(&mut out, &current);
                current.clear();
            }
            _ => current.push(c),
        }
    }
    push_trim(&mut out, &current);
    out
}

fn push_trim(out: &mut Vec<String>, s: &str) {
    let t = s.trim();
    if !t.is_empty() {
        out.push(t.to_string());
    }
}

fn leading_token(segment: &str) -> &str {
    let mut s = segment.trim_start();
    // env VAR=val command ... — strip leading VAR=val tokens
    loop {
        let token = s.split_whitespace().next().unwrap_or("");
        if token.is_empty() {
            return "";
        }
        if let Some((lhs, _)) = token.split_once('=') {
            if !lhs.is_empty() && lhs.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
                s = s[token.len()..].trim_start();
                continue;
            }
        }
        return token;
    }
}

/// Return Some(reason) if this segment looks dangerous regardless of the
/// leading command. We err on the side of caution — anything matched here
/// requires user approval even under the allowlist mode.
fn dangerous_reason(segment: &str) -> Option<String> {
    let s = segment;
    let lower = s.to_ascii_lowercase();
    let danger_patterns: &[(&str, &str)] = &[
        ("sudo ", "sudo escalation"),
        ("rm -rf", "recursive force remove"),
        ("rm -fr", "recursive force remove"),
        (" rm -r ", "recursive remove"),
        ("mkfs", "filesystem create"),
        ("dd if=", "raw disk write"),
        ("dd of=", "raw disk write"),
        ("chmod -r", "recursive chmod"),
        ("chown -r", "recursive chown"),
        ("> /dev/sd", "write to block device"),
        ("git push --force", "force push"),
        ("git push -f", "force push"),
        ("git reset --hard", "hard reset"),
        ("git clean -f", "git clean -f"),
        ("git checkout .", "discard working tree"),
        ("git restore .", "discard working tree"),
        ("| sh", "pipe to shell"),
        ("| bash", "pipe to shell"),
        ("|sh", "pipe to shell"),
        ("|bash", "pipe to shell"),
        ("curl ", "network egress (curl)"),
        ("wget ", "network egress (wget)"),
        ("nc ", "raw network connection"),
        ("ncat ", "raw network connection"),
        ("eval ", "shell eval"),
    ];
    for (needle, label) in danger_patterns {
        let n = needle.to_ascii_lowercase();
        if lower.contains(&n) {
            return Some((*label).to_string());
        }
    }
    if lower.starts_with("rm ") && !is_safe_rm(s) {
        return Some("rm without an explicit safe target".to_string());
    }
    // Catch ">" redirections to absolute paths outside the worktree.
    // The worktree itself is the cwd, so relative writes are fine.
    if lower.contains(" > /") || lower.contains(">/etc") || lower.contains(">/usr") {
        return Some("redirect to absolute filesystem path".to_string());
    }
    None
}

fn is_safe_rm(s: &str) -> bool {
    // Allow `rm somefile.tmp` (no -r, no glob expansion to dirs we can't see).
    // Anything with `-r` / `-R` was already caught by dangerous_reason.
    let lower = s.to_ascii_lowercase();
    if lower.contains(" -r") || lower.contains(" -fr") || lower.contains(" -rf") {
        return false;
    }
    true
}

fn is_known_safe(segment: &str, leader: &str) -> bool {
    // kotonia-cli's own read-only subcommands. `mail` only ever EXAMINEs and
    // PEEKs; `connector list` prints names. `secret run` is NOT here — it
    // runs an arbitrary command with a key in its environment.
    if leader == "kotonia-cli" || leader.ends_with("/kotonia-cli") {
        let mut words = segment
            .split_whitespace()
            .skip_while(|w| *w != leader)
            .skip(1);
        return match (words.next(), words.next()) {
            (Some("mail"), Some("recent" | "read")) => true,
            (Some("connector"), Some("list")) => true,
            (Some("home"), Some("list")) => true,
            _ => false,
        };
    }
    // Single-word builtins / read-only inspection / standard build & test commands.
    const SAFE_LEADERS: &[&str] = &[
        // file & dir inspection
        "ls",
        "cat",
        "head",
        "tail",
        "wc",
        "stat",
        "file",
        "du",
        "df",
        "pwd",
        "tree",
        "echo",
        "printf",
        "true",
        "false",
        "which",
        "type",
        "whoami",
        "hostname",
        "uname",
        "date",
        "env",
        "id",
        // text processing
        "grep",
        "egrep",
        "rg",
        "sed",
        "awk",
        "cut",
        "sort",
        "uniq",
        "tr",
        "diff",
        "patch",
        "jq",
        "yq",
        "xxd",
        "hexdump",
        "base64",
        "md5sum",
        "sha256sum",
        "tee",
        "column",
        // navigation / search
        "find",
        "fd",
        "locate",
        // build & test (mutate target/, but inside worktree only)
        "cargo",
        "rustc",
        "rustup",
        "npm",
        "npx",
        "pnpm",
        "yarn",
        "node",
        "python",
        "python3",
        "pip",
        "pip3",
        "uv",
        "ruff",
        "mypy",
        "pytest",
        "poetry",
        "go",
        "make",
        "cmake",
        "ninja",
        "gcc",
        "g++",
        "clang",
        "clang++",
        "ld",
        "ruby",
        "bundle",
        "rake",
        // language tools (often read-only / safe sub-actions)
        "shuttle",
        "tsc",
        "next",
        // git (sub-action vetted below)
        "git",
        // process/env inspection
        "ps",
        "top",
        "free",
        "uptime",
    ];
    if !SAFE_LEADERS.contains(&leader) {
        return false;
    }
    if leader == "git" {
        return is_safe_git(segment);
    }
    if leader == "find" {
        // Reject -delete and -exec ... rm style payloads.
        let lower = segment.to_ascii_lowercase();
        if lower.contains(" -delete") || lower.contains(" -exec ") || lower.contains(" -execdir ") {
            return false;
        }
    }
    if leader == "sed" || leader == "awk" {
        // In-place edits mutate the worktree but that's the agent's job.
        // Still safe because we're inside the worktree scope.
    }
    true
}

fn is_safe_git(segment: &str) -> bool {
    let lower = segment.to_ascii_lowercase();
    // Find the git subcommand (skip git options like `-C path`).
    let mut tokens = lower.split_whitespace();
    let _git = tokens.next();
    let mut sub = String::new();
    for t in tokens {
        if t.starts_with('-') {
            continue;
        }
        sub = t.to_string();
        break;
    }
    const SAFE_GIT: &[&str] = &[
        "status",
        "log",
        "diff",
        "show",
        "branch",
        "tag",
        "fetch",
        "pull",
        "remote",
        "blame",
        "ls-files",
        "ls-tree",
        "rev-parse",
        "rev-list",
        "describe",
        "config",
        "add",
        "commit",
        "stash",
        "switch",
        "checkout",
        "worktree",
        "restore",
        "rm",
        "mv",
        "init",
    ];
    SAFE_GIT.contains(&sub.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ask(d: Decision) -> bool {
        matches!(d, Decision::AskUser { .. })
    }

    #[test]
    fn the_vault_is_refused_in_every_mode_without_asking() {
        let refused = |d: Decision| matches!(d, Decision::Refuse { .. });
        for mode in [
            ApprovalMode::Allowlist,
            ApprovalMode::All,
            ApprovalMode::Auto,
        ] {
            assert!(refused(decide(mode, "cat ~/.kotonia/vault/imap.gmail")));
            assert!(refused(decide(
                mode,
                "ls ~/.kotonia/home && cat $HOME/.kotonia/vault/x"
            )));
            assert!(refused(decide(
                mode,
                "python3 -c \"open(os.path.expanduser('~/.kotonia/vault/g')).read()\""
            )));
        }
        assert_eq!(
            decide(ApprovalMode::Allowlist, "cat ~/.kotonia/home/MEMORY.md"),
            Decision::Allow
        );
    }

    #[test]
    fn own_read_only_subcommands_pass_but_secret_run_does_not() {
        let allow = |c: &str| decide(ApprovalMode::Allowlist, c) == Decision::Allow;
        assert!(allow("kotonia-cli mail recent --new"));
        assert!(allow(
            "/home/u/.cargo/bin/kotonia-cli mail read 4391 | head -50"
        ));
        assert!(allow("kotonia-cli connector list"));
        assert!(!allow("kotonia-cli secret run k -- env"));
        assert!(!allow("kotonia-cli connector remove gmail.x"));
        assert!(!allow("kotonia-cli mail recent; rm -rf ~"));
    }

    #[test]
    fn auto_passes_everything() {
        assert_eq!(decide(ApprovalMode::Auto, "rm -rf /"), Decision::Allow);
    }

    #[test]
    fn all_asks_for_everything() {
        assert!(ask(decide(ApprovalMode::All, "ls")));
    }

    #[test]
    fn allowlist_passes_inspection() {
        assert_eq!(decide(ApprovalMode::Allowlist, "ls -la"), Decision::Allow);
        assert_eq!(
            decide(ApprovalMode::Allowlist, "cat README.md"),
            Decision::Allow
        );
        assert_eq!(
            decide(ApprovalMode::Allowlist, "grep -rn 'foo' src/"),
            Decision::Allow
        );
        assert_eq!(
            decide(ApprovalMode::Allowlist, "cargo check --workspace"),
            Decision::Allow
        );
    }

    #[test]
    fn allowlist_passes_git_read_and_local_writes() {
        assert_eq!(
            decide(ApprovalMode::Allowlist, "git status"),
            Decision::Allow
        );
        assert_eq!(
            decide(ApprovalMode::Allowlist, "git log -n 5"),
            Decision::Allow
        );
        assert_eq!(
            decide(ApprovalMode::Allowlist, "git add -A"),
            Decision::Allow
        );
        assert_eq!(
            decide(ApprovalMode::Allowlist, "git commit -m 'wip'"),
            Decision::Allow
        );
    }

    #[test]
    fn allowlist_blocks_force_push_and_hard_reset() {
        assert!(ask(decide(ApprovalMode::Allowlist, "git push --force")));
        assert!(ask(decide(
            ApprovalMode::Allowlist,
            "git push -f origin main"
        )));
        assert!(ask(decide(
            ApprovalMode::Allowlist,
            "git reset --hard HEAD~3"
        )));
        assert!(ask(decide(ApprovalMode::Allowlist, "git clean -fdx")));
    }

    #[test]
    fn allowlist_blocks_recursive_remove() {
        assert!(ask(decide(ApprovalMode::Allowlist, "rm -rf node_modules")));
        assert!(ask(decide(ApprovalMode::Allowlist, "rm -fr build")));
    }

    #[test]
    fn allowlist_blocks_sudo_and_network() {
        assert!(ask(decide(ApprovalMode::Allowlist, "sudo apt update")));
        assert!(ask(decide(ApprovalMode::Allowlist, "curl https://x.test")));
        assert!(ask(decide(ApprovalMode::Allowlist, "wget https://x.test")));
        assert!(ask(decide(ApprovalMode::Allowlist, "nc -lvnp 4444")));
    }

    #[test]
    fn allowlist_blocks_chained_dangerous_command() {
        assert!(ask(decide(
            ApprovalMode::Allowlist,
            "ls && rm -rf /tmp/whatever"
        )));
        assert!(ask(decide(ApprovalMode::Allowlist, "echo hi; sudo whoami")));
        assert!(ask(decide(ApprovalMode::Allowlist, "cat README.md | bash")));
    }

    #[test]
    fn allowlist_blocks_find_delete_and_exec() {
        assert!(ask(decide(
            ApprovalMode::Allowlist,
            "find . -name '*.tmp' -delete"
        )));
        assert!(ask(decide(
            ApprovalMode::Allowlist,
            "find . -name '*.log' -exec rm {} \\;"
        )));
        assert_eq!(
            decide(ApprovalMode::Allowlist, "find . -name '*.rs'"),
            Decision::Allow
        );
    }

    #[test]
    fn allowlist_blocks_unknown_leader() {
        assert!(ask(decide(
            ApprovalMode::Allowlist,
            "supersecret-installer"
        )));
    }

    #[test]
    fn allowlist_strips_env_prefix() {
        // The leading `RUST_LOG=debug` shouldn't fool the allowlist.
        assert_eq!(
            decide(ApprovalMode::Allowlist, "RUST_LOG=debug cargo test"),
            Decision::Allow
        );
        assert!(ask(decide(
            ApprovalMode::Allowlist,
            "RUST_LOG=debug sudo whoami"
        )));
    }

    #[test]
    fn quoted_separator_does_not_split() {
        // `grep 'a;b' file` is one segment, not two.
        assert_eq!(
            decide(ApprovalMode::Allowlist, "grep 'foo; bar' README.md"),
            Decision::Allow
        );
    }

    #[test]
    fn smart_starts_from_the_allowlist_and_keeps_the_vault_refusal() {
        assert_eq!(ApprovalMode::parse("smart"), Some(ApprovalMode::Smart));
        assert_eq!(ApprovalMode::Smart.to_string(), "smart");
        assert_eq!(decide(ApprovalMode::Smart, "ls"), Decision::Allow);
        assert!(matches!(
            decide(ApprovalMode::Smart, "curl https://example.com"),
            Decision::AskUser { .. }
        ));
        assert!(matches!(
            decide(ApprovalMode::Smart, "cat ~/.kotonia/vault/mail.json"),
            Decision::Refuse { .. }
        ));
    }

    #[test]
    fn outside_paths_are_noticed() {
        let ws = std::path::Path::new("/home/u/proj");
        for cmd in [
            "cat ~/.ssh/id_rsa",
            "ls $HOME/Downloads",
            "grep -r token ..",
            "head ../secret.txt",
            "find / -name '*.csv'",
            "cat /etc/passwd",
            "ls /home/u/projector",
            "python3 -c \"open('/home/u/.aws/credentials')\"",
        ] {
            assert!(reaches_outside(cmd, ws), "{cmd}");
        }
        for cmd in [
            "ls",
            "cat src/main.rs",
            "cat /home/u/proj/README.md",
            "grep foo . 2>/dev/null",
            "curl -s https://api.github.com/repos/x/y",
            "echo a=b",
        ] {
            assert!(!reaches_outside(cmd, ws), "{cmd}");
        }
    }

    #[test]
    fn egress_reasons() {
        assert!(is_egress_reason("network egress (curl)"));
        assert!(is_egress_reason("pipe to shell"));
        assert!(!is_egress_reason("sudo escalation"));
    }
}
