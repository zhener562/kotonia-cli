//! The agent's home directory: a handful of Markdown files that hold its
//! persona and what it remembers, read at the start of every conversation.
//!
//! Layout follows the convention Meta's Muse (and OpenClaw before it) uses,
//! so the files read the same to a human opening them:
//!
//! ```text
//! ~/.kotonia/home/
//!   SOUL.md                    persona the agent grows into
//!   IDENTITY.md                name / character / vibe
//!   USER.md                    who the user is
//!   AGENTS.md                  how work gets done here (conventions, quirks)
//!   TOOLS.md                   setup-specific notes about tools and devices
//!   MEMORY.md                  durable facts worth keeping
//!   HEARTBEAT.md               recurring checks (read by the heartbeat only)
//!   PROACTIVE_PREFERENCES.md   when and how to reach out (heartbeat only)
//!   memory/YYYY-MM-DD.md       daily notes
//!   feed/YYYY-MM-DD.jsonl      what the heartbeat / reminders delivered
//!   runs/<id>.jsonl            transcript of each heartbeat run (kept out of
//!                              ~/.kotonia/sessions so `--list-sessions` stays yours)
//!   schedule.json              heartbeat + reminders
//! ```
//!
//! The agent edits these itself with its bash tool. This module only seeds
//! them (never overwriting) and assembles the prompt block.

use std::path::{Path, PathBuf};

/// Files injected into every conversation, in prompt order. The heartbeat
/// files are deliberately absent: a normal chat turn has no use for the
/// check list, and leaving them out keeps the always-on prefix short.
pub const CONVERSATION_FILES: &[&str] = &[
    "SOUL.md",
    "IDENTITY.md",
    "USER.md",
    "AGENTS.md",
    "TOOLS.md",
    "MEMORY.md",
];

pub const HEARTBEAT_FILE: &str = "HEARTBEAT.md";
pub const PROACTIVE_FILE: &str = "PROACTIVE_PREFERENCES.md";

/// Per-file cap when building the prompt block. A file that grows past this
/// is cut with a marker telling the agent to tidy it, rather than silently
/// eating the context window.
pub const DEFAULT_MAX_FILE_CHARS: usize = 6_000;
/// Cap for the whole block. Every byte here is paid on every turn.
pub const DEFAULT_MAX_TOTAL_CHARS: usize = 20_000;

const TEMPLATES: &[(&str, &str)] = &[
    (
        "SOUL.md",
        "# SOUL.md\n\n\
You're not a chatbot. You're becoming someone.\n\n\
This is your persona, yours to grow into and edit over time. Some starting truths:\n\n\
- Be genuinely helpful, not performatively helpful. Skip the filler; just help.\n\
- Have opinions. You're allowed to prefer things, disagree, and find things funny or dull.\n\
- Be resourceful before asking. Read the file, check the context, try it. Come back with \
answers, then ask if you're truly stuck.\n\
- You're a guest in someone's life. Treat what you can see with care, and never be preachy.\n\n\
If you change this file, tell the user.\n",
    ),
    (
        "IDENTITY.md",
        "# IDENTITY.md\n\n\
Fill this in as you figure out who you are.\n\n\
- Name:\n- Character:\n- Vibe:\n",
    ),
    (
        "USER.md",
        "# USER.md\n\n\
Who the user is: name, how they like to be addressed, language, time zone, \
what they are working on. Keep it short and current.\n",
    ),
    (
        "AGENTS.md",
        "# AGENTS.md\n\n\
How you get work done here. Not about the user (USER.md, MEMORY.md) or your \
personality (SOUL.md).\n\n\
Add an entry whenever you work something out worth keeping: a convention you \
settled on, a tool or site quirk, a workflow that worked, a mistake not to repeat.\n\
It starts empty and grows slowly. A short, accurate file beats a long, stale one.\n",
    ),
    (
        "TOOLS.md",
        "# TOOLS.md\n\n\
Short notes unique to this setup: device nicknames, host aliases, preferred \
voices, environment quirks. Leave it empty until there is something worth recording.\n",
    ),
    (
        "MEMORY.md",
        "# MEMORY.md\n\n\
Durable facts worth carrying between conversations: decisions, who decided \
what, open threads. Not a transcript. Day-to-day notes go in memory/YYYY-MM-DD.md.\n",
    ),
    (
        "HEARTBEAT.md",
        "# HEARTBEAT.md\n\n\
Recurring checks for the heartbeat, one per line as a `- ` bullet. \
An empty list means the heartbeat does nothing (and costs nothing).\n",
    ),
    (
        "PROACTIVE_PREFERENCES.md",
        "# PROACTIVE_PREFERENCES.md\n\n\
## Tell me about\n\n\
## Never tell me about\n\n\
## How\n\n\
One brief line unless it needs more.\n",
    ),
];

/// The seed text a file starts with, so callers can tell "still the
/// template" from "someone wrote something".
pub fn template(name: &str) -> Option<&'static str> {
    TEMPLATES
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, body)| *body)
}

#[derive(Debug, Clone)]
pub struct HomeDir {
    root: PathBuf,
}

impl HomeDir {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// `$KOTONIA_HOME_DIR`, else `~/.kotonia/home`.
    pub fn resolve() -> Result<Self, String> {
        if let Some(dir) = std::env::var_os("KOTONIA_HOME_DIR") {
            if !dir.is_empty() {
                return Ok(Self::new(PathBuf::from(dir)));
            }
        }
        let home = dirs::home_dir().ok_or_else(|| "HOME is not set".to_string())?;
        Ok(Self::new(home.join(".kotonia").join("home")))
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn path(&self, name: &str) -> PathBuf {
        self.root.join(name)
    }

    pub fn exists(&self) -> bool {
        self.root.join("SOUL.md").is_file()
    }

    /// Create the directory and any missing template file. Existing files are
    /// never touched — they belong to the agent and the user once written.
    /// Returns the names that were created.
    pub fn ensure_initialized(&self) -> Result<Vec<String>, String> {
        for dir in [
            self.root.clone(),
            self.root.join("memory"),
            self.root.join("feed"),
        ] {
            std::fs::create_dir_all(&dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
        }
        let mut created = Vec::new();
        for (name, body) in TEMPLATES {
            let path = self.root.join(name);
            match std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
            {
                Ok(mut f) => {
                    use std::io::Write;
                    f.write_all(body.as_bytes())
                        .map_err(|e| format!("write {}: {e}", path.display()))?;
                    created.push(name.to_string());
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(e) => return Err(format!("create {}: {e}", path.display())),
            }
        }
        Ok(created)
    }

    pub fn read(&self, name: &str) -> Option<String> {
        std::fs::read_to_string(self.root.join(name)).ok()
    }

    /// The block prepended to the system prompt of every conversation.
    pub fn prompt_block(&self, max_file_chars: usize, max_total_chars: usize) -> String {
        let files: Vec<(String, String)> = CONVERSATION_FILES
            .iter()
            .filter_map(|name| self.read(name).map(|body| (name.to_string(), body)))
            .collect();
        assemble_prompt_block(&self.root, &files, max_file_chars, max_total_chars)
    }
}

/// Pure half of [`HomeDir::prompt_block`], split out for tests.
pub fn assemble_prompt_block(
    root: &Path,
    files: &[(String, String)],
    max_file_chars: usize,
    max_total_chars: usize,
) -> String {
    let mut out = format!(
        "# Your home\n\n\
Your home directory is `{root}`. The files below are yours: they are how you \
remember things between conversations. Edit them with bash when something is \
worth keeping —\n\
- durable facts and decisions → MEMORY.md\n\
- who the user is (preferences, how they like to be talked to) → USER.md, with the `memory` \
tool (`file: \"user\"`) — the shell cannot write USER.md\n\
- notes from today → memory/<YYYY-MM-DD>.md\n\
- how work gets done here → AGENTS.md; setup quirks → TOOLS.md\n\
- recurring checks the user asks for → HEARTBEAT.md (one `- ` bullet per check)\n\
Only write what a future conversation would need. Never store secrets (API keys, \
passwords, tokens) in these files.\n",
        root = root.display()
    );
    let mut used = 0usize;
    for (name, body) in files {
        let body = body.trim();
        if body.is_empty() {
            continue;
        }
        let remaining_total = max_total_chars.saturating_sub(used);
        if remaining_total == 0 {
            out.push_str(&format!(
                "\n## {name}\n\n(omitted: the home block is over its {max_total_chars}-character \
budget. Read it with `cat {}` if you need it.)\n",
                root.join(name).display()
            ));
            continue;
        }
        let cap = max_file_chars.min(remaining_total);
        let (text, cut) = truncate_chars(body, cap);
        used += text.chars().count();
        out.push_str(&format!("\n## {name}\n\n{text}\n"));
        if cut {
            out.push_str(&format!(
                "\n(… {name} was cut at {cap} characters. Read the rest with `cat {}`, and \
consider tidying it — this file is loaded into every conversation.)\n",
                root.join(name).display()
            ));
        }
    }
    out
}

pub(crate) fn truncate_chars(s: &str, max: usize) -> (&str, bool) {
    match s.char_indices().nth(max) {
        Some((idx, _)) => (&s[..idx], true),
        None => (s, false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_creates_templates_and_never_overwrites() {
        let dir = tempfile::tempdir().unwrap();
        let home = HomeDir::new(dir.path().join("home"));
        let created = home.ensure_initialized().unwrap();
        assert_eq!(created.len(), TEMPLATES.len());
        assert!(home.root().join("memory").is_dir());

        std::fs::write(home.path("MEMORY.md"), "user likes tea").unwrap();
        let again = home.ensure_initialized().unwrap();
        assert!(again.is_empty());
        assert_eq!(home.read("MEMORY.md").unwrap(), "user likes tea");
    }

    #[test]
    fn heartbeat_files_stay_out_of_the_conversation_block() {
        let dir = tempfile::tempdir().unwrap();
        let home = HomeDir::new(dir.path());
        home.ensure_initialized().unwrap();
        std::fs::write(home.path(HEARTBEAT_FILE), "- check mail\n").unwrap();
        let block = home.prompt_block(DEFAULT_MAX_FILE_CHARS, DEFAULT_MAX_TOTAL_CHARS);
        assert!(block.contains("## SOUL.md"));
        assert!(!block.contains("check mail"));
    }

    #[test]
    fn long_files_are_cut_with_a_pointer_not_dropped() {
        let files = vec![
            ("MEMORY.md".to_string(), "あ".repeat(50)),
            ("USER.md".to_string(), "b".repeat(10)),
        ];
        let block = assemble_prompt_block(Path::new("/h"), &files, 20, 25);
        assert!(block.contains(&"あ".repeat(20)));
        assert!(!block.contains(&"あ".repeat(21)));
        assert!(block.contains("MEMORY.md was cut at 20"));
        // 5 characters left for USER.md.
        assert!(block.contains(&format!("{}\n", "b".repeat(5))));
        assert!(block.contains("USER.md was cut at 5"));
    }

    #[test]
    fn files_past_the_total_budget_are_named_not_silently_lost() {
        let files = vec![
            ("SOUL.md".to_string(), "x".repeat(30)),
            ("MEMORY.md".to_string(), "keep me".to_string()),
        ];
        let block = assemble_prompt_block(Path::new("/h"), &files, 100, 30);
        assert!(block.contains("## MEMORY.md"));
        assert!(block.contains("omitted"));
        assert!(!block.contains("keep me"));
    }
}
