//! Per-folder memory: notes the agent keeps about one project, plus the
//! user's profile from the home directory, read at the start of every
//! session in that folder.
//!
//! ```text
//! ~/.kotonia/projects/<folder>-<8 hex>/
//!   ROOT                    the folder this belongs to (for humans and UIs)
//!   MEMORY.md               durable facts about this folder
//!   memory/YYYY-MM-DD.md    daily notes
//! ```
//!
//! Wired into `Agent::new`, so the CLI, the daemon and kotonia-desktop all
//! get it without each caller loading files itself.
//!
//! Decisions worth keeping (and the alternatives that were turned down):
//!
//! - **The key is the main worktree's root, not the session's cwd.** The CLI
//!   runs each session in a fresh `/tmp/kotonia-agent-xxxx` worktree; keyed
//!   by cwd, every session would start with an empty memory. Subdirectories
//!   of a repo map to the repo too.
//! - **The files live outside the folder.** A `.kotonia/MEMORY.md` inside
//!   the repo would ride along with `git add .` into a public repository,
//!   carrying whatever the agent learned about the user. Instructions meant
//!   to be shared with a team belong in the repo's own AGENTS.md instead.
//! - **The directory is created on first write.** Opening a folder once
//!   should not leave a directory behind.
//! - **The agent writes through the `memory` tool, never bash.** Bash is
//!   refused for these paths, so every write passes one place that can see
//!   it — and ask the user when the session has read mail or web pages
//!   (text from outside that could otherwise plant a lasting instruction).
//!   An approval prompt on every write was turned down: asked all the time,
//!   people approve without reading.
//! - **Mail is granted per folder, by the user** (`kotonia-cli project
//!   allow-mail`), in `grants.json` next to the memory directories, which
//!   the agent cannot reach.
//! - **USER.md is shared with every folder** (who the user is, how they like
//!   to be talked to). It is a few hundred characters, and a coding session
//!   that knows the user's preferences wastes fewer turns on them.

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::home::workspace::{self, HomeDir};

#[derive(Debug, Clone)]
pub struct ProjectMemory {
    /// The folder the memory belongs to (main worktree root for git repos).
    root: PathBuf,
    /// `~/.kotonia/projects` (or `$KOTONIA_PROJECTS_DIR`).
    base: PathBuf,
    /// Where this folder's files live, under `base`.
    dir: PathBuf,
}

/// Which memory file a `memory` tool call is about.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryFile {
    /// This folder's MEMORY.md.
    Folder,
    /// This folder's memory/<today>.md.
    Daily,
    /// The home USER.md, shared by every folder.
    User,
}

impl MemoryFile {
    pub const NAMES: &'static [&'static str] = &["folder", "daily", "user"];

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "folder" => Some(Self::Folder),
            "daily" => Some(Self::Daily),
            "user" => Some(Self::User),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryEdit {
    Read,
    Append(String),
    /// Replace the one occurrence of `old` with `new` (empty `new` deletes).
    Replace {
        old: String,
        new: String,
    },
    /// Replace the whole file (for tidying).
    Rewrite(String),
}

impl MemoryEdit {
    pub fn writes(&self) -> bool {
        !matches!(self, Self::Read)
    }
}

/// One `memory` call may write at most this much. Memory is loaded into
/// every session; a call that wants to write more is writing a transcript.
pub const MAX_MEMORY_WRITE_CHARS: usize = 4_000;

/// What the user granted a folder, kept in `<base>/grants.json`. It lives
/// under the memory directory on purpose: the agent cannot reach that
/// directory with bash at all, and the `memory` tool only ever opens
/// Markdown files, so nothing the agent reads can grant itself an account.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct Grants {
    #[serde(default)]
    pub folders: std::collections::BTreeMap<String, FolderGrant>,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct FolderGrant {
    /// Connector ids (from `~/.kotonia/connectors.json`) whose mail the
    /// agent may read in this folder.
    #[serde(default)]
    pub mail: Vec<String>,
}

impl ProjectMemory {
    /// Memory for the folder containing `workspace`, under
    /// `$KOTONIA_PROJECTS_DIR`, else `~/.kotonia/projects`.
    pub fn resolve(workspace: &Path) -> Result<Self, String> {
        Ok(Self::with_base(&projects_base()?, workspace))
    }

    pub fn with_base(base: &Path, workspace: &Path) -> Self {
        let root = project_root(workspace);
        let dir = base.join(dir_name(&root));
        Self {
            root,
            base: base.to_path_buf(),
            dir,
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn read(&self, name: &str) -> Option<String> {
        std::fs::read_to_string(self.dir.join(name)).ok()
    }

    /// Record which folder this directory belongs to, once it exists. The
    /// directory name alone is lossy (a slug and a hash).
    pub fn note_root(&self) {
        if self.dir.is_dir() && !self.dir.join("ROOT").exists() {
            let _ = std::fs::write(self.dir.join("ROOT"), format!("{}\n", self.root.display()));
        }
    }

    /// The block appended to the system prompt. `home` supplies USER.md;
    /// pass `None` when the caller already put the home files in front
    /// (the resident agent), so the profile is not sent twice. `writable`
    /// is false when the `memory` tool is not offered (the delimiter loop).
    pub fn prompt_block(&self, home: Option<&HomeDir>, writable: bool) -> String {
        let mut out = format!(
            "# This folder's memory\n\n\
You are working in `{root}`. You keep notes about it between sessions; they are stored \
outside the folder, so they are never committed.\n",
            root = self.root.display(),
        );
        if writable {
            out.push_str(
                "Keep them with the `memory` tool (not bash — the shell cannot reach them) when \
something is worth carrying into a later session:\n\
- `folder` — durable facts about this folder: decisions, conventions settled with the user, \
open threads\n\
- `daily` — notes from today\n",
            );
            if home.is_some() {
                out.push_str(
                    "- `user` — what holds about the user everywhere: their preferences, how they \
like to be talked to, how they like work done (shared by every folder)\n",
                );
            }
            out.push_str(
                "Only write what a future session would need; not a transcript. Never store \
secrets (API keys, passwords, tokens). If a note turns out to be wrong, fix or delete it \
(`replace` with an empty `text` deletes).\n",
            );
        } else {
            out.push_str("In this mode the notes are read-only.\n");
        }

        if let Some(home) = home {
            let profile = home.read("USER.md").filter(|body| {
                let body = body.trim();
                !body.is_empty() && Some(body) != workspace::template("USER.md").map(str::trim)
            });
            if let Some(body) = profile {
                push_file(&mut out, "About the user (USER.md)", &body, "user");
            }
        }
        match self.read("MEMORY.md").filter(|b| !b.trim().is_empty()) {
            Some(body) => push_file(&mut out, "This folder (MEMORY.md)", &body, "folder"),
            None => out.push_str("\n## This folder (MEMORY.md)\n\n(nothing yet)\n"),
        }
        out
    }

    fn path_for(
        &self,
        file: MemoryFile,
        home: Option<&HomeDir>,
        today: &str,
    ) -> Result<PathBuf, String> {
        match file {
            MemoryFile::Folder => Ok(self.dir.join("MEMORY.md")),
            MemoryFile::Daily => Ok(self.dir.join("memory").join(format!("{today}.md"))),
            MemoryFile::User => home
                .map(|h| h.path("USER.md"))
                .ok_or_else(|| "the user profile is not available here".to_string()),
        }
    }

    /// Run one `memory` tool call. `today` is `YYYY-MM-DD` (passed in so
    /// tests do not depend on the clock). Returns the text for the model.
    pub fn apply(
        &self,
        home: Option<&HomeDir>,
        file: MemoryFile,
        edit: &MemoryEdit,
        today: &str,
    ) -> Result<String, String> {
        let path = self.path_for(file, home, today)?;
        let current = match std::fs::read_to_string(&path) {
            Ok(s) => Some(s),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => return Err(format!("read {}: {e}", path.display())),
        };
        let written = match edit {
            MemoryEdit::Read => {
                return Ok(match current {
                    Some(s) if !s.trim().is_empty() => s,
                    _ => "(empty)".to_string(),
                })
            }
            MemoryEdit::Append(text) => {
                check_size(text)?;
                let mut body = current.unwrap_or_default();
                if !body.is_empty() && !body.ends_with('\n') {
                    body.push('\n');
                }
                body.push_str(text.trim_end());
                body.push('\n');
                body
            }
            MemoryEdit::Replace { old, new } => {
                check_size(new)?;
                let body = current.unwrap_or_default();
                if old.is_empty() {
                    return Err("`old` is empty; say which text to replace".into());
                }
                match body.matches(old.as_str()).count() {
                    0 => return Err("`old` was not found; read the file first".into()),
                    1 => {}
                    n => {
                        return Err(format!(
                            "`old` occurs {n} times; include more surrounding text"
                        ))
                    }
                }
                let mut out = body.replacen(old.as_str(), new, 1);
                if new.is_empty() {
                    // Deleting a whole line should not leave a blank one.
                    out = out.replace("\n\n\n", "\n\n");
                }
                out
            }
            MemoryEdit::Rewrite(text) => {
                check_size(text)?;
                format!("{}\n", text.trim_end())
            }
        };
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("create {}: {e}", parent.display()))?;
        }
        let tmp = path.with_extension("md.tmp");
        std::fs::write(&tmp, &written).map_err(|e| format!("write {}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, &path).map_err(|e| format!("rename {}: {e}", path.display()))?;
        self.note_root();
        Ok(format!(
            "saved ({} characters in the file now)",
            written.chars().count()
        ))
    }

    fn grants_path(&self) -> PathBuf {
        self.base.join("grants.json")
    }

    pub fn load_grants(&self) -> Result<Grants, String> {
        let path = self.grants_path();
        match std::fs::read_to_string(&path) {
            Ok(raw) => serde_json::from_str(&raw).map_err(|e| format!("{}: {e}", path.display())),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Grants::default()),
            Err(e) => Err(format!("read {}: {e}", path.display())),
        }
    }

    fn key(&self) -> String {
        self.root.to_string_lossy().into_owned()
    }

    /// Mail accounts the user allowed for this folder.
    pub fn mail_grants(&self) -> Vec<String> {
        self.load_grants()
            .ok()
            .and_then(|g| g.folders.get(&self.key()).map(|f| f.mail.clone()))
            .unwrap_or_default()
    }

    /// Allow or withdraw one mail account for this folder. Returns whether
    /// anything changed.
    pub fn set_mail_grant(&self, connector: &str, allow: bool) -> Result<bool, String> {
        let mut grants = self.load_grants()?;
        let key = self.key();
        let entry = grants.folders.entry(key.clone()).or_default();
        let had = entry.mail.iter().any(|m| m == connector);
        let changed = match (allow, had) {
            (true, false) => {
                entry.mail.push(connector.to_string());
                true
            }
            (false, true) => {
                entry.mail.retain(|m| m != connector);
                true
            }
            _ => false,
        };
        if entry.mail.is_empty() {
            grants.folders.remove(&key);
        }
        if changed {
            std::fs::create_dir_all(&self.base)
                .map_err(|e| format!("create {}: {e}", self.base.display()))?;
            let path = self.grants_path();
            let tmp = path.with_extension("json.tmp");
            let json =
                serde_json::to_string_pretty(&grants).map_err(|e| format!("serialize: {e}"))?;
            std::fs::write(&tmp, json).map_err(|e| format!("write {}: {e}", tmp.display()))?;
            std::fs::rename(&tmp, &path).map_err(|e| format!("rename {}: {e}", path.display()))?;
        }
        Ok(changed)
    }
}

/// `$KOTONIA_PROJECTS_DIR`, else `~/.kotonia/projects`.
pub fn projects_base() -> Result<PathBuf, String> {
    match std::env::var_os("KOTONIA_PROJECTS_DIR") {
        Some(dir) if !dir.is_empty() => Ok(PathBuf::from(dir)),
        _ => Ok(dirs::home_dir()
            .ok_or_else(|| "HOME is not set".to_string())?
            .join(".kotonia")
            .join("projects")),
    }
}

/// True when a shell command reaches the memory files, the grants, or the
/// home USER.md. String-level, like the vault check: it stops the honest
/// routes, and the `memory` tool is the easy way. A determined script can
/// still go around it.
pub fn command_touches_memory(command: &str, base: &Path, home: Option<&HomeDir>) -> bool {
    let lower = command.to_ascii_lowercase();
    let base = base.to_string_lossy().to_ascii_lowercase();
    if lower.contains(".kotonia/projects")
        || lower.contains("kotonia_projects_dir")
        || (!base.is_empty() && lower.contains(&base))
    {
        return true;
    }
    let in_home = lower.contains(".kotonia/home")
        || lower.contains("kotonia_home_dir")
        || home.is_some_and(|h| {
            let root = h.root().to_string_lossy().to_ascii_lowercase();
            !root.is_empty() && lower.contains(&root)
        });
    in_home && lower.contains("user.md")
}

fn check_size(text: &str) -> Result<(), String> {
    let n = text.chars().count();
    if n > MAX_MEMORY_WRITE_CHARS {
        return Err(format!(
            "{n} characters is too much for one note (limit {MAX_MEMORY_WRITE_CHARS}); keep \
memory short — it is loaded into every session"
        ));
    }
    Ok(())
}

fn push_file(out: &mut String, title: &str, body: &str, file: &str) {
    let cap = workspace::DEFAULT_MAX_FILE_CHARS;
    let (text, cut) = workspace::truncate_chars(body.trim(), cap);
    out.push_str(&format!("\n## {title}\n\n{text}\n"));
    if cut {
        out.push_str(&format!(
            "\n(… cut at {cap} characters. Read the rest with the `memory` tool (`read`, \
`{file}`), and consider tidying it — it is loaded into every session.)\n"
        ));
    }
}

/// The folder a workspace belongs to: the main worktree's root inside a git
/// repository (so linked worktrees and subdirectories share it), else the
/// directory itself.
pub fn project_root(workspace: &Path) -> PathBuf {
    let out = Command::new("git")
        .arg("-C")
        .arg(workspace)
        .args(["rev-parse", "--path-format=absolute", "--git-common-dir"])
        .env_remove("GIT_DIR")
        .env_remove("GIT_WORK_TREE")
        .output();
    if let Ok(out) = out {
        let common = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if out.status.success() && !common.is_empty() {
            let common = PathBuf::from(common);
            // `<repo>/.git` for a normal repo; a bare repo is its own root.
            let root = match common.file_name() {
                Some(name) if name == ".git" => common.parent().map(Path::to_path_buf),
                _ => None,
            }
            .unwrap_or(common);
            return canonical(&root);
        }
    }
    canonical(workspace)
}

fn canonical(p: &Path) -> PathBuf {
    std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf())
}

/// `<folder name>-<8 hex of the full path>`: readable, and two folders with
/// the same name do not collide. UUID v5 so the hash never changes between
/// builds (std's hasher makes no such promise).
fn dir_name(root: &Path) -> String {
    let full = root.to_string_lossy();
    let hash = uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_URL, full.as_bytes()).simple();
    let base: String = root
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '-'
            }
        })
        .take(40)
        .collect();
    let base = base.trim_matches('-');
    let base = if base.is_empty() { "root" } else { base };
    format!("{base}-{}", &hash.to_string()[..8])
}

/// True when `workspace` is the home directory or inside it — the resident
/// agent's own runs, which already carry the home files.
pub fn is_inside_home(workspace: &Path, home: &HomeDir) -> bool {
    canonical(workspace).starts_with(canonical(home.root()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn git(dir: &Path, args: &[&str]) {
        let ok = Command::new("git")
            .arg("-C")
            .arg(dir)
            .args(["-c", "user.name=t", "-c", "user.email=t@example.com"])
            .args(args)
            .env_remove("GIT_DIR")
            .env_remove("GIT_WORK_TREE")
            .status()
            .unwrap()
            .success();
        assert!(ok, "git {args:?}");
    }

    fn repo_with_worktree() -> (tempfile::TempDir, PathBuf, PathBuf) {
        let tmp = tempfile::tempdir().unwrap();
        let repo = tmp.path().join("my repo");
        std::fs::create_dir_all(repo.join("src/deep")).unwrap();
        git(&repo, &["init", "-q"]);
        git(&repo, &["commit", "-q", "--allow-empty", "-m", "init"]);
        let wt = tmp.path().join("wt");
        git(
            &repo,
            &["worktree", "add", "-q", "-b", "side", wt.to_str().unwrap()],
        );
        (tmp, repo, wt)
    }

    #[test]
    fn worktrees_and_subdirectories_share_the_main_root() {
        let (_tmp, repo, wt) = repo_with_worktree();
        let main = canonical(&repo);
        assert_eq!(project_root(&repo), main);
        assert_eq!(project_root(&repo.join("src/deep")), main);
        assert_eq!(project_root(&wt), main);

        let base = Path::new("/nonexistent/projects");
        let a = ProjectMemory::with_base(base, &wt);
        let b = ProjectMemory::with_base(base, &repo.join("src"));
        assert_eq!(a.dir(), b.dir());
        let name = a.dir().file_name().unwrap().to_string_lossy().into_owned();
        assert!(name.starts_with("my-repo-"), "{name}");
    }

    #[test]
    fn a_plain_folder_is_its_own_root() {
        let tmp = tempfile::tempdir().unwrap();
        let folder = tmp.path().join("おつかい");
        std::fs::create_dir_all(&folder).unwrap();
        assert_eq!(project_root(&folder), canonical(&folder));
        let name = dir_name(&project_root(&folder));
        assert!(name.starts_with("おつかい-"), "{name}");
    }

    #[test]
    fn same_name_different_paths_do_not_collide() {
        assert_ne!(dir_name(Path::new("/a/app")), dir_name(Path::new("/b/app")));
        assert_eq!(dir_name(Path::new("/a/app")), dir_name(Path::new("/a/app")));
    }

    #[test]
    fn one_folders_memory_never_shows_up_in_another() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path().join("projects");
        let (one, two) = (tmp.path().join("one"), tmp.path().join("two"));
        std::fs::create_dir_all(&one).unwrap();
        std::fs::create_dir_all(&two).unwrap();

        let a = ProjectMemory::with_base(&base, &one);
        std::fs::create_dir_all(a.dir()).unwrap();
        std::fs::write(a.dir().join("MEMORY.md"), "secret-of-one").unwrap();

        assert!(a.prompt_block(None, true).contains("secret-of-one"));
        let b = ProjectMemory::with_base(&base, &two);
        let block = b.prompt_block(None, true);
        assert!(!block.contains("secret-of-one"));
        assert!(block.contains("(nothing yet)"));
    }

    #[test]
    fn the_user_profile_is_shared_but_the_template_is_not_sent() {
        let tmp = tempfile::tempdir().unwrap();
        let home = HomeDir::new(tmp.path().join("home"));
        home.ensure_initialized().unwrap();
        let folder = tmp.path().join("f");
        std::fs::create_dir_all(&folder).unwrap();
        let pm = ProjectMemory::with_base(&tmp.path().join("projects"), &folder);

        let block = pm.prompt_block(Some(&home), true);
        assert!(!block.contains("About the user"), "template sent: {block}");
        assert!(block.contains("`user`"), "write target not advertised");

        std::fs::write(home.path("USER.md"), "Prefers short answers in Japanese.").unwrap();
        assert!(pm
            .prompt_block(Some(&home), true)
            .contains("Prefers short answers in Japanese."));
        // The resident agent already has the home files up front.
        assert!(!pm
            .prompt_block(None, true)
            .contains("Prefers short answers"));
    }

    #[test]
    fn root_is_noted_only_once_the_directory_exists() {
        let tmp = tempfile::tempdir().unwrap();
        let folder = tmp.path().join("f");
        std::fs::create_dir_all(&folder).unwrap();
        let pm = ProjectMemory::with_base(&tmp.path().join("projects"), &folder);
        pm.note_root();
        assert!(
            !pm.dir().exists(),
            "opening a folder must not create its memory dir"
        );
        std::fs::create_dir_all(pm.dir()).unwrap();
        pm.note_root();
        assert_eq!(
            std::fs::read_to_string(pm.dir().join("ROOT"))
                .unwrap()
                .trim(),
            canonical(&folder).to_string_lossy()
        );
    }

    #[test]
    fn the_home_directory_is_recognised() {
        let tmp = tempfile::tempdir().unwrap();
        let home = HomeDir::new(tmp.path().join("home"));
        home.ensure_initialized().unwrap();
        assert!(is_inside_home(home.root(), &home));
        assert!(is_inside_home(&home.root().join("memory"), &home));
        assert!(!is_inside_home(tmp.path(), &home));
    }

    fn folder_memory() -> (tempfile::TempDir, ProjectMemory, HomeDir) {
        let tmp = tempfile::tempdir().unwrap();
        let folder = tmp.path().join("f");
        std::fs::create_dir_all(&folder).unwrap();
        let home = HomeDir::new(tmp.path().join("home"));
        home.ensure_initialized().unwrap();
        let pm = ProjectMemory::with_base(&tmp.path().join("projects"), &folder);
        (tmp, pm, home)
    }

    #[test]
    fn append_creates_the_directory_and_notes_the_root() {
        let (_tmp, pm, home) = folder_memory();
        assert!(!pm.dir().exists());
        pm.apply(
            Some(&home),
            MemoryFile::Folder,
            &MemoryEdit::Append("- uses pnpm".into()),
            "2026-11-03",
        )
        .unwrap();
        pm.apply(
            Some(&home),
            MemoryFile::Folder,
            &MemoryEdit::Append("- tabs, not spaces".into()),
            "2026-11-03",
        )
        .unwrap();
        assert_eq!(
            pm.read("MEMORY.md").unwrap(),
            "- uses pnpm\n- tabs, not spaces\n"
        );
        assert!(pm.dir().join("ROOT").is_file());
        pm.apply(
            Some(&home),
            MemoryFile::Daily,
            &MemoryEdit::Append("looked at CI".into()),
            "2026-11-03",
        )
        .unwrap();
        assert!(pm.dir().join("memory/2026-11-03.md").is_file());
    }

    #[test]
    fn replace_needs_exactly_one_match_and_empty_text_deletes() {
        let (_tmp, pm, home) = folder_memory();
        let write = |e: MemoryEdit| pm.apply(Some(&home), MemoryFile::Folder, &e, "d");
        write(MemoryEdit::Rewrite("- a\n- b\n- a2\n- c".into())).unwrap();
        let rep = |old: &str, new: &str| MemoryEdit::Replace {
            old: old.into(),
            new: new.into(),
        };
        assert!(write(rep("- z", "x")).unwrap_err().contains("not found"));
        assert!(write(rep("- a", "x")).unwrap_err().contains("2 times"));
        write(rep("- b\n", "")).unwrap();
        write(rep("- c", "- C")).unwrap();
        assert_eq!(pm.read("MEMORY.md").unwrap(), "- a\n- a2\n- C\n");
        assert!(write(rep("", "x")).is_err());
    }

    #[test]
    fn writes_are_capped_and_user_needs_a_home() {
        let (_tmp, pm, home) = folder_memory();
        let long = "あ".repeat(MAX_MEMORY_WRITE_CHARS + 1);
        assert!(pm
            .apply(
                Some(&home),
                MemoryFile::Folder,
                &MemoryEdit::Append(long),
                "d"
            )
            .unwrap_err()
            .contains("too much"));
        assert!(pm
            .apply(None, MemoryFile::User, &MemoryEdit::Append("x".into()), "d")
            .is_err());
        pm.apply(
            Some(&home),
            MemoryFile::User,
            &MemoryEdit::Append("- prefers 日本語".into()),
            "d",
        )
        .unwrap();
        assert!(home.read("USER.md").unwrap().contains("prefers 日本語"));
        assert_eq!(
            pm.apply(Some(&home), MemoryFile::Folder, &MemoryEdit::Read, "d")
                .unwrap(),
            "(empty)"
        );
    }

    #[test]
    fn the_shell_is_kept_away_from_memory_and_grants() {
        let (tmp, _pm, home) = folder_memory();
        let base = tmp.path().join("projects");
        let touches = |c: &str| command_touches_memory(c, &base, Some(&home));
        assert!(touches("cat >> ~/.kotonia/projects/x/MEMORY.md"));
        assert!(touches(&format!(
            "echo '{{}}' > {}/grants.json",
            base.display()
        )));
        assert!(touches("cd ~/.kotonia/home && echo hi >> USER.md"));
        assert!(touches(&format!(
            "sed -i s/a/b/ {}",
            home.path("USER.md").display()
        )));
        assert!(!touches("cat ~/.kotonia/home/MEMORY.md"));
        assert!(!touches("cat README.md"));
    }

    #[test]
    fn mail_grants_belong_to_one_folder() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path().join("projects");
        let (one, two) = (tmp.path().join("one"), tmp.path().join("two"));
        std::fs::create_dir_all(&one).unwrap();
        std::fs::create_dir_all(&two).unwrap();
        let a = ProjectMemory::with_base(&base, &one);
        let b = ProjectMemory::with_base(&base, &two);
        assert!(a.set_mail_grant("imap.gmail.me", true).unwrap());
        assert!(!a.set_mail_grant("imap.gmail.me", true).unwrap());
        assert_eq!(a.mail_grants(), vec!["imap.gmail.me".to_string()]);
        assert!(b.mail_grants().is_empty());
        assert!(a.set_mail_grant("imap.gmail.me", false).unwrap());
        assert!(a.mail_grants().is_empty());
        assert!(a.load_grants().unwrap().folders.is_empty());
    }
}
