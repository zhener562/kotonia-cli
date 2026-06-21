//! Worktree management for the kotonia-cli agent.
//!
//! Default mode creates `git worktree add <tmp>` against a base branch so the
//! agent's edits never touch the operator's working copy. `--in-place` skips
//! all of this and operates directly on the launch cwd, intended for users
//! who want speed over isolation.

use std::path::{Path, PathBuf};

use crate::execution::host::{HostExecutor, HostExecutorError};

#[derive(Debug)]
pub enum WorktreeError {
    NotAGitRepo(PathBuf),
    Executor(HostExecutorError),
    GitFailed { command: String, output: String },
    Io(std::io::Error),
}

impl std::fmt::Display for WorktreeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WorktreeError::NotAGitRepo(p) => write!(f, "{} is not inside a git repo", p.display()),
            WorktreeError::Executor(e) => write!(f, "host executor: {e}"),
            WorktreeError::GitFailed { command, output } => {
                write!(f, "git command failed: `{command}`\n{output}")
            }
            WorktreeError::Io(e) => write!(f, "I/O error: {e}"),
        }
    }
}

impl std::error::Error for WorktreeError {}

impl From<HostExecutorError> for WorktreeError {
    fn from(e: HostExecutorError) -> Self {
        WorktreeError::Executor(e)
    }
}

impl From<std::io::Error> for WorktreeError {
    fn from(e: std::io::Error) -> Self {
        WorktreeError::Io(e)
    }
}

/// What the agent is going to operate inside. Either an isolated git worktree
/// (the default) or the caller's own cwd (`--in-place`).
pub struct AgentWorkspace {
    pub root: PathBuf,
    pub mode: WorkspaceMode,
}

pub enum WorkspaceMode {
    /// Spawned `git worktree add <root>`. Should be cleaned up on exit.
    Worktree {
        repo_root: PathBuf,
        base_ref: String,
        branch: String,
    },
    /// `--in-place` — the launch cwd is the workspace.
    InPlace,
}

impl AgentWorkspace {
    pub fn in_place<P: Into<PathBuf>>(cwd: P) -> Self {
        Self {
            root: cwd.into(),
            mode: WorkspaceMode::InPlace,
        }
    }

    /// Build a fresh worktree off `base_ref` in the repo containing `cwd`.
    /// `base_ref` defaults to the current branch when None.
    pub async fn create_worktree(
        cwd: &Path,
        base_ref: Option<&str>,
    ) -> Result<Self, WorktreeError> {
        let exec = HostExecutor::new(cwd.to_path_buf());

        let repo_root = run_git(&exec, "git rev-parse --show-toplevel").await?;
        let repo_root = PathBuf::from(repo_root.trim());
        if !repo_root.exists() {
            return Err(WorktreeError::NotAGitRepo(cwd.to_path_buf()));
        }

        let resolved_base = match base_ref {
            Some(b) => b.to_string(),
            None => {
                let head = run_git(&exec, "git rev-parse --abbrev-ref HEAD").await?;
                head.trim().to_string()
            }
        };

        // Worktree path: /tmp/kotonia-agent-<8 char uuid>
        let suffix: String = uuid::Uuid::new_v4()
            .to_string()
            .chars()
            .take(8)
            .collect();
        let wt_path = std::env::temp_dir().join(format!("kotonia-agent-{suffix}"));
        let branch = format!("kotonia-agent/{suffix}");

        // git worktree add -b <branch> <path> <base-ref>
        let cmd = format!(
            "git worktree add -b {branch} {wt} {base}",
            wt = shell_escape(&wt_path.to_string_lossy()),
            base = shell_escape(&resolved_base),
        );
        run_git(&exec, &cmd).await?;

        Ok(Self {
            root: wt_path,
            mode: WorkspaceMode::Worktree {
                repo_root,
                base_ref: resolved_base,
                branch,
            },
        })
    }

    /// Tear down the worktree. No-op for `in_place`.
    pub async fn cleanup(self, keep: bool) -> Result<(), WorktreeError> {
        match self.mode {
            WorkspaceMode::InPlace => Ok(()),
            WorkspaceMode::Worktree {
                repo_root, branch, ..
            } => {
                if keep {
                    // Leave the worktree on disk so the operator can inspect /
                    // merge it later. They can run `git worktree remove` manually.
                    return Ok(());
                }
                let exec = HostExecutor::new(repo_root);
                let path = self.root;
                let remove_cmd = format!(
                    "git worktree remove --force {}",
                    shell_escape(&path.to_string_lossy())
                );
                let _ = run_git(&exec, &remove_cmd).await; // best-effort
                let _ = run_git(&exec, &format!("git branch -D {branch}")).await;
                Ok(())
            }
        }
    }

    pub fn is_worktree(&self) -> bool {
        matches!(self.mode, WorkspaceMode::Worktree { .. })
    }

    pub fn branch(&self) -> Option<&str> {
        match &self.mode {
            WorkspaceMode::Worktree { branch, .. } => Some(branch),
            WorkspaceMode::InPlace => None,
        }
    }
}

async fn run_git(exec: &HostExecutor, cmd: &str) -> Result<String, WorktreeError> {
    let r = exec.bash(cmd).await?;
    if r.exit_code != 0 {
        return Err(WorktreeError::GitFailed {
            command: cmd.to_string(),
            output: r.combined,
        });
    }
    Ok(r.combined)
}

fn shell_escape(s: &str) -> String {
    if s.chars().all(|c| {
        c.is_ascii_alphanumeric() || c == '/' || c == '.' || c == '_' || c == '-'
    }) {
        s.to_string()
    } else {
        let escaped = s.replace('\'', "'\\''");
        format!("'{escaped}'")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn in_place_workspace_is_passthrough() {
        let w = AgentWorkspace::in_place("/tmp");
        assert_eq!(w.root, PathBuf::from("/tmp"));
        assert!(!w.is_worktree());
        // cleanup is a no-op and must not error
        w.cleanup(false).await.unwrap();
    }

    #[tokio::test]
    async fn create_and_cleanup_worktree() {
        // Bootstrap a tiny repo in tmpdir.
        let dir = tempfile::tempdir().unwrap();
        let repo = dir.path();
        let setup = HostExecutor::new(repo.to_path_buf());
        setup
            .bash(
                "git init -q && git config user.email a@b && git config user.name a \
                 && echo hi > f.txt && git add f.txt && git commit -qm init",
            )
            .await
            .unwrap();

        let w = AgentWorkspace::create_worktree(repo, None).await.unwrap();
        assert!(w.is_worktree());
        assert!(w.root.exists());
        assert!(w.root.join("f.txt").exists());
        let branch = w.branch().unwrap().to_string();
        let root = w.root.clone();
        w.cleanup(false).await.unwrap();
        assert!(!root.exists(), "worktree dir should be removed");

        // Branch should also be gone.
        let check = setup
            .bash(&format!("git branch --list {branch}"))
            .await
            .unwrap();
        assert!(check.combined.trim().is_empty());
    }
}
