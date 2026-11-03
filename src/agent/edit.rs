//! `edit_file`: change part of a text file by exact replacement.
//!
//! Without it the agent rewrote whole files through heredocs, `sed -i` or
//! Python one-offs to change a line — long outputs, and the shell versions
//! fail silently (a `sed` pattern that matches nothing exits 0). An exact
//! match that must be unique fails loudly instead, and the result shows the
//! edited lines so the agent can check its own work.

use std::path::Path;

/// Files bigger than this are not edited in one piece (logs, generated
/// bundles); the agent should not be loading them into its context anyway.
pub const MAX_EDIT_BYTES: u64 = 2 * 1024 * 1024;

/// Lines of context shown around the edit.
const CONTEXT_LINES: usize = 3;
/// Cap on the snippet returned to the model.
const MAX_SNIPPET_LINES: usize = 60;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EditOutcome {
    pub replacements: usize,
    pub created: bool,
    /// The edited region with line numbers (`  12| text`).
    pub snippet: String,
}

pub fn apply_edit(
    path: &Path,
    old: &str,
    new: &str,
    replace_all: bool,
) -> Result<EditOutcome, String> {
    if old.is_empty() {
        return create(path, new);
    }
    if old == new {
        return Err("`old_text` and `new_text` are the same; nothing to change".into());
    }
    let meta = std::fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if !meta.is_file() {
        return Err(format!("{} is not a regular file", path.display()));
    }
    if meta.len() > MAX_EDIT_BYTES {
        return Err(format!(
            "{} is {} bytes (limit {MAX_EDIT_BYTES}); edit it with a script instead",
            path.display(),
            meta.len()
        ));
    }
    let content = std::fs::read_to_string(path).map_err(|e| {
        format!(
            "{}: {e} (only UTF-8 text files can be edited)",
            path.display()
        )
    })?;

    // A CRLF file and an `old_text` written with plain newlines would never
    // match; retry with the file's line endings.
    let (old, new) = match (content.matches(old).count(), content.contains("\r\n")) {
        (0, true) if old.contains('\n') && !old.contains("\r\n") => {
            (old.replace('\n', "\r\n"), new.replace('\n', "\r\n"))
        }
        _ => (old.to_string(), new.to_string()),
    };

    let positions: Vec<usize> = content.match_indices(&old).map(|(i, _)| i).collect();
    match positions.len() {
        0 => return Err(not_found_hint(&content, &old)),
        1 => {}
        n if !replace_all => {
            let lines: Vec<String> = positions
                .iter()
                .take(10)
                .map(|&i| line_of(&content, i).to_string())
                .collect();
            return Err(format!(
                "`old_text` appears {n} times (lines {}); add surrounding lines so it is \
unique, or set `replace_all`",
                lines.join(", ")
            ));
        }
        _ => {}
    }

    let first = positions[0];
    let updated = if replace_all {
        content.replace(&old, &new)
    } else {
        content.replacen(&old, &new, 1)
    };
    std::fs::write(path, &updated).map_err(|e| format!("write {}: {e}", path.display()))?;

    let start = line_of(&updated, first);
    let end = start + new.matches('\n').count();
    Ok(EditOutcome {
        replacements: if replace_all { positions.len() } else { 1 },
        created: false,
        snippet: snippet(&updated, start, end),
    })
}

fn create(path: &Path, text: &str) -> Result<EditOutcome, String> {
    if let Ok(existing) = std::fs::read(path) {
        if !existing.is_empty() {
            return Err(format!(
                "{} already exists; give `old_text` to change part of it",
                path.display()
            ));
        }
    }
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("create {}: {e}", parent.display()))?;
        }
    }
    std::fs::write(path, text).map_err(|e| format!("write {}: {e}", path.display()))?;
    let end = text.lines().count().max(1);
    Ok(EditOutcome {
        replacements: 0,
        created: true,
        snippet: snippet(text, 1, end),
    })
}

/// 1-based line number of byte offset `i`.
fn line_of(content: &str, i: usize) -> usize {
    content[..i].matches('\n').count() + 1
}

fn snippet(content: &str, start: usize, end: usize) -> String {
    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return "(file is empty)".into();
    }
    let from = start.saturating_sub(CONTEXT_LINES).max(1);
    let to = (end + CONTEXT_LINES).min(lines.len());
    let width = to.to_string().len();
    let mut out: Vec<String> = (from..=to)
        .take(MAX_SNIPPET_LINES)
        .map(|n| format!("{n:>width$}| {}", lines[n - 1]))
        .collect();
    if to + 1 - from > MAX_SNIPPET_LINES {
        out.push(format!(
            "… ({} more lines)",
            to + 1 - from - MAX_SNIPPET_LINES
        ));
    }
    out.join("\n")
}

/// Why nothing matched, when we can tell: usually indentation or trailing
/// whitespace copied wrong.
fn not_found_hint(content: &str, old: &str) -> String {
    let squash = |s: &str| s.split_whitespace().collect::<Vec<_>>().join(" ");
    let first = old.lines().map(str::trim).find(|l| !l.is_empty());
    if let Some(first) = first {
        let hits: Vec<usize> = content
            .lines()
            .enumerate()
            .filter(|(_, l)| l.contains(first))
            .map(|(i, _)| i + 1)
            .take(5)
            .collect();
        if !hits.is_empty() && squash(content).contains(&squash(old)) {
            return format!(
                "`old_text` was not found exactly, but matches if whitespace is ignored — \
check indentation and line breaks (its first line is on line {})",
                hits.iter()
                    .map(|n| n.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        if !hits.is_empty() {
            return format!(
                "`old_text` was not found. Its first line appears on line {}; read those lines \
again and copy them exactly",
                hits.iter()
                    .map(|n| n.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
    }
    "`old_text` was not found; read the file again and copy the text exactly".into()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(body: &str) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.rs");
        std::fs::write(&path, body).unwrap();
        (dir, path)
    }

    #[test]
    fn replaces_one_unique_match_and_shows_numbered_lines() {
        let (_d, p) = file("fn a() {\n    1\n}\n\nfn b() {\n    2\n}\n");
        let out = apply_edit(&p, "    2\n", "    20\n    21\n", false).unwrap();
        assert_eq!(out.replacements, 1);
        assert_eq!(
            std::fs::read_to_string(&p).unwrap(),
            "fn a() {\n    1\n}\n\nfn b() {\n    20\n    21\n}\n"
        );
        assert!(out.snippet.contains("6|     20"), "{}", out.snippet);
        assert!(out.snippet.contains("7|     21"), "{}", out.snippet);
    }

    #[test]
    fn several_matches_need_more_context_or_replace_all() {
        let (_d, p) = file("x = 1\ny = 2\nx = 1\n");
        let err = apply_edit(&p, "x = 1", "x = 3", false).unwrap_err();
        assert!(
            err.contains("2 times") && err.contains("lines 1, 3"),
            "{err}"
        );
        assert_eq!(
            std::fs::read_to_string(&p).unwrap(),
            "x = 1\ny = 2\nx = 1\n"
        );
        let out = apply_edit(&p, "x = 1", "x = 3", true).unwrap();
        assert_eq!(out.replacements, 2);
        assert_eq!(
            std::fs::read_to_string(&p).unwrap(),
            "x = 3\ny = 2\nx = 3\n"
        );
    }

    #[test]
    fn whitespace_mistakes_get_a_useful_hint() {
        let (_d, p) = file("fn a() {\n    let x = 1;\n}\n");
        let err = apply_edit(&p, "fn a() {\n  let x = 1;", "z", false).unwrap_err();
        assert!(err.contains("whitespace is ignored"), "{err}");
        let err = apply_edit(&p, "nothing like this", "z", false).unwrap_err();
        assert!(err.contains("not found"), "{err}");
    }

    #[test]
    fn crlf_files_match_plain_newlines() {
        let (_d, p) = file("a\r\nb\r\nc\r\n");
        apply_edit(&p, "a\nb", "a\nB", false).unwrap();
        assert_eq!(std::fs::read_to_string(&p).unwrap(), "a\r\nB\r\nc\r\n");
    }

    #[test]
    fn empty_old_text_creates_but_never_clobbers() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("new/dir/b.txt");
        let out = apply_edit(&p, "", "hello\n", false).unwrap();
        assert!(out.created);
        assert_eq!(std::fs::read_to_string(&p).unwrap(), "hello\n");
        assert!(apply_edit(&p, "", "other", false)
            .unwrap_err()
            .contains("already exists"));
    }

    #[test]
    fn same_text_and_binary_files_are_refused() {
        let (_d, p) = file("a\n");
        assert!(apply_edit(&p, "a", "a", false).is_err());
        let dir = tempfile::tempdir().unwrap();
        let bin = dir.path().join("x.bin");
        std::fs::write(&bin, [0xff, 0xfe, 0x00, 0x61]).unwrap();
        assert!(apply_edit(&bin, "a", "b", false)
            .unwrap_err()
            .contains("UTF-8"));
    }
}
