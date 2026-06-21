//! Parse model output into the next agent action.
//!
//! Looks for the first occurrence of either a `<<<BASH>>>...<<<END_BASH>>>`
//! block or a `<<<FINAL_ANSWER>>>...<<<END_FINAL_ANSWER>>>` block. Whichever
//! appears first wins, so a model that emits both gets the bash (and we'll
//! tell it next turn to stop emitting two blocks).

use super::prompt::{BASH_CLOSE, BASH_OPEN, FINAL_CLOSE, FINAL_OPEN};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    Bash(String),
    Final(String),
    /// Model emitted neither a bash block nor a final answer. The caller
    /// should prod it to try again rather than silently looping.
    Malformed { excerpt: String },
}

pub fn parse(response: &str) -> Action {
    let bash = extract(response, BASH_OPEN, BASH_CLOSE);
    let final_ = extract(response, FINAL_OPEN, FINAL_CLOSE);
    match (bash, final_) {
        (Some((b_start, b_body)), Some((f_start, f_body))) => {
            if b_start <= f_start {
                Action::Bash(b_body)
            } else {
                Action::Final(f_body)
            }
        }
        (Some((_, b)), None) => Action::Bash(b),
        (None, Some((_, f))) => Action::Final(f),
        (None, None) => fallback(response),
    }
}

/// Long-context chat models (Gemma4 26B especially) sometimes drop the
/// `<<<FINAL_ANSWER>>>` wrapper and just write the answer in prose. Force-
/// retrying via "you must use the format" loses an iteration and looks like
/// an error to the operator. We instead accept the prose as the final answer
/// — UNLESS it reads as "I'm about to run a command" (in which case we want
/// the model to actually emit a BASH block on retry, not terminate).
fn fallback(response: &str) -> Action {
    let trimmed = response.trim();
    if trimmed.is_empty() {
        return Action::Malformed {
            excerpt: response.chars().take(400).collect(),
        };
    }
    if looks_like_action_intention(trimmed) {
        return Action::Malformed {
            excerpt: trimmed.chars().take(400).collect(),
        };
    }
    Action::Final(trimmed.to_string())
}

/// "I'll run X" / "Let me check Y" / "次は Z をやる" patterns. If the model
/// is announcing an action instead of taking it, we want a retry so it
/// actually emits the BASH block, not a premature FINAL_ANSWER.
fn looks_like_action_intention(text: &str) -> bool {
    let head: String = text.chars().take(80).collect();
    let lower = head.to_ascii_lowercase();
    const EN_PHRASES: &[&str] = &[
        "i'll ", "i will ", "i shall ", "let me ", "let's ", "let us ", "now i ",
        "first, ", "first i ", "next, ", "i'm going to ", "i am going to ",
        "i'd like to ", "i need to run ", "i should run ", "i'll execute ",
        "i'll check ", "i'll look ", "i'll start ", "going to run ",
    ];
    if EN_PHRASES.iter().any(|p| lower.starts_with(p)) {
        return true;
    }
    const JA_PREFIXES: &[&str] = &[
        "次は", "次に", "じゃあ", "では", "それでは", "まず", "まずは",
        "今から", "確認します", "実行します", "やってみます", "見てみます",
        "やります", "実行してみます", "確認してみます", "ちょっと",
    ];
    JA_PREFIXES.iter().any(|p| head.starts_with(p))
}

fn extract(hay: &str, open: &str, close: &str) -> Option<(usize, String)> {
    let start = hay.find(open)?;
    let body_start = start + open.len();
    let rest = &hay[body_start..];
    let end_rel = rest.find(close)?;
    let body = rest[..end_rel].trim().to_string();
    Some((start, body))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_bash() {
        let r = "thinking...\n<<<BASH>>>\nls -la\n<<<END_BASH>>>";
        assert_eq!(parse(r), Action::Bash("ls -la".to_string()));
    }

    #[test]
    fn parses_final() {
        let r = "<<<FINAL_ANSWER>>>\nAll good.\n<<<END_FINAL_ANSWER>>>";
        assert_eq!(parse(r), Action::Final("All good.".to_string()));
    }

    #[test]
    fn bash_wins_when_first() {
        let r = "<<<BASH>>>ls<<<END_BASH>>>\n<<<FINAL_ANSWER>>>done<<<END_FINAL_ANSWER>>>";
        assert_eq!(parse(r), Action::Bash("ls".to_string()));
    }

    #[test]
    fn final_wins_when_first() {
        let r = "<<<FINAL_ANSWER>>>done<<<END_FINAL_ANSWER>>>\n<<<BASH>>>ls<<<END_BASH>>>";
        assert_eq!(parse(r), Action::Final("done".to_string()));
    }

    #[test]
    fn prose_without_tags_treated_as_final() {
        // Long-context drift: model emits a clean prose answer without tags.
        // Lenient fallback treats this as the final answer (no wasteful retry).
        let r = "現在のディレクトリは /tmp で、ファイルは 12 個あります。";
        match parse(r) {
            Action::Final(text) => assert!(text.contains("12 個")),
            other => panic!("expected Final, got {other:?}"),
        }
    }

    #[test]
    fn action_intention_without_tags_still_retries() {
        // "I'll run X" without a BASH block — we want the model to actually
        // run the command on retry, not terminate early with a Final.
        for r in [
            "I'll list the files now.",
            "Let me check the README.",
            "次は ls を実行します。",
            "じゃあ cat README.md を試してみます。",
        ] {
            match parse(r) {
                Action::Malformed { excerpt } => assert!(!excerpt.is_empty(), "{r}"),
                other => panic!("expected Malformed for `{r}`, got {other:?}"),
            }
        }
    }

    #[test]
    fn empty_response_is_malformed() {
        match parse("   \n\n  ") {
            Action::Malformed { .. } => {}
            other => panic!("expected Malformed for empty response, got {other:?}"),
        }
    }

    #[test]
    fn handles_multiline_bash() {
        let r = "<<<BASH>>>\ncargo check 2>&1 \\\n  | tail -20\n<<<END_BASH>>>";
        if let Action::Bash(cmd) = parse(r) {
            assert!(cmd.contains("cargo check"));
            assert!(cmd.contains("tail -20"));
        } else {
            panic!("expected bash");
        }
    }
}
