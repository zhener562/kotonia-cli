//! Keep the conversation inside the model's context window.
//!
//! Every request resends the whole conversation, and in an agent session
//! that conversation is mostly command output. Two stages, both run only
//! when the estimate crosses `trigger` and both cut down to `target`, so
//! the conversation is rewritten rarely (an event, not a per-call
//! recompute) and the provider's prefix cache keeps hitting in between:
//!
//! 1. **Stub tool output from finished turns.** No LLM. The model has
//!    already answered from those outputs, and the repository, not the
//!    conversation, is the agent's source of truth, so an old `cat` or
//!    `grep` can be re-run (oversized outputs also point at a saved copy,
//!    `execution::host`).
//! 2. **Summarise the older part.** When stage 1 is not enough — in
//!    particular inside the current turn, whose outputs the model may still
//!    need. The summariser reads those outputs before they go, so facts the
//!    task needs move into the notes along with the task state (goal, files
//!    touched, decisions, what failed, what is left). File contents are
//!    left out on purpose: a model misremembers them, `git diff` does not.
//!
//! Stubbing inside the current turn was tried first and loops: with the
//! window holding 2 of 8 files the agent was asked to read, it re-read the
//! stubbed one, lost the next, and went round until the iteration cap.
//!
//! This is deliberately different from kotonia.ai's character-chat
//! compaction (generational chapters): there the past conversation is the
//! only record of the story, here it is a cache of things the agent can
//! look at again.

use crate::ai::{AiContent, AiMessage, AiRole};

use super::provider::{ChatMsg, ChatRole};

/// Rough tokens from UTF-8 bytes. Bytes/token is steadier across languages
/// than chars/token (Japanese ~3 bytes/char at ~1.3 chars/token, English
/// and code ~3.5-4 bytes/token), and 3 errs towards compacting early.
const BYTES_PER_TOKEN: usize = 3;
/// An inline image costs a fixed amount of tokens, not its base64 length.
const IMAGE_TOKENS: usize = 1_500;
/// Marker that makes stubbing idempotent.
const STUB_MARK: &str = "[older output elided";
/// Upper bound for one stub's leading header lines.
const STUB_HEADER_MAX_BYTES: usize = 300;
/// Cap for one tool result inside the text handed to the summariser.
const SUMMARY_RESULT_MAX_BYTES: usize = 3_000;
/// Room the notes are expected to take in the context after compaction.
pub const SUMMARY_RESERVE_TOKENS: usize = 1_500;

pub const SUMMARY_SYSTEM: &str = "\
You are compacting the working memory of a coding agent whose context window is full. \
You will be given the earlier part of its session (task messages, its own reasoning, \
the commands it ran and their outputs). Long outputs are shortened *in this transcript \
only* (marked `[… N bytes omitted here …]`): the agent saw them in full, so never \
record them as truncated or incomplete. Write the notes the agent needs to carry \
on, in the language the user wrote in, under these headings:

## Goal
What the user asked for, including constraints and preferences they stated.
## Done so far
Files created/changed and what was changed, commands that mattered, results.
## Facts gathered
Values, names, paths and line numbers found in the outputs that the task still \
needs. The outputs themselves are being dropped, so anything not written here is lost.
## Decisions
Choices made and why.
## Tried and failed
Approaches that did not work and why — so they are not repeated.
## Open
What is left, and anything the user still has to answer.

Do not copy file contents or long outputs: the agent can re-read files and run \
`git diff`, and those are the source of truth. Be concrete (paths, names, numbers). \
Output only the notes.";

const SUMMARY_PREAMBLE: &str = "\
[Notes on the earlier part of this session, written when the context window filled up. \
The repository is the source of truth: re-read files and `git diff` instead of trusting \
remembered contents.]";

/// When to compact and how far down.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Budget {
    pub trigger_tokens: usize,
    pub target_tokens: usize,
}

impl Budget {
    pub fn for_model(model: &str, max_output_tokens: u32) -> Self {
        Self::for_window(window_tokens_for(model), max_output_tokens)
    }

    pub fn for_window(window: usize, max_output_tokens: u32) -> Self {
        let trigger = (window * 3 / 4).saturating_sub(max_output_tokens as usize);
        Self {
            trigger_tokens: trigger.max(1),
            target_tokens: (window / 2).min(trigger).max(1),
        }
    }
}

/// Context window in tokens. Only models whose window we have checked get
/// more than the conservative default; `KOTONIA_CONTEXT_WINDOW_TOKENS`
/// overrides everything (custom providers, tests).
pub fn window_tokens_for(model: &str) -> usize {
    if let Some(n) = std::env::var("KOTONIA_CONTEXT_WINDOW_TOKENS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|n| *n > 0)
    {
        return n;
    }
    let base = model.split(':').next().unwrap_or(model);
    match base {
        // kotonia.ai's local model: 64K per llama-server slot.
        "kotonia-llm" | "kotonia-llm-basic" => 65_536,
        // DeepSeek behind kotonia.ai, or direct.
        "kotonia-llm-standard" | "deepseek-chat" | "deepseek-reasoner" => 131_072,
        _ => 32_768,
    }
}

pub fn estimate_text(s: &str) -> usize {
    s.len().div_ceil(BYTES_PER_TOKEN)
}

pub fn estimate_native(msgs: &[AiMessage]) -> usize {
    msgs.iter()
        .flat_map(|m| &m.content)
        .map(|c| match c {
            AiContent::Text { text } => estimate_text(text),
            AiContent::Image { .. } => IMAGE_TOKENS,
            AiContent::ToolUse { name, input, .. } => {
                estimate_text(name) + estimate_text(&input.to_string())
            }
            AiContent::ToolResult { content, .. } => estimate_text(content),
        })
        .sum()
}

pub fn estimate_chat(msgs: &[ChatMsg]) -> usize {
    msgs.iter().map(|m| estimate_text(&m.content)).sum()
}

/// One-line stand-in for an old tool output: keeps the leading `[...]`
/// header lines (exit code, saved-file path, `[tool bash (id=..)]`).
fn stub(content: &str) -> String {
    let mut header = String::new();
    for line in content.lines().take(2) {
        if !line.starts_with('[') || header.len() + line.len() > STUB_HEADER_MAX_BYTES {
            break;
        }
        header.push_str(line);
        header.push('\n');
    }
    format!(
        "{header}{STUB_MARK}: {} bytes. Re-run the command, or read the saved file if one \
         is named above, if you need it again.]",
        content.len()
    )
}

fn is_stub(content: &str) -> bool {
    content.contains(STUB_MARK)
}

/// Stage 1 for the native conversation. Stubs tool results and inline
/// images in `msgs[..turn_start]` (finished turns) oldest first, until the
/// estimate (`fixed` + messages) drops to `target`. Returns how many.
pub fn stub_old_results_native(
    msgs: &mut [AiMessage],
    turn_start: usize,
    fixed: usize,
    target: usize,
) -> usize {
    let mut estimate = fixed + estimate_native(msgs);
    let mut stubbed = 0;
    let end = turn_start.min(msgs.len());
    'outer: for m in msgs[..end].iter_mut() {
        for c in m.content.iter_mut() {
            if estimate <= target {
                break 'outer;
            }
            match c {
                AiContent::ToolResult { content, .. } if !is_stub(content) => {
                    let s = stub(content);
                    estimate = estimate + estimate_text(&s) - estimate_text(content);
                    *content = s;
                    stubbed += 1;
                }
                AiContent::Image { .. } => {
                    *c = AiContent::text("[image elided to save context]");
                    estimate = estimate - IMAGE_TOKENS + 10;
                    stubbed += 1;
                }
                _ => {}
            }
        }
    }
    stubbed
}

/// Delimiter-mode observations are plain user messages with these prefixes
/// (`HostExecutor::as_observation` and the native loop's legacy mirror).
fn is_observation_text(content: &str) -> bool {
    content.starts_with("[exit ")
        || content.starts_with("[timed out")
        || content.starts_with("[tool ")
}

/// Stage 1 for the delimiter conversation (`msgs[0]` is the system
/// prompt), over `msgs[..turn_start]`.
pub fn stub_old_results_chat(msgs: &mut [ChatMsg], turn_start: usize, target: usize) -> usize {
    let mut estimate = estimate_chat(msgs);
    let mut stubbed = 0;
    let end = turn_start.min(msgs.len());
    for m in msgs[..end].iter_mut() {
        if estimate <= target {
            break;
        }
        if m.role != ChatRole::User || !is_observation_text(&m.content) {
            continue;
        }
        let content = &mut m.content;
        if is_stub(content) {
            continue;
        }
        let s = stub(content);
        estimate = estimate + estimate_text(&s) - estimate_text(content);
        *content = s;
        stubbed += 1;
    }
    stubbed
}

/// Stage 2 cut point for the native conversation: the smallest index `c`
/// such that `msgs[c..]` fits in `keep_tokens`, at a place where no tool
/// call is separated from its result. If even the last exchange does not
/// fit, the latest valid cut (keep as little as possible). `None` only if
/// there is no valid cut at all.
pub fn cut_native(msgs: &[AiMessage], keep_tokens: usize) -> Option<usize> {
    let valid = |c: usize| -> bool {
        let prev = &msgs[c - 1];
        match prev.role {
            // After a tool-result message the next call is a fresh exchange.
            AiRole::Tool => true,
            // A new user message is a turn boundary unless the assistant
            // before it is still waiting for tool results.
            _ => {
                msgs[c].role == AiRole::User
                    && !prev
                        .content
                        .iter()
                        .any(|b| matches!(b, AiContent::ToolUse { .. }))
            }
        }
    };
    let mut tail = 0;
    let mut best = None;
    let mut latest = None;
    for c in (1..msgs.len()).rev() {
        tail += estimate_native(&msgs[c..=c]);
        if valid(c) {
            latest.get_or_insert(c);
            if tail <= keep_tokens {
                best = Some(c);
            }
        }
        if tail > keep_tokens && latest.is_some() {
            break;
        }
    }
    best.or(latest)
}

/// Stage 2 cut point for the delimiter conversation (`msgs[0]` is the
/// system prompt, kept). Cuts land before a user message; same fallback.
pub fn cut_chat(msgs: &[ChatMsg], keep_tokens: usize) -> Option<usize> {
    let mut tail = 0;
    let mut best = None;
    let mut latest = None;
    for c in (2..msgs.len()).rev() {
        tail += estimate_text(&msgs[c].content);
        if msgs[c].role == ChatRole::User {
            latest.get_or_insert(c);
            if tail <= keep_tokens {
                best = Some(c);
            }
        }
        if tail > keep_tokens && latest.is_some() {
            break;
        }
    }
    best.or(latest)
}

/// Head and tail of `s`, for the summariser's transcript. The marker says
/// the cut is local to the transcript: a first version cut only the tail
/// with `… [+N bytes]`, and the summariser wrote down "every cat was
/// truncated, the last lines were never seen" — the agent then re-read
/// every file.
fn clip(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut h = max * 2 / 3;
    while !s.is_char_boundary(h) {
        h -= 1;
    }
    let mut t = s.len() - (max - max * 2 / 3);
    while !s.is_char_boundary(t) {
        t += 1;
    }
    format!(
        "{}\n[… {} bytes omitted here; the agent saw them …]\n{}",
        &s[..h],
        t - h,
        &s[t..]
    )
}

/// Keep the head (where the goal is) and the larger tail of a transcript
/// that would not fit the summariser's own window.
fn fit(s: String, max: usize) -> String {
    if s.len() <= max {
        return s;
    }
    let head = max / 3;
    let tail = max - head;
    let mut h = head;
    while !s.is_char_boundary(h) {
        h -= 1;
    }
    let mut t = s.len() - tail;
    while !s.is_char_boundary(t) {
        t += 1;
    }
    format!("{}\n\n[… middle omitted …]\n\n{}", &s[..h], &s[t..])
}

pub fn render_native(msgs: &[AiMessage], max_bytes: usize) -> String {
    let mut out = String::new();
    for m in msgs {
        for c in &m.content {
            match c {
                AiContent::Text { text } if !text.trim().is_empty() => {
                    let who = match m.role {
                        AiRole::User => "USER",
                        AiRole::Assistant => "AGENT",
                        AiRole::Tool => "TOOL",
                        AiRole::System => "SYSTEM",
                    };
                    out.push_str(&format!("{who}: {}\n\n", text.trim()));
                }
                AiContent::ToolUse { name, input, .. } => {
                    out.push_str(&format!(
                        "AGENT ran {name}: {}\n\n",
                        clip(&input.to_string(), 600)
                    ));
                }
                AiContent::ToolResult { content, .. } => {
                    out.push_str(&format!(
                        "OUTPUT: {}\n\n",
                        clip(content.trim(), SUMMARY_RESULT_MAX_BYTES)
                    ));
                }
                _ => {}
            }
        }
    }
    fit(out, max_bytes)
}

pub fn render_chat(msgs: &[ChatMsg], max_bytes: usize) -> String {
    let mut out = String::new();
    for m in msgs {
        let (who, text) = match m.role {
            ChatRole::System => continue,
            ChatRole::User if is_observation_text(&m.content) => {
                ("OUTPUT", clip(m.content.trim(), SUMMARY_RESULT_MAX_BYTES))
            }
            ChatRole::User => ("USER", m.content.trim().to_string()),
            ChatRole::Assistant => ("AGENT", m.content.trim().to_string()),
        };
        out.push_str(&format!("{who}: {text}\n\n"));
    }
    fit(out, max_bytes)
}

/// What replaces the summarised prefix.
pub fn replacement_text(summary: &str) -> String {
    format!("{SUMMARY_PREAMBLE}\n\n{}", summary.trim())
}

/// Fallback when the summariser is unavailable: stub every tool result in
/// `msgs[..cut]`, current turn included, but keep the task and reasoning
/// text. Dropping the prefix instead lost the goal itself in a live run
/// (one 502 from the summariser, and the agent ended with "if the earlier
/// session had a goal it left no trace").
pub fn stub_prefix_native(msgs: &mut [AiMessage], cut: usize) -> usize {
    let end = cut.min(msgs.len());
    stub_old_results_native(&mut msgs[..end], end, 0, 0)
}

pub fn stub_prefix_chat(msgs: &mut [ChatMsg], cut: usize) -> usize {
    let end = cut.min(msgs.len());
    stub_old_results_chat(&mut msgs[..end], end, 0)
}

/// Replace `msgs[..cut]` with one user message carrying `note`. If the
/// kept part starts with a user message, the note is merged into it so
/// the conversation never has two user messages in a row. Returns where
/// `turn_start` (the current turn's task message) moved to.
pub fn splice_native(
    msgs: &mut Vec<AiMessage>,
    cut: usize,
    note: String,
    turn_start: usize,
) -> usize {
    msgs.drain(..cut);
    let inserted = match msgs.first_mut() {
        Some(first) if first.role == AiRole::User => {
            first.content.insert(0, AiContent::text(note));
            0
        }
        _ => {
            msgs.insert(0, AiMessage::user_text(note));
            1
        }
    };
    // A summarised task message survives only as notes; the kept part is
    // then all current turn.
    turn_start.saturating_sub(cut) + if turn_start >= cut { inserted } else { 0 }
}

/// Same for the delimiter conversation, keeping `msgs[0]` (system).
pub fn splice_chat(msgs: &mut Vec<ChatMsg>, cut: usize, note: String, turn_start: usize) -> usize {
    msgs.drain(1..cut);
    let inserted = if msgs.len() > 1 && msgs[1].role == ChatRole::User {
        msgs[1].content = format!("{note}\n\n---\n\n{}", msgs[1].content);
        0
    } else {
        msgs.insert(1, ChatMsg::user(note));
        1
    };
    if turn_start >= cut {
        turn_start - (cut - 1) + inserted
    } else {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result(id: &str, content: &str) -> AiMessage {
        AiMessage {
            role: AiRole::Tool,
            content: vec![AiContent::ToolResult {
                tool_use_id: id.into(),
                content: content.into(),
                is_error: false,
            }],
        }
    }

    fn call(id: &str) -> AiMessage {
        AiMessage {
            role: AiRole::Assistant,
            content: vec![AiContent::ToolUse {
                id: id.into(),
                name: "bash".into(),
                input: serde_json::json!({"command": "ls"}),
                metadata: None,
            }],
        }
    }

    fn session(results: usize, size: usize) -> Vec<AiMessage> {
        let mut v = vec![AiMessage::user_text("fix the bug")];
        for i in 0..results {
            v.push(call(&i.to_string()));
            v.push(result(
                &i.to_string(),
                &format!("[exit 0]\n{}", "o".repeat(size)),
            ));
        }
        v
    }

    fn result_text(m: &AiMessage) -> &str {
        match &m.content[0] {
            AiContent::ToolResult { content, .. } => content,
            _ => panic!("not a result"),
        }
    }

    #[test]
    fn budgets_follow_the_model() {
        let local = Budget::for_model("kotonia-llm-basic", 4096);
        assert_eq!(local.trigger_tokens, 65_536 * 3 / 4 - 4096);
        assert_eq!(local.target_tokens, 32_768);
        let think = Budget::for_model("kotonia-llm-basic:think", 4096);
        assert_eq!(think, local);
        let unknown = Budget::for_model("some-model", 4096);
        assert!(unknown.trigger_tokens < local.trigger_tokens);
    }

    /// Two turns: 6 results in the finished one, 4 in the current one.
    fn two_turns() -> (Vec<AiMessage>, usize) {
        let mut v = session(6, 30_000);
        let turn_start = v.len();
        v.extend(session(4, 30_000));
        (v, turn_start)
    }

    #[test]
    fn stubs_finished_turns_oldest_first() {
        let (mut v, turn_start) = two_turns();
        let before = estimate_native(&v);
        let n = stub_old_results_native(&mut v, turn_start, 0, before / 2);
        assert!(n > 0 && n <= 6);
        assert!(is_stub(result_text(&v[2])), "oldest stubbed");
        assert!(result_text(&v[2]).starts_with("[exit 0]\n"), "header kept");
        assert!(estimate_native(&v) <= before / 2);
    }

    #[test]
    fn never_stubs_the_current_turn() {
        // The loop seen live: stubbing what the model is still working with
        // makes it re-run the command, forever.
        let (mut v, turn_start) = two_turns();
        assert_eq!(stub_old_results_native(&mut v, turn_start, 0, 0), 6);
        for m in &v[turn_start..] {
            if m.role == AiRole::Tool {
                assert!(!is_stub(result_text(m)));
            }
        }
    }

    #[test]
    fn stubbing_stops_at_target_and_is_idempotent() {
        let (mut v, turn_start) = two_turns();
        let target = estimate_native(&v) - 5_000;
        assert_eq!(stub_old_results_native(&mut v, turn_start, 0, target), 1);
        assert_eq!(stub_old_results_native(&mut v, turn_start, 0, 0), 5);
        assert_eq!(stub_old_results_native(&mut v, turn_start, 0, 0), 0);
    }

    #[test]
    fn cut_falls_back_to_the_latest_boundary_when_nothing_fits() {
        let v = session(4, 30_000);
        let c = cut_native(&v, 10).unwrap();
        assert_eq!(c, v.len() - 2, "keep only the last call and its result");
    }

    #[test]
    fn cut_never_separates_a_call_from_its_result() {
        let v = session(10, 3_000);
        for keep in [500, 2_000, 5_000, 20_000] {
            if let Some(c) = cut_native(&v, keep) {
                assert!(c >= 1 && c < v.len());
                assert_ne!(v[c].role, AiRole::Tool, "kept part starts with a result");
            }
        }
    }

    #[test]
    fn splice_merges_into_a_leading_user_message() {
        let mut v = vec![
            AiMessage::user_text("a"),
            call("1"),
            result("1", "x"),
            AiMessage::user_text("next task"),
            call("2"),
        ];
        // Everything fits; the earliest boundary is right after the result.
        let c = cut_native(&v, 10_000).unwrap();
        assert_eq!(c, 3);
        assert_eq!(splice_native(&mut v, c, replacement_text("notes"), 3), 0);
        assert_eq!(v.len(), 2);
        assert_eq!(v[0].role, AiRole::User);
        assert!(v[0].text().contains("notes") && v[0].text().contains("next task"));
    }

    #[test]
    fn splice_inserts_a_user_message_before_an_assistant_one() {
        let mut v = session(3, 10);
        // The task message was summarised away: all of what is kept is the
        // current turn.
        assert_eq!(splice_native(&mut v, 3, replacement_text("notes"), 0), 0);
        assert_eq!(v[0].role, AiRole::User);
        assert!(v[0].text().contains("notes"));
        assert_eq!(v[1].role, AiRole::Assistant);
    }

    #[test]
    fn chat_mode_stubs_and_cuts() {
        let mut v = vec![ChatMsg::system("sys")];
        let mut turn_start = 0;
        for turn in 0..2 {
            turn_start = v.len();
            v.push(ChatMsg::user(format!("task {turn}")));
            for _ in 0..4 {
                v.push(ChatMsg::assistant("<<<BASH>>>ls"));
                v.push(ChatMsg::user(format!("[exit 0]\n{}", "o".repeat(20_000))));
            }
        }
        assert!(stub_old_results_chat(&mut v, turn_start, 0) == 4);
        assert!(is_stub(&v[3].content));
        assert!(!is_stub(&v[v.len() - 1].content), "current turn kept");
        assert!(!is_stub(&v[1].content), "the task is never stubbed");
        let c = cut_chat(&v, estimate_chat(&v) / 2).unwrap();
        let ts = splice_chat(&mut v, c, replacement_text("notes"), turn_start);
        assert!(ts >= 1 && ts < v.len());
        assert_eq!(v[0].role, ChatRole::System);
        assert_eq!(v[1].role, ChatRole::User);
        assert!(v[1].content.starts_with("[Notes"));
    }

    #[test]
    fn fallback_keeps_the_task_and_stubs_outputs() {
        let mut v = session(4, 30_000);
        let cut = cut_native(&v, 10).unwrap();
        assert_eq!(stub_prefix_native(&mut v, cut), 3);
        assert_eq!(v[0].text(), "fix the bug");
        assert!(!is_stub(result_text(&v[v.len() - 1])));
    }

    #[test]
    fn clip_keeps_the_tail_and_says_the_agent_saw_it() {
        let body: String = (0..500).map(|i| format!("def f_{i}(): pass\n")).collect();
        let c = clip(&body, 3_000);
        assert!(c.contains("def f_0()") && c.contains("def f_499()"));
        assert!(c.contains("the agent saw them"));
    }

    #[test]
    fn render_clips_outputs_and_fits() {
        let v = session(5, 10_000);
        let r = render_native(&v, 1_000_000);
        assert!(r.contains("USER: fix the bug"));
        assert!(r.contains("AGENT ran bash"));
        assert!(r.len() < 5 * (SUMMARY_RESULT_MAX_BYTES + 200) + 200);
        let small = render_native(&v, 2_000);
        assert!(small.len() < 2_100 && small.contains("fix the bug"));
    }
}
