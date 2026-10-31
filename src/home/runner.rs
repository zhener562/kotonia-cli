//! The resident loop: wake every `poll_secs`, ask the schedule what is due
//! *now* (wall clock), run it, save.

use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, FixedOffset, Utc};

use super::feed::{Delivery, FeedItem, FeedKind};
use super::heartbeat;
use super::schedule::{local_offset_now, Fire, Schedule, SleepDetector};
use super::workspace::{HomeDir, HEARTBEAT_FILE, PROACTIVE_FILE};
use crate::agent::agent::{Agent, AgentConfig, ApprovalHandler, ApprovalOutcome, Event, EventSink};
use crate::agent::approval::ApprovalMode;
use crate::agent::history::HistoryStore;
use crate::agent::provider::Provider;
use crate::notifier::Notifier;

#[derive(Debug, Clone)]
pub struct RunnerConfig {
    pub model: String,
    pub provider: Option<String>,
    pub poll_secs: u64,
    /// Cap on one heartbeat's agent run (iterations).
    pub max_iterations: u32,
    /// Wall-time cap on one heartbeat's agent run.
    pub heartbeat_timeout: Duration,
    /// Evaluate the schedule and print what would fire, without running the
    /// model, delivering, or advancing the schedule.
    pub dry_run: bool,
}

/// Holds an exclusive lock on `<home>/home.lock` for the life of the runner,
/// so two `home run` processes on one machine cannot both fire the same tick.
/// (Two machines are the server's job — see the design doc.)
pub struct RunnerLock {
    #[allow(dead_code)]
    file: std::fs::File,
}

impl RunnerLock {
    pub fn acquire(home: &HomeDir) -> Result<Self, String> {
        let path = home.root().join("home.lock");
        let file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&path)
            .map_err(|e| format!("open {}: {e}", path.display()))?;
        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            // SAFETY: flock on a valid, owned fd; released when `file` drops.
            let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
            if rc != 0 {
                return Err(format!(
                    "another `kotonia-cli home run` already holds {}",
                    path.display()
                ));
            }
        }
        Ok(Self { file })
    }
}

/// Approval handler for unattended runs: never approves. Every command the
/// policy wanted a human for is remembered and surfaced in the feed.
#[derive(Default)]
struct HeadlessApproval {
    refused: Vec<String>,
}

impl ApprovalHandler for HeadlessApproval {
    fn ask(&mut self, command: &str, reason: &str) -> ApprovalOutcome {
        self.refused.push(format!("{command}  ({reason})"));
        ApprovalOutcome::Deny
    }
}

struct QuietSink;

impl EventSink for QuietSink {
    fn emit(&mut self, _event: Event) {}
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct TickReport {
    pub fired: usize,
    pub heartbeat_skipped_empty: bool,
}

/// One evaluation of the schedule. Separate from the loop so `home tick`
/// can run it once (cron-style, or for debugging).
pub async fn tick(
    home: &HomeDir,
    cfg: &RunnerConfig,
    delivery: &Delivery,
    now: DateTime<Utc>,
    offset: FixedOffset,
) -> Result<TickReport, String> {
    let mut schedule = Schedule::load(home.root())?;
    let fires = schedule.collect_due(now, offset);
    let mut report = TickReport {
        fired: fires.len(),
        ..Default::default()
    };
    if cfg.dry_run {
        for f in &fires {
            println!("would fire: {f:?}");
        }
        if fires.is_empty() {
            println!("nothing due");
        }
        return Ok(report);
    }
    // Save before running anything: a crash or a long heartbeat must not
    // make the next tick fire the same jobs again.
    schedule.save(home.root())?;

    for fire in fires {
        match fire {
            Fire::Reminder {
                text, late_by_secs, ..
            } => {
                let text = if late_by_secs >= 600 {
                    format!(
                        "⏰ {text}\n(予定より {} 分遅れ。PC のスリープ中か、常駐が止まっていました)",
                        late_by_secs / 60
                    )
                } else {
                    format!("⏰ {text}")
                };
                delivery
                    .deliver(
                        FeedItem {
                            ts: now,
                            kind: FeedKind::Reminder,
                            text,
                        },
                        offset,
                    )
                    .await;
            }
            Fire::StaleReminder {
                text, late_by_secs, ..
            } => {
                // Recorded, not pushed: a 3am buzz for a reminder that was due
                // at 9pm yesterday helps nobody.
                delivery.record(
                    &FeedItem {
                        ts: now,
                        kind: FeedKind::StaleReminder,
                        text: format!(
                            "{text}\n(予定から {} 時間過ぎていたので通知しませんでした)",
                            late_by_secs / 3600
                        ),
                    },
                    offset,
                );
            }
            Fire::Heartbeat { missed, .. } => {
                let checks = heartbeat::checklist(&home.read(HEARTBEAT_FILE).unwrap_or_default());
                if checks.is_empty() {
                    report.heartbeat_skipped_empty = true;
                    continue;
                }
                run_heartbeat(home, cfg, delivery, &checks, now, offset, missed).await;
            }
        }
    }
    Ok(report)
}

async fn run_heartbeat(
    home: &HomeDir,
    cfg: &RunnerConfig,
    delivery: &Delivery,
    checks: &[String],
    now: DateTime<Utc>,
    offset: FixedOffset,
    missed: u32,
) {
    let fail = |text: String| FeedItem {
        ts: now,
        kind: FeedKind::Error,
        text,
    };
    let provider = match Provider::resolve(cfg.provider.as_deref(), &cfg.model) {
        Ok(p) => p,
        Err(e) => {
            delivery.record(
                &fail(format!("heartbeat: cannot use model `{}`: {e}", cfg.model)),
                offset,
            );
            return;
        }
    };
    // Allowlist, never auto: read-only commands run, anything that changes
    // the world waits for a human who is not here.
    let mut config = AgentConfig::new(ApprovalMode::Allowlist, true);
    config.max_iterations = cfg.max_iterations;
    // The tool is advertised so the model knows connecting exists, but the
    // headless handler answers "nobody here" — a heartbeat never prompts.
    config.connectors = true;
    let exe = std::env::current_exe()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|_| "kotonia-cli".to_string());
    let mut prefix = home.prompt_block(
        super::workspace::DEFAULT_MAX_FILE_CHARS,
        super::workspace::DEFAULT_MAX_TOTAL_CHARS,
    );
    prefix.push_str(&crate::connectors::prompt_section(&exe));
    config.persona_prefix = Some(prefix);
    config.kotonia_api_base = std::env::var("KOTONIA_API_KEY").ok().map(|_| {
        std::env::var("KOTONIA_API_BASE").unwrap_or_else(|_| "https://kotonia.ai".to_string())
    });
    let mut agent = Agent::new(home.root(), provider, config);
    let session_id = format!(
        "home-heartbeat-{}",
        now.with_timezone(&offset).format("%Y%m%d-%H%M%S")
    );
    if let Ok(mut store) = HistoryStore::open_in(home.root().join("runs"), &session_id) {
        let _ = store.write_header(
            agent.model_id(),
            agent.backend_label(),
            &ApprovalMode::Allowlist.to_string(),
            home.root(),
            true,
        );
        agent = agent.with_history(store);
        agent.log_initial_system();
    }

    let preferences = home.read(PROACTIVE_FILE);
    let prompt = heartbeat::build_prompt(checks, preferences.as_deref(), now, offset, missed);
    let mut approval = HeadlessApproval::default();
    let mut sink = QuietSink;
    let result = tokio::time::timeout(
        cfg.heartbeat_timeout,
        agent.run_turn(&prompt, &mut approval, &mut sink),
    )
    .await;

    for refused in approval.refused {
        delivery.record(
            &FeedItem {
                ts: now,
                kind: FeedKind::ApprovalNeeded,
                text: refused,
            },
            offset,
        );
    }

    match result {
        Ok(Ok(answer)) if heartbeat::is_quiet(&answer) => {
            eprintln!("[home] heartbeat: nothing to report ({session_id})");
        }
        Ok(Ok(answer)) => {
            delivery
                .deliver(
                    FeedItem {
                        ts: now,
                        kind: FeedKind::Heartbeat,
                        text: answer.trim().to_string(),
                    },
                    offset,
                )
                .await;
        }
        Ok(Err(e)) => delivery.record(
            &fail(format!("heartbeat failed: {e} ({session_id})")),
            offset,
        ),
        Err(_) => delivery.record(
            &fail(format!(
                "heartbeat timed out after {}s ({session_id})",
                cfg.heartbeat_timeout.as_secs()
            )),
            offset,
        ),
    }
}

/// Run until killed.
pub async fn run_forever(
    home: HomeDir,
    cfg: RunnerConfig,
    notifier: Option<Arc<dyn Notifier>>,
) -> Result<(), String> {
    let _lock = RunnerLock::acquire(&home)?;
    let delivery = Delivery::new(home.clone(), notifier, true);
    let mut sleep = SleepDetector::new();
    eprintln!(
        "[home] running in {} (poll every {}s, model {})",
        home.root().display(),
        cfg.poll_secs,
        cfg.model
    );
    loop {
        if let Some(gap) = sleep.observe(120) {
            eprintln!("[home] woke up after ~{} min asleep", gap / 60);
        }
        let now = Utc::now();
        let offset = local_offset_now();
        if let Err(e) = tick(&home, &cfg, &delivery, now, offset).await {
            eprintln!("[home] tick failed: {e}");
        }
        tokio::time::sleep(Duration::from_secs(cfg.poll_secs.max(5))).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn cfg() -> RunnerConfig {
        RunnerConfig {
            // Unresolvable on purpose: the tests must never reach a model.
            model: "no-such-model-for-tests".into(),
            provider: Some("no-such-provider".into()),
            poll_secs: 60,
            max_iterations: 1,
            heartbeat_timeout: Duration::from_secs(1),
            dry_run: false,
        }
    }

    fn jst() -> FixedOffset {
        FixedOffset::east_opt(9 * 3600).unwrap()
    }

    #[tokio::test]
    async fn empty_checklist_never_calls_the_model() {
        let dir = tempfile::tempdir().unwrap();
        let home = HomeDir::new(dir.path());
        home.ensure_initialized().unwrap();
        let now = jst()
            .with_ymd_and_hms(2026, 11, 2, 10, 0, 0)
            .unwrap()
            .with_timezone(&Utc);
        let mut s = Schedule::default();
        s.set_heartbeat(1800, now - chrono::Duration::hours(1));
        s.save(home.root()).unwrap();

        let delivery = Delivery::new(home.clone(), None, false);
        let report = tick(&home, &cfg(), &delivery, now, jst()).await.unwrap();
        assert_eq!(report.fired, 1);
        assert!(report.heartbeat_skipped_empty);
        // Had it tried the (bogus) model, an Error item would be in the feed.
        assert!(!dir.path().join("feed/2026-11-02.jsonl").exists());
    }

    #[tokio::test]
    async fn reminders_land_in_the_feed_and_the_schedule_advances() {
        let dir = tempfile::tempdir().unwrap();
        let home = HomeDir::new(dir.path());
        home.ensure_initialized().unwrap();
        let now = jst()
            .with_ymd_and_hms(2026, 11, 2, 10, 0, 0)
            .unwrap()
            .with_timezone(&Utc);
        let mut s = Schedule::default();
        s.add_reminder(now - chrono::Duration::minutes(30), "請求書".into(), now);
        s.add_reminder(now - chrono::Duration::hours(8), "昨日の件".into(), now);
        s.save(home.root()).unwrap();

        let delivery = Delivery::new(home.clone(), None, false);
        tick(&home, &cfg(), &delivery, now, jst()).await.unwrap();
        let feed = std::fs::read_to_string(dir.path().join("feed/2026-11-02.jsonl")).unwrap();
        let lines: Vec<FeedItem> = feed
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(lines.len(), 2);
        assert!(lines
            .iter()
            .any(|i| i.kind == FeedKind::Reminder && i.text.contains("30 分遅れ")));
        assert!(lines.iter().any(|i| i.kind == FeedKind::StaleReminder));
        assert!(Schedule::load(home.root()).unwrap().jobs.is_empty());

        // Running the same instant again delivers nothing new.
        tick(&home, &cfg(), &delivery, now, jst()).await.unwrap();
        let again = std::fs::read_to_string(dir.path().join("feed/2026-11-02.jsonl")).unwrap();
        assert_eq!(again.lines().count(), 2);
    }

    #[tokio::test]
    async fn dry_run_does_not_advance_the_schedule() {
        let dir = tempfile::tempdir().unwrap();
        let home = HomeDir::new(dir.path());
        home.ensure_initialized().unwrap();
        let now = Utc::now();
        let mut s = Schedule::default();
        s.add_reminder(now - chrono::Duration::minutes(1), "x".into(), now);
        s.save(home.root()).unwrap();
        let delivery = Delivery::new(home.clone(), None, false);
        let mut c = cfg();
        c.dry_run = true;
        tick(&home, &c, &delivery, now, jst()).await.unwrap();
        assert_eq!(Schedule::load(home.root()).unwrap().jobs.len(), 1);
    }

    #[cfg(unix)]
    #[test]
    fn second_runner_on_the_same_home_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let home = HomeDir::new(dir.path());
        home.ensure_initialized().unwrap();
        let _first = RunnerLock::acquire(&home).unwrap();
        assert!(RunnerLock::acquire(&home).is_err());
    }
}
