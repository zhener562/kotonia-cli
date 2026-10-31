//! Wall-clock schedule for the resident agent: the heartbeat and one-off
//! reminders.
//!
//! The machine this runs on sleeps. Two consequences shape everything here:
//!
//! 1. **Due times are wall-clock UTC instants, checked by polling.** A
//!    `tokio::time::sleep(30min)` is measured on a clock that stops while the
//!    machine is suspended (Linux CLOCK_MONOTONIC, macOS CLOCK_UPTIME_RAW), so
//!    it would fire 30 minutes of *awake* time later — hours late after a
//!    night's sleep. The runner instead wakes every minute and asks
//!    [`Schedule::collect_due`] what is past due *now*.
//! 2. **Missed ticks collapse into one.** After waking from an eight-hour
//!    sleep, sixteen overdue heartbeats fire once (with `missed = 15`), not
//!    sixteen times. A reminder that is too far past its time is reported as
//!    stale instead of being delivered as if it were current.
//!
//! Everything here is pure (the caller passes `now` and the local offset) so
//! the sleep/wake behaviour is unit-tested without sleeping.

use std::path::Path;

use chrono::{DateTime, Duration, FixedOffset, NaiveTime, TimeZone, Utc};
use serde::{Deserialize, Serialize};

pub const SCHEDULE_FILE: &str = "schedule.json";

/// A reminder this far past its time is not delivered as if it were current.
pub const DEFAULT_REMINDER_STALE_SECS: i64 = 6 * 3600;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum JobKind {
    /// Runs the checklist in HEARTBEAT.md every `every_secs`, inside the
    /// active hours only.
    Heartbeat { every_secs: i64 },
    /// Delivers `text` once at `next_due`. Fires regardless of active hours —
    /// the user picked the time explicitly.
    Reminder {
        text: String,
        #[serde(default = "default_stale")]
        stale_after_secs: i64,
    },
}

fn default_stale() -> i64 {
    DEFAULT_REMINDER_STALE_SECS
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Job {
    pub id: String,
    #[serde(flatten)]
    pub kind: JobKind,
    pub next_due: DateTime<Utc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_run: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
}

/// Local window in which the heartbeat may reach out. `start > end` wraps
/// past midnight (e.g. 22:00–06:00).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActiveHours {
    pub start: NaiveTime,
    pub end: NaiveTime,
}

impl Default for ActiveHours {
    /// Muse's default: 09:00–21:30.
    fn default() -> Self {
        Self {
            start: NaiveTime::from_hms_opt(9, 0, 0).unwrap(),
            end: NaiveTime::from_hms_opt(21, 30, 0).unwrap(),
        }
    }
}

impl ActiveHours {
    /// Parse `HH:MM-HH:MM`.
    pub fn parse(s: &str) -> Option<Self> {
        let (a, b) = s.split_once('-')?;
        let start = NaiveTime::parse_from_str(a.trim(), "%H:%M").ok()?;
        let end = NaiveTime::parse_from_str(b.trim(), "%H:%M").ok()?;
        (start != end).then_some(Self { start, end })
    }

    pub fn contains(&self, local: NaiveTime) -> bool {
        if self.start < self.end {
            local >= self.start && local < self.end
        } else {
            local >= self.start || local < self.end
        }
    }

    /// The next instant (UTC) at which the window opens, strictly after `now`
    /// unless `now` is already inside — callers check `contains` first.
    pub fn next_open(&self, now: DateTime<Utc>, offset: FixedOffset) -> DateTime<Utc> {
        let local = now.with_timezone(&offset);
        let today = local.date_naive();
        let mut candidate = offset
            .from_local_datetime(&today.and_time(self.start))
            .single()
            .expect("fixed offsets are unambiguous")
            .with_timezone(&Utc);
        if candidate <= now {
            candidate += Duration::days(1);
        }
        candidate
    }
}

/// What [`Schedule::collect_due`] decided should happen now.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Fire {
    Heartbeat {
        id: String,
        /// Ticks that fell due while the machine was asleep or the runner was
        /// down, folded into this one.
        missed: u32,
    },
    Reminder {
        id: String,
        text: String,
        late_by_secs: i64,
    },
    /// Too late to deliver as if current; the runner says so instead.
    StaleReminder {
        id: String,
        text: String,
        late_by_secs: i64,
    },
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Schedule {
    #[serde(default)]
    pub active_hours: Option<ActiveHours>,
    #[serde(default)]
    pub jobs: Vec<Job>,
}

impl Schedule {
    pub fn load(dir: &Path) -> Result<Self, String> {
        let path = dir.join(SCHEDULE_FILE);
        match std::fs::read_to_string(&path) {
            Ok(raw) => serde_json::from_str(&raw).map_err(|e| format!("{}: {e}", path.display())),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(format!("read {}: {e}", path.display())),
        }
    }

    /// Write via a temp file + rename so a crash or a second process never
    /// sees half a file.
    pub fn save(&self, dir: &Path) -> Result<(), String> {
        let path = dir.join(SCHEDULE_FILE);
        let tmp = dir.join(format!(".{SCHEDULE_FILE}.tmp"));
        let json = serde_json::to_string_pretty(self).map_err(|e| format!("serialize: {e}"))?;
        std::fs::write(&tmp, json).map_err(|e| format!("write {}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, &path).map_err(|e| format!("rename {}: {e}", path.display()))
    }

    pub fn active_hours(&self) -> ActiveHours {
        self.active_hours.unwrap_or_default()
    }

    pub fn heartbeat(&self) -> Option<&Job> {
        self.jobs
            .iter()
            .find(|j| matches!(j.kind, JobKind::Heartbeat { .. }))
    }

    /// Turn the heartbeat on (or change its interval). There is only ever one.
    pub fn set_heartbeat(&mut self, every_secs: i64, now: DateTime<Utc>) {
        let every_secs = every_secs.max(60);
        if let Some(job) = self
            .jobs
            .iter_mut()
            .find(|j| matches!(j.kind, JobKind::Heartbeat { .. }))
        {
            job.kind = JobKind::Heartbeat { every_secs };
            job.next_due = now + Duration::seconds(every_secs);
            return;
        }
        self.jobs.push(Job {
            id: "heartbeat".to_string(),
            kind: JobKind::Heartbeat { every_secs },
            next_due: now + Duration::seconds(every_secs),
            last_run: None,
            created_at: now,
        });
    }

    pub fn clear_heartbeat(&mut self) -> bool {
        let before = self.jobs.len();
        self.jobs
            .retain(|j| !matches!(j.kind, JobKind::Heartbeat { .. }));
        before != self.jobs.len()
    }

    pub fn add_reminder(&mut self, at: DateTime<Utc>, text: String, now: DateTime<Utc>) -> String {
        let id = format!("r-{}", &uuid::Uuid::new_v4().simple().to_string()[..8]);
        self.jobs.push(Job {
            id: id.clone(),
            kind: JobKind::Reminder {
                text,
                stale_after_secs: DEFAULT_REMINDER_STALE_SECS,
            },
            next_due: at,
            last_run: None,
            created_at: now,
        });
        id
    }

    pub fn remove(&mut self, id: &str) -> bool {
        let before = self.jobs.len();
        self.jobs.retain(|j| j.id != id);
        before != self.jobs.len()
    }

    /// Decide what is due at `now` and advance the schedule accordingly. The
    /// caller saves the schedule afterwards; running the same `now` twice
    /// fires nothing the second time.
    pub fn collect_due(&mut self, now: DateTime<Utc>, offset: FixedOffset) -> Vec<Fire> {
        let hours = self.active_hours();
        let mut fires = Vec::new();
        let mut done = Vec::new();
        for job in self.jobs.iter_mut() {
            if job.next_due > now {
                continue;
            }
            let late = (now - job.next_due).num_seconds();
            match &job.kind {
                JobKind::Heartbeat { every_secs } => {
                    let every = (*every_secs).max(60);
                    let local = now.with_timezone(&offset).time();
                    if !hours.contains(local) {
                        // Do not fire, and do not count the night as missed:
                        // waiting for the window is the intended behaviour.
                        job.next_due = hours.next_open(now, offset);
                        continue;
                    }
                    let missed = (late / every).min(u32::MAX as i64) as u32;
                    fires.push(Fire::Heartbeat {
                        id: job.id.clone(),
                        missed,
                    });
                    job.last_run = Some(now);
                    job.next_due = now + Duration::seconds(every);
                }
                JobKind::Reminder {
                    text,
                    stale_after_secs,
                } => {
                    let fire = if late > *stale_after_secs {
                        Fire::StaleReminder {
                            id: job.id.clone(),
                            text: text.clone(),
                            late_by_secs: late,
                        }
                    } else {
                        Fire::Reminder {
                            id: job.id.clone(),
                            text: text.clone(),
                            late_by_secs: late,
                        }
                    };
                    fires.push(fire);
                    done.push(job.id.clone());
                }
            }
        }
        self.jobs.retain(|j| !done.contains(&j.id));
        fires
    }
}

/// Parse a user-supplied reminder time:
/// - `+30m`, `+2h`, `+1d` — relative to now
/// - `HH:MM` — the next occurrence in local time (today, else tomorrow)
/// - `YYYY-MM-DD HH:MM` — local time
/// - RFC 3339 — as given
pub fn parse_when(s: &str, now: DateTime<Utc>, offset: FixedOffset) -> Option<DateTime<Utc>> {
    let s = s.trim();
    if let Some(rel) = s.strip_prefix('+') {
        let (num, unit) = rel.split_at(rel.len().checked_sub(1)?);
        let n: i64 = num.parse().ok()?;
        let d = match unit {
            "m" => Duration::minutes(n),
            "h" => Duration::hours(n),
            "d" => Duration::days(n),
            _ => return None,
        };
        return (n > 0).then(|| now + d);
    }
    if let Ok(t) = NaiveTime::parse_from_str(s, "%H:%M") {
        let local_today = now.with_timezone(&offset).date_naive();
        let mut at = offset
            .from_local_datetime(&local_today.and_time(t))
            .single()?
            .with_timezone(&Utc);
        if at <= now {
            at += Duration::days(1);
        }
        return Some(at);
    }
    if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M") {
        return Some(
            offset
                .from_local_datetime(&dt)
                .single()?
                .with_timezone(&Utc),
        );
    }
    DateTime::parse_from_rfc3339(s)
        .ok()
        .map(|d| d.with_timezone(&Utc))
}

/// The local UTC offset right now. Taken fresh on every tick so a DST change
/// or a laptop that travelled is picked up without a restart.
pub fn local_offset_now() -> FixedOffset {
    *chrono::Local::now().offset()
}

/// Notices when the machine slept: the wall clock jumps ahead while the
/// monotonic clock does not. Used for logging only — scheduling is already
/// correct across sleep because it is wall-clock based.
#[derive(Debug)]
pub struct SleepDetector {
    last_wall: DateTime<Utc>,
    last_mono: std::time::Instant,
}

impl SleepDetector {
    pub fn new() -> Self {
        Self {
            last_wall: Utc::now(),
            last_mono: std::time::Instant::now(),
        }
    }

    /// Returns the seconds of wall time that passed without monotonic time
    /// (≈ how long the machine slept), when that exceeds `threshold_secs`.
    pub fn observe(&mut self, threshold_secs: i64) -> Option<i64> {
        let wall = Utc::now();
        let mono = std::time::Instant::now();
        let gap = sleep_gap(
            (wall - self.last_wall).num_seconds(),
            mono.duration_since(self.last_mono).as_secs() as i64,
            threshold_secs,
        );
        self.last_wall = wall;
        self.last_mono = mono;
        gap
    }
}

impl Default for SleepDetector {
    fn default() -> Self {
        Self::new()
    }
}

fn sleep_gap(wall_secs: i64, mono_secs: i64, threshold_secs: i64) -> Option<i64> {
    let gap = wall_secs - mono_secs;
    (gap > threshold_secs).then_some(gap)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn jst() -> FixedOffset {
        FixedOffset::east_opt(9 * 3600).unwrap()
    }

    /// A UTC instant given as JST wall time.
    fn at(h: u32, m: u32) -> DateTime<Utc> {
        jst()
            .with_ymd_and_hms(2026, 11, 2, h, m, 0)
            .unwrap()
            .with_timezone(&Utc)
    }

    #[test]
    fn heartbeat_fires_once_after_a_long_sleep() {
        let mut s = Schedule::default();
        s.set_heartbeat(1800, at(9, 0)); // due 09:30
                                         // Machine slept 09:10 → 13:00: seven ticks fell due.
        let fires = s.collect_due(at(13, 0), jst());
        assert_eq!(
            fires,
            vec![Fire::Heartbeat {
                id: "heartbeat".into(),
                missed: 7
            }]
        );
        assert_eq!(s.heartbeat().unwrap().next_due, at(13, 30));
        // Same instant again: nothing.
        assert!(s.collect_due(at(13, 0), jst()).is_empty());
    }

    #[test]
    fn heartbeat_waits_for_the_morning_without_counting_the_night() {
        let mut s = Schedule::default();
        s.set_heartbeat(1800, at(21, 20)); // due 21:50, after the 21:30 close
        assert!(s.collect_due(at(21, 50), jst()).is_empty());
        let next = s.heartbeat().unwrap().next_due;
        assert_eq!(next, at(9, 0) + Duration::days(1));
        let fires = s.collect_due(next, jst());
        assert_eq!(
            fires,
            vec![Fire::Heartbeat {
                id: "heartbeat".into(),
                missed: 0
            }]
        );
    }

    #[test]
    fn reminders_fire_late_then_go_stale() {
        let mut s = Schedule::default();
        s.add_reminder(at(9, 0), "invoice".into(), at(8, 0));
        s.add_reminder(at(9, 0), "call".into(), at(8, 0));
        // 2h late: delivered (late). Reminders ignore active hours.
        let mut s2 = s.clone();
        let fires = s2.collect_due(at(11, 0), jst());
        assert!(matches!(
            &fires[0],
            Fire::Reminder {
                late_by_secs: 7200,
                ..
            }
        ));
        assert!(s2.jobs.is_empty());
        // 7h late: stale.
        let fires = s.collect_due(at(16, 0), jst());
        assert_eq!(fires.len(), 2);
        assert!(fires
            .iter()
            .all(|f| matches!(f, Fire::StaleReminder { .. })));
        assert!(s.jobs.is_empty());
    }

    #[test]
    fn active_hours_wrap_past_midnight() {
        let h = ActiveHours::parse("22:00-06:00").unwrap();
        assert!(h.contains(NaiveTime::from_hms_opt(23, 0, 0).unwrap()));
        assert!(h.contains(NaiveTime::from_hms_opt(5, 59, 0).unwrap()));
        assert!(!h.contains(NaiveTime::from_hms_opt(12, 0, 0).unwrap()));
        assert!(ActiveHours::parse("09:00-09:00").is_none());
    }

    #[test]
    fn parse_when_forms() {
        let now = at(10, 0);
        assert_eq!(parse_when("+30m", now, jst()), Some(at(10, 30)));
        assert_eq!(parse_when("15:00", now, jst()), Some(at(15, 0)));
        // Earlier today → tomorrow.
        assert_eq!(
            parse_when("09:00", now, jst()),
            Some(at(9, 0) + Duration::days(1))
        );
        assert_eq!(parse_when("2026-11-02 18:00", now, jst()), Some(at(18, 0)));
        assert_eq!(
            parse_when("2026-11-02T09:00:00Z", now, jst()),
            Some(Utc.with_ymd_and_hms(2026, 11, 2, 9, 0, 0).unwrap())
        );
        assert_eq!(parse_when("+0m", now, jst()), None);
        assert_eq!(parse_when("soon", now, jst()), None);
    }

    #[test]
    fn schedule_round_trips_on_disk() {
        let dir = tempfile::tempdir().unwrap();
        let mut s = Schedule::default();
        s.set_heartbeat(1800, at(9, 0));
        s.add_reminder(at(12, 0), "lunch".into(), at(9, 0));
        s.active_hours = ActiveHours::parse("08:00-22:00");
        s.save(dir.path()).unwrap();
        assert_eq!(Schedule::load(dir.path()).unwrap(), s);
        assert_eq!(
            Schedule::load(tempfile::tempdir().unwrap().path()).unwrap(),
            Schedule::default()
        );
    }

    #[test]
    fn sleep_gap_needs_wall_to_outrun_monotonic() {
        assert_eq!(sleep_gap(60, 60, 120), None);
        assert_eq!(sleep_gap(3600, 60, 120), Some(3540));
    }
}
