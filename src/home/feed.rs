//! Where the resident agent's output lands: an append-only JSONL feed per
//! local day (the record, and what a UI will render later), stdout, and the
//! paired phone notifier when there is one.

use std::io::Write;
use std::sync::Arc;

use chrono::{DateTime, FixedOffset, Utc};
use serde::{Deserialize, Serialize};

use super::workspace::HomeDir;
use crate::notifier::Notifier;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeedKind {
    Heartbeat,
    Reminder,
    StaleReminder,
    /// A command the heartbeat wanted to run but was not allowed to without
    /// a human. Recorded so the user sees what it tried.
    ApprovalNeeded,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedItem {
    pub ts: DateTime<Utc>,
    pub kind: FeedKind,
    pub text: String,
}

pub struct Delivery {
    home: HomeDir,
    notifier: Option<Arc<dyn Notifier>>,
    /// Print to stdout as well (the `home run` foreground UX).
    echo: bool,
}

impl Delivery {
    pub fn new(home: HomeDir, notifier: Option<Arc<dyn Notifier>>, echo: bool) -> Self {
        Self {
            home,
            notifier,
            echo,
        }
    }

    /// Record only — no push. For entries the user should be able to find
    /// later but that are not worth buzzing a phone over.
    pub fn record(&self, item: &FeedItem, offset: FixedOffset) {
        let day = item.ts.with_timezone(&offset).format("%Y-%m-%d");
        let path = self.home.root().join("feed").join(format!("{day}.jsonl"));
        let res = (|| -> std::io::Result<()> {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let mut f = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)?;
            let line = serde_json::to_string(item).map_err(std::io::Error::other)?;
            writeln!(f, "{line}")
        })();
        if let Err(e) = res {
            eprintln!("[home] cannot write feed {}: {e}", path.display());
        }
        if self.echo {
            let local = item.ts.with_timezone(&offset).format("%m-%d %H:%M");
            println!("[{local}] {:?}: {}", item.kind, item.text);
        }
    }

    /// Record and push to the phone.
    pub async fn deliver(&self, item: FeedItem, offset: FixedOffset) {
        self.record(&item, offset);
        if let Some(n) = &self.notifier {
            if let Err(e) = n.send_text(&item.text).await {
                eprintln!("[home] notifier failed: {e}");
            }
        }
    }
}
