//! Resident-agent skeleton ("home"): a persistent workspace of Markdown
//! files the agent reads at the start of every conversation, plus a
//! wall-clock schedule (heartbeat + reminders) that keeps working across
//! the machine sleeping.
//!
//! Modelled on Meta's Muse; the research notes and the risks of running
//! resident on a user's PC are in the hage repo,
//! `docs/MUSE_SPEC_RESEARCH_2026-11-01.md`.

pub mod feed;
pub mod heartbeat;
pub mod runner;
pub mod schedule;
pub mod workspace;
