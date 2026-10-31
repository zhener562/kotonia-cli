//! kotonia-cli library surface.
//!
//! The binary in `src/main.rs` wires the CLI argument parser, the I/O
//! sinks, and history persistence. Everything reusable — provider clients,
//! the agent loop, the executor, the prompt builder — lives here so it
//! can be exercised by integration tests and (eventually) embedded by
//! the `/chat/studio` web frontend.

pub mod agent;
pub mod ai;
pub mod client_info;
pub mod config;
pub mod daemon;
pub mod execution;
pub mod home;
pub mod login;
pub mod notifier;
pub mod providers;
pub mod serve;
pub mod telemetry;
