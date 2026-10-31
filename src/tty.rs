//! Terminal input helpers for the connector card.

use std::io::{self, BufRead, Write};

/// Read one line from stdin with echo turned off (passwords, API keys).
/// The prompt goes to stderr so stdout stays clean for piping.
pub fn read_hidden_line(prompt: &str) -> io::Result<String> {
    eprint!("{prompt}");
    io::stderr().flush()?;
    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;
        let fd = io::stdin().as_raw_fd();
        // SAFETY: plain termios get/set on stdin; the original settings are
        // restored before returning, including on read errors.
        let mut term: libc::termios = unsafe { std::mem::zeroed() };
        let is_tty = unsafe { libc::tcgetattr(fd, &mut term) } == 0;
        if is_tty {
            let original = term;
            term.c_lflag &= !libc::ECHO;
            term.c_lflag |= libc::ECHONL;
            unsafe { libc::tcsetattr(fd, libc::TCSANOW, &term) };
            let mut line = String::new();
            let res = io::stdin().lock().read_line(&mut line);
            unsafe { libc::tcsetattr(fd, libc::TCSANOW, &original) };
            res?;
            return Ok(trim_newline(line));
        }
    }
    // Not a terminal (or not unix): read normally. Nothing is echoed by us;
    // on Windows consoles the input is visible.
    let mut line = String::new();
    io::stdin().lock().read_line(&mut line)?;
    Ok(trim_newline(line))
}

pub fn read_visible_line(prompt: &str, default: Option<&str>) -> io::Result<String> {
    match default {
        Some(d) if !d.is_empty() => eprint!("{prompt} [{d}]: "),
        _ => eprint!("{prompt}: "),
    }
    io::stderr().flush()?;
    let mut line = String::new();
    io::stdin().lock().read_line(&mut line)?;
    let line = trim_newline(line);
    Ok(match default {
        Some(d) if line.trim().is_empty() => d.to_string(),
        _ => line,
    })
}

fn trim_newline(mut s: String) -> String {
    while s.ends_with('\n') || s.ends_with('\r') {
        s.pop();
    }
    s
}

/// Render a connector card on stderr and collect the fields. Shared by the
/// REPL (agent asked mid-turn) and `kotonia-cli connector add`.
pub fn fill_connector_card(
    card: &crate::connectors::ConnectorCard,
    ask_consent: bool,
) -> io::Result<Option<std::collections::BTreeMap<String, String>>> {
    eprintln!("\n┌─ 接続: {} ─────────────", card.title);
    if !card.reason.trim().is_empty() {
        eprintln!("│ {}", card.reason.trim());
    }
    for line in &card.consent {
        eprintln!("│ ・{line}");
    }
    if let Some(h) = &card.help {
        eprintln!("│ {h}");
    }
    if let Some(u) = &card.help_url {
        eprintln!("│ {u}");
    }
    eprintln!("└──────────────────────");
    if ask_consent {
        let ok = read_visible_line("連携しますか？ [y/N]", None)?;
        if !matches!(ok.trim().to_ascii_lowercase().as_str(), "y" | "yes") {
            return Ok(None);
        }
    }
    let mut values = std::collections::BTreeMap::new();
    for f in &card.fields {
        let v = if f.secret {
            read_hidden_line(&format!("{} (入力は表示されません): ", f.label))?
        } else {
            read_visible_line(&f.label, f.default.as_deref())?
        };
        values.insert(f.key.clone(), v);
    }
    Ok(Some(values))
}
