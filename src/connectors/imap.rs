//! A deliberately small, read-only IMAP client: LOGIN, EXAMINE (never
//! SELECT, so nothing can change flags), UID SEARCH, UID FETCH with
//! BODY.PEEK (never marks mail as read), LOGOUT. Over implicit TLS (993).
//!
//! MIME and charsets (ISO-2022-JP and friends are still common in Japanese
//! mail) are left to `mail-parser`.

use std::sync::Arc;
use std::time::Duration;

use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;
use tokio_rustls::client::TlsStream;
use tokio_rustls::rustls::{ClientConfig, RootCertStore};
use tokio_rustls::TlsConnector;

const IO_TIMEOUT: Duration = Duration::from_secs(30);
/// Cap on a single literal (one message). Partial fetches ask for less;
/// this guards against a hostile server.
const MAX_LITERAL: usize = 8 * 1024 * 1024;

#[derive(Debug)]
pub enum ImapError {
    Connect(String),
    /// The server rejected the credentials.
    Auth(String),
    Protocol(String),
}

impl std::fmt::Display for ImapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ImapError::Connect(m) => write!(f, "cannot connect: {m}"),
            ImapError::Auth(m) => write!(f, "login rejected: {m}"),
            ImapError::Protocol(m) => write!(f, "imap: {m}"),
        }
    }
}

impl std::error::Error for ImapError {}

/// One untagged response line, with any `{n}` literals pulled out.
#[derive(Debug, Default, Clone)]
pub struct Untagged {
    pub text: String,
    pub literals: Vec<Vec<u8>>,
}

#[derive(Debug, Clone, Default)]
pub struct MailboxInfo {
    pub exists: u32,
    pub uid_validity: Option<u32>,
    pub uid_next: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct FetchedMessage {
    pub uid: u32,
    pub seen: bool,
    pub raw: Vec<u8>,
}

/// Generic over the transport so the protocol can be tested against a
/// scripted in-memory server; production always uses TLS.
pub struct ImapSession<S = TlsStream<TcpStream>> {
    stream: BufReader<S>,
    tag: u32,
}

impl ImapSession<TlsStream<TcpStream>> {
    pub async fn connect(host: &str, port: u16) -> Result<Self, ImapError> {
        let mut roots = RootCertStore::empty();
        roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
        let config = ClientConfig::builder()
            .with_root_certificates(roots)
            .with_no_client_auth();
        let connector = TlsConnector::from(Arc::new(config));
        let tcp = tokio::time::timeout(IO_TIMEOUT, TcpStream::connect((host, port)))
            .await
            .map_err(|_| ImapError::Connect(format!("{host}:{port} timed out")))?
            .map_err(|e| ImapError::Connect(format!("{host}:{port}: {e}")))?;
        let server_name = host
            .to_string()
            .try_into()
            .map_err(|_| ImapError::Connect(format!("invalid host name `{host}`")))?;
        let tls = tokio::time::timeout(IO_TIMEOUT, connector.connect(server_name, tcp))
            .await
            .map_err(|_| ImapError::Connect("TLS handshake timed out".into()))?
            .map_err(|e| ImapError::Connect(format!("TLS: {e}")))?;
        ImapSession::over(tls).await
    }
}

impl<S: AsyncRead + AsyncWrite + Unpin> ImapSession<S> {
    /// Start a session over an already-connected stream (reads the greeting).
    pub async fn over(stream: S) -> Result<Self, ImapError> {
        let mut session = Self {
            stream: BufReader::new(stream),
            tag: 0,
        };
        let greeting = session.read_line().await?;
        if !greeting.starts_with("* OK") && !greeting.starts_with("* PREAUTH") {
            return Err(ImapError::Protocol(format!(
                "unexpected greeting: {greeting}"
            )));
        }
        Ok(session)
    }

    pub async fn login(&mut self, user: &str, password: &str) -> Result<(), ImapError> {
        let cmd = format!("LOGIN {} {}", quote(user), quote(password));
        match self.command(&cmd).await {
            Ok(_) => Ok(()),
            // Never echo the command: it carries the password.
            Err(ImapError::Protocol(m)) => Err(ImapError::Auth(m)),
            Err(e) => Err(e),
        }
    }

    /// Open a mailbox read-only.
    pub async fn examine(&mut self, mailbox: &str) -> Result<MailboxInfo, ImapError> {
        let lines = self.command(&format!("EXAMINE {}", quote(mailbox))).await?;
        let mut info = MailboxInfo::default();
        for u in &lines {
            let t = &u.text;
            if let Some(n) = t.strip_prefix("* ").and_then(|r| r.strip_suffix(" EXISTS")) {
                info.exists = n.trim().parse().unwrap_or(0);
            }
            if let Some(v) = bracket_number(t, "UIDVALIDITY") {
                info.uid_validity = Some(v);
            }
            if let Some(v) = bracket_number(t, "UIDNEXT") {
                info.uid_next = Some(v);
            }
        }
        Ok(info)
    }

    /// `criteria` is raw IMAP search syntax, e.g. `SINCE 01-Nov-2026` or
    /// `UID 120:*`. Callers build it from typed values, never from model text.
    pub async fn uid_search(&mut self, criteria: &str) -> Result<Vec<u32>, ImapError> {
        let lines = self.command(&format!("UID SEARCH {criteria}")).await?;
        let mut uids = Vec::new();
        for u in &lines {
            if let Some(rest) = u.text.strip_prefix("* SEARCH") {
                uids.extend(
                    rest.split_whitespace()
                        .filter_map(|n| n.parse::<u32>().ok()),
                );
            }
        }
        uids.sort_unstable();
        uids.dedup();
        Ok(uids)
    }

    /// Fetch up to `max_bytes` of each message without marking it read.
    pub async fn uid_fetch(
        &mut self,
        uids: &[u32],
        max_bytes: usize,
    ) -> Result<Vec<FetchedMessage>, ImapError> {
        if uids.is_empty() {
            return Ok(Vec::new());
        }
        let set = uids
            .iter()
            .map(|u| u.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let lines = self
            .command(&format!(
                "UID FETCH {set} (UID FLAGS BODY.PEEK[]<0.{max_bytes}>)"
            ))
            .await?;
        let mut out = Vec::new();
        for u in lines {
            if !u.text.contains(" FETCH ") {
                continue;
            }
            let Some(uid) = word_number_after(&u.text, "UID") else {
                continue;
            };
            let seen = u.text.contains("\\Seen");
            let raw = u.literals.into_iter().next().unwrap_or_default();
            out.push(FetchedMessage { uid, seen, raw });
        }
        out.sort_by_key(|m| m.uid);
        Ok(out)
    }

    pub async fn logout(mut self) {
        let _ = self.command("LOGOUT").await;
    }

    /// Find the mailbox flagged `\All` (Gmail's "All Mail", whose visible
    /// name is localised — e.g. `[Gmail]/すべてのメール` in modified UTF-7).
    pub async fn all_mail_mailbox(&mut self) -> Result<Option<String>, ImapError> {
        let lines = self.command("LIST \"\" \"*\"").await?;
        Ok(lines
            .iter()
            .find(|u| u.text.starts_with("* LIST") && u.text.contains("\\All"))
            .and_then(|u| list_mailbox_name(&u.text)))
    }

    /// Full-text search with a UTF-8 query sent as a literal (Japanese
    /// cannot go in a quoted string). On Gmail this is `X-GM-RAW`, i.e. the
    /// same query language as the Gmail search box; elsewhere `TEXT`.
    pub async fn uid_search_query(
        &mut self,
        query: &str,
        gmail: bool,
    ) -> Result<Vec<u32>, ImapError> {
        let key = if gmail { "X-GM-RAW" } else { "TEXT" };
        let lines = self
            .command_literal(
                &format!("UID SEARCH CHARSET UTF-8 {key}"),
                query.as_bytes(),
                "",
            )
            .await?;
        let mut uids = Vec::new();
        for u in &lines {
            if let Some(rest) = u.text.strip_prefix("* SEARCH") {
                uids.extend(
                    rest.split_whitespace()
                        .filter_map(|n| n.parse::<u32>().ok()),
                );
            }
        }
        uids.sort_unstable();
        uids.dedup();
        Ok(uids)
    }

    /// A command with one synchronising literal: send `prefix {n}`, wait for
    /// the server's `+` go-ahead, then the bytes and the rest of the line.
    async fn command_literal(
        &mut self,
        prefix: &str,
        literal: &[u8],
        suffix: &str,
    ) -> Result<Vec<Untagged>, ImapError> {
        self.tag += 1;
        let tag = format!("k{}", self.tag);
        let head = format!("{tag} {prefix} {{{}}}\r\n", literal.len());
        self.write(head.as_bytes()).await?;
        let go = self.read_line().await?;
        if !go.starts_with('+') {
            return Err(ImapError::Protocol(format!(
                "server refused the literal: {go}"
            )));
        }
        let mut rest = literal.to_vec();
        rest.extend_from_slice(suffix.as_bytes());
        rest.extend_from_slice(b"\r\n");
        self.write(&rest).await?;
        let mut untagged = Vec::new();
        loop {
            let resp = self.read_response().await?;
            if let Some(r) = resp.text.strip_prefix(&format!("{tag} ")) {
                if r.starts_with("OK") {
                    return Ok(untagged);
                }
                return Err(ImapError::Protocol(r.to_string()));
            }
            untagged.push(resp);
        }
    }

    async fn write(&mut self, bytes: &[u8]) -> Result<(), ImapError> {
        tokio::time::timeout(IO_TIMEOUT, self.stream.get_mut().write_all(bytes))
            .await
            .map_err(|_| ImapError::Protocol("write timed out".into()))?
            .map_err(|e| ImapError::Protocol(format!("write: {e}")))
    }

    async fn command(&mut self, cmd: &str) -> Result<Vec<Untagged>, ImapError> {
        self.tag += 1;
        let tag = format!("k{}", self.tag);
        let line = format!("{tag} {cmd}\r\n");
        tokio::time::timeout(IO_TIMEOUT, self.stream.get_mut().write_all(line.as_bytes()))
            .await
            .map_err(|_| ImapError::Protocol("write timed out".into()))?
            .map_err(|e| ImapError::Protocol(format!("write: {e}")))?;
        let mut untagged = Vec::new();
        loop {
            let resp = self.read_response().await?;
            if let Some(rest) = resp.text.strip_prefix(&format!("{tag} ")) {
                if rest.starts_with("OK") {
                    return Ok(untagged);
                }
                return Err(ImapError::Protocol(rest.to_string()));
            }
            untagged.push(resp);
        }
    }

    /// One logical response: a line, plus any `{n}` literals it announces,
    /// continued until a line that does not end in a literal marker.
    async fn read_response(&mut self) -> Result<Untagged, ImapError> {
        let mut out = Untagged::default();
        loop {
            let line = self.read_line().await?;
            match trailing_literal_len(&line) {
                Some(n) => {
                    if n > MAX_LITERAL {
                        return Err(ImapError::Protocol(format!("literal of {n} bytes")));
                    }
                    let cut = line.rfind('{').unwrap_or(line.len());
                    out.text.push_str(&line[..cut]);
                    out.text.push_str("{literal}");
                    let mut buf = vec![0u8; n];
                    tokio::time::timeout(IO_TIMEOUT, self.stream.read_exact(&mut buf))
                        .await
                        .map_err(|_| ImapError::Protocol("read timed out".into()))?
                        .map_err(|e| ImapError::Protocol(format!("read: {e}")))?;
                    out.literals.push(buf);
                }
                None => {
                    out.text.push_str(&line);
                    return Ok(out);
                }
            }
        }
    }

    async fn read_line(&mut self) -> Result<String, ImapError> {
        let mut buf = Vec::new();
        let n = tokio::time::timeout(IO_TIMEOUT, self.stream.read_until(b'\n', &mut buf))
            .await
            .map_err(|_| ImapError::Protocol("read timed out".into()))?
            .map_err(|e| ImapError::Protocol(format!("read: {e}")))?;
        if n == 0 {
            return Err(ImapError::Protocol("server closed the connection".into()));
        }
        while matches!(buf.last(), Some(b'\n' | b'\r')) {
            buf.pop();
        }
        Ok(String::from_utf8_lossy(&buf).into_owned())
    }
}

/// IMAP quoted string. Credentials with CR/LF cannot be quoted and are
/// rejected by the server anyway; strip them rather than break framing.
pub fn quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' | '\\' => {
                out.push('\\');
                out.push(c);
            }
            '\r' | '\n' => {}
            _ => out.push(c),
        }
    }
    out.push('"');
    out
}

/// The mailbox name at the end of a `* LIST (flags) "/" name` line, quoted
/// or not. Returned unquoted (and still in modified UTF-7, which is what
/// EXAMINE wants back).
pub fn list_mailbox_name(line: &str) -> Option<String> {
    let t = line.trim_end();
    if let Some(stripped) = t.strip_suffix('"') {
        let start = stripped.rfind(" \"")? + 2;
        Some(
            stripped[start..]
                .replace("\\\"", "\"")
                .replace("\\\\", "\\"),
        )
    } else {
        t.rsplit(' ').next().map(str::to_string)
    }
}

fn trailing_literal_len(line: &str) -> Option<usize> {
    let rest = line.strip_suffix('}')?;
    let open = rest.rfind('{')?;
    rest[open + 1..].trim_end_matches('+').parse().ok()
}

fn bracket_number(text: &str, key: &str) -> Option<u32> {
    let i = text.find(&format!("[{key} "))?;
    let rest = &text[i + key.len() + 2..];
    rest.split(|c: char| !c.is_ascii_digit())
        .next()?
        .parse()
        .ok()
}

fn word_number_after(text: &str, key: &str) -> Option<u32> {
    let mut words = text.split(|c: char| c.is_whitespace() || c == '(' || c == ')');
    while let Some(w) = words.next() {
        if w.eq_ignore_ascii_case(key) {
            return words.next()?.parse().ok();
        }
    }
    None
}

/// `01-Nov-2026` — the date format IMAP SEARCH wants.
pub fn imap_date(d: chrono::NaiveDate) -> String {
    d.format("%d-%b-%Y").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scripted server: for each client line it expects (prefix after the
    /// tag), it sends back the given bytes, with `TAG` replaced.
    async fn fake_server(
        io: tokio::io::DuplexStream,
        script: Vec<(&'static str, Vec<u8>)>,
    ) -> Vec<String> {
        let mut reader = BufReader::new(io);
        reader
            .get_mut()
            .write_all(b"* OK IMAP4rev1 ready\r\n")
            .await
            .unwrap();
        let mut seen = Vec::new();
        for (expect, reply) in script {
            let mut line = String::new();
            reader.read_line(&mut line).await.unwrap();
            let (tag, rest) = line.trim_end().split_once(' ').unwrap();
            assert!(
                rest.starts_with(expect),
                "expected `{expect}`, got `{rest}`"
            );
            seen.push(rest.to_string());
            let reply = String::from_utf8(reply).unwrap().replace("TAG", tag);
            reader.get_mut().write_all(reply.as_bytes()).await.unwrap();
        }
        seen
    }

    #[tokio::test]
    async fn read_only_session_end_to_end() {
        let (client, server) = tokio::io::duplex(64 * 1024);
        // A body line that looks like a literal marker must not confuse framing.
        let msg = "Subject: hi\r\n\r\nbody {12}\r\n";
        let n = msg.len();
        let script = vec![
            ("LOGIN", b"TAG OK LOGIN completed\r\n".to_vec()),
            (
                "EXAMINE \"INBOX\"",
                b"* 3 EXISTS\r\n* OK [UIDVALIDITY 77] ok\r\n* OK [UIDNEXT 105] ok\r\nTAG OK [READ-ONLY] done\r\n".to_vec(),
            ),
            ("UID SEARCH SINCE", b"* SEARCH 104 102\r\nTAG OK\r\n".to_vec()),
            (
                "UID FETCH 104,102 (UID FLAGS BODY.PEEK[]<0.100>)",
                format!(
                    "* 2 FETCH (UID 102 FLAGS (\\Seen) BODY[]<0> {{{n}}}\r\n{msg})\r\n* 3 FETCH (UID 104 FLAGS () BODY[]<0> {{{n}}}\r\n{msg})\r\nTAG OK\r\n"
                )
                .into_bytes(),
            ),
            ("LOGOUT", b"* BYE\r\nTAG OK\r\n".to_vec()),
        ];
        let server = tokio::spawn(fake_server(server, script));
        let mut s = ImapSession::over(client).await.unwrap();
        s.login("u@x", "p\"w").await.unwrap();
        let info = s.examine("INBOX").await.unwrap();
        assert_eq!(
            (info.exists, info.uid_validity, info.uid_next),
            (3, Some(77), Some(105))
        );
        let uids = s.uid_search("SINCE 01-Nov-2026").await.unwrap();
        assert_eq!(uids, vec![102, 104]);
        let got = s.uid_fetch(&[104, 102], 100).await.unwrap();
        assert_eq!(got.len(), 2);
        assert_eq!((got[0].uid, got[0].seen), (102, true));
        assert_eq!((got[1].uid, got[1].seen), (104, false));
        assert_eq!(got[1].raw, msg.as_bytes());
        s.logout().await;
        let seen = server.await.unwrap();
        // Only read-only verbs went over the wire.
        assert!(seen
            .iter()
            .all(|c| !c.starts_with("SELECT") && !c.contains("STORE")));
        assert!(seen[0].contains(r#""p\"w""#));
    }

    #[tokio::test]
    async fn japanese_search_goes_as_a_literal_and_all_mail_is_found() {
        let (client, server) = tokio::io::duplex(8192);
        let q = "PC 修理";
        let n = q.len();
        let script = vec![
            (
                "LIST \"\" \"*\"",
                b"* LIST (\\HasNoChildren) \"/\" \"INBOX\"\r\n* LIST (\\All \\HasNoChildren) \"/\" \"[Gmail]/&MFkweTBmMG4w4TD8MOs-\"\r\nTAG OK\r\n".to_vec(),
            ),
            (
                // The literal head: the server must answer `+` before the bytes.
                "UID SEARCH CHARSET UTF-8 X-GM-RAW {",
                b"+ go ahead\r\n".to_vec(),
            ),
        ];
        let server = tokio::spawn(async move {
            let seen = fake_server(server, script).await;
            seen
        });
        let mut s = ImapSession::over(client).await.unwrap();
        let all = s.all_mail_mailbox().await.unwrap();
        assert_eq!(all.as_deref(), Some("[Gmail]/&MFkweTBmMG4w4TD8MOs-"));
        // Second half of the exchange is driven by hand below.
        let search = tokio::spawn(async move {
            let r = s.uid_search_query(q, true).await;
            (s, r)
        });
        let seen = server.await.unwrap();
        assert!(seen[1].ends_with(&format!("{{{n}}}")), "{}", seen[1]);
        drop(search);
    }

    #[test]
    fn list_names() {
        assert_eq!(
            list_mailbox_name(r#"* LIST (\All) "/" "[Gmail]/All Mail""#).as_deref(),
            Some("[Gmail]/All Mail")
        );
        assert_eq!(
            list_mailbox_name(r#"* LIST () "/" INBOX"#).as_deref(),
            Some("INBOX")
        );
    }

    #[tokio::test]
    async fn rejected_login_is_an_auth_error_without_the_password() {
        let (client, server) = tokio::io::duplex(4096);
        let server = tokio::spawn(fake_server(
            server,
            vec![(
                "LOGIN",
                b"TAG NO [AUTHENTICATIONFAILED] Invalid credentials\r\n".to_vec(),
            )],
        ));
        let mut s = ImapSession::over(client).await.unwrap();
        let err = s.login("u@x", "hunter2-secret").await.unwrap_err();
        server.await.unwrap();
        assert!(matches!(err, ImapError::Auth(_)));
        assert!(!err.to_string().contains("hunter2"));
    }

    #[test]
    fn quoting_escapes_and_drops_line_breaks() {
        assert_eq!(quote(r#"a"b\c"#), r#""a\"b\\c""#);
        assert_eq!(quote("x\r\ny"), "\"xy\"");
    }

    #[test]
    fn literal_markers() {
        assert_eq!(
            trailing_literal_len("* 1 FETCH (UID 5 BODY[]<0> {123}"),
            Some(123)
        );
        assert_eq!(trailing_literal_len("a {12+}"), Some(12));
        assert_eq!(trailing_literal_len("* OK done"), None);
    }

    #[test]
    fn response_fields() {
        assert_eq!(
            bracket_number("* OK [UIDVALIDITY 3857529045] UIDs valid", "UIDVALIDITY"),
            Some(3857529045)
        );
        assert_eq!(
            bracket_number("* OK [UIDNEXT 4392] Predicted", "UIDNEXT"),
            Some(4392)
        );
        assert_eq!(
            word_number_after(
                "* 12 FETCH (UID 4391 FLAGS (\\Seen) BODY[]<0> {literal}",
                "UID"
            ),
            Some(4391)
        );
        assert_eq!(
            imap_date(chrono::NaiveDate::from_ymd_opt(2026, 11, 1).unwrap()),
            "01-Nov-2026"
        );
    }
}
