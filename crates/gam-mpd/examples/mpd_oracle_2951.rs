//! The investigator's measured-intervention server (#2951): loads named native models
//! (`NAME=PATH`, a Hugging Face checkpoint directory or a language-model export) into one
//! `oracle::Session` and answers one JSON request per line on a Unix socket with one JSON line,
//! `{"ok": ...}` or `{"error": "..."}`. Requests run one at a time; each is parallel inside.
//!
//! usage: mpd_oracle_2951 SOCKET NAME=PATH [NAME=PATH ...]

use gam_mpd::oracle::{Native, Request, Session};
use serde_json::json;
use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixListener;
use std::path::Path;

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [socket, models @ ..] = &args[..] else {
        return Err("usage: mpd_oracle_2951 SOCKET NAME=PATH [NAME=PATH ...]".into());
    };
    if models.is_empty() {
        return Err("at least one NAME=PATH".into());
    }
    let mut loaded = BTreeMap::new();
    for spec in models {
        let (name, path) = spec.split_once('=').ok_or_else(|| format!("{spec}: not NAME=PATH"))?;
        let start = std::time::Instant::now();
        loaded.insert(name.to_string(), Native::load(Path::new(path))?);
        eprintln!("loaded {name} from {path} in {:.1} s", start.elapsed().as_secs_f64());
    }
    let mut session = Session::new(loaded);
    // A socket file left by an earlier server is replaced.
    if Path::new(socket).exists() {
        std::fs::remove_file(socket).map_err(|e| format!("{socket}: {e}"))?;
    }
    let listener = UnixListener::bind(socket).map_err(|e| format!("{socket}: {e}"))?;
    eprintln!("listening on {socket}");
    for stream in listener.incoming() {
        let mut stream = match stream {
            Ok(s) => s,
            Err(e) => {
                eprintln!("connection: {e}");
                continue;
            }
        };
        let mut line = String::new();
        if let Err(e) = BufReader::new(&stream).read_line(&mut line) {
            eprintln!("read: {e}");
            continue;
        }
        let start = std::time::Instant::now();
        let reply = match serde_json::from_str::<Request>(&line) {
            Ok(request) => match session.handle(&request) {
                Ok(value) => json!({"ok": value, "seconds": start.elapsed().as_secs_f64()}),
                Err(e) => json!({"error": e}),
            },
            Err(e) => json!({"error": format!("request: {e}")}),
        };
        let mut text = reply.to_string();
        text.push('\n');
        if let Err(e) = stream.write_all(text.as_bytes()) {
            eprintln!("write: {e}");
        }
    }
    Ok(())
}
