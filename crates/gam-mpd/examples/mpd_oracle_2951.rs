//! The investigator's measured-intervention server (#2951): loads named native models
//! (`NAME=PATH`, a Hugging Face checkpoint directory or a language-model export) into one
//! `oracle::Session` and answers one JSON request per line with one JSON line, `{"ok": ...}` or
//! `{"error": "..."}`, on a Unix socket (ADDRESS a path) or TCP (ADDRESS HOST:PORT). Requests run
//! one at a time; each is parallel inside. A run's node values stay within WORK_GIB: requests run
//! their sequences in groups that fit.
//!
//! usage: mpd_oracle_2951 ADDRESS WORK_GIB NAME=PATH [NAME=PATH ...]

use gam_mpd::oracle::{Native, Request, Session};
use serde_json::json;
use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::os::unix::net::UnixListener;
use std::path::Path;

/// Answer every request line of one connection.
fn serve<S: Read + Write>(session: &mut Session, stream: S) -> Result<(), String> {
    let mut reader = BufReader::new(stream);
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).map_err(|e| format!("read: {e}"))? == 0 {
            return Ok(());
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
        let stream = reader.get_mut();
        stream.write_all(text.as_bytes()).map_err(|e| format!("write: {e}"))?;
        stream.flush().map_err(|e| format!("flush: {e}"))?;
    }
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: mpd_oracle_2951 ADDRESS WORK_GIB NAME=PATH [NAME=PATH ...]";
    let [address, work, models @ ..] = &args[..] else {
        return Err(usage.into());
    };
    if models.is_empty() {
        return Err(usage.into());
    }
    let work: f64 = work.parse().map_err(|e| format!("WORK_GIB {work}: {e}"))?;
    let mut loaded = BTreeMap::new();
    for spec in models {
        let (name, path) = spec.split_once('=').ok_or_else(|| format!("{spec}: not NAME=PATH"))?;
        let start = std::time::Instant::now();
        loaded.insert(name.to_string(), Native::load(Path::new(path))?);
        eprintln!("loaded {name} from {path} in {:.1} s", start.elapsed().as_secs_f64());
    }
    let mut session = Session::new(loaded, (work * f64::from(1u32 << 30)) as usize);
    let tcp = address.rsplit_once(':').is_some_and(|(_, port)| port.parse::<u16>().is_ok()) && !address.contains('/');
    if tcp {
        let listener = TcpListener::bind(address).map_err(|e| format!("{address}: {e}"))?;
        eprintln!("listening on {address}");
        for stream in listener.incoming() {
            match stream {
                Ok(s) => serve(&mut session, s).unwrap_or_else(|e| eprintln!("{e}")),
                Err(e) => eprintln!("connection: {e}"),
            }
        }
    } else {
        // A socket file left by an earlier server is replaced.
        if Path::new(address).exists() {
            std::fs::remove_file(address).map_err(|e| format!("{address}: {e}"))?;
        }
        let listener = UnixListener::bind(address).map_err(|e| format!("{address}: {e}"))?;
        eprintln!("listening on {address}");
        for stream in listener.incoming() {
            match stream {
                Ok(s) => serve(&mut session, s).unwrap_or_else(|e| eprintln!("{e}")),
                Err(e) => eprintln!("connection: {e}"),
            }
        }
    }
    Ok(())
}
