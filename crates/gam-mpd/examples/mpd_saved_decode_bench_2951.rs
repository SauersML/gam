//! CPU-only saved native-artifact decoder diagnostic. No fitting or fidelity claim.
//! SAVED OUT vocab=N max_bytes=N codec_bytes=N warmup=N repeats=N
use gam_mpd::{artifact::Artifact, codec::BitString, coder_capture::sha256,
    operator_program::{Declarations, Domain, NativeOperatorCodec, Slot}};
use serde_json::json;
use std::{collections::BTreeMap, io::Write, path::Path, time::Instant};
fn peak_rss_bytes() -> Option<u64> {
    std::fs::read_to_string("/proc/self/status").ok()?.lines().find_map(|line| line.strip_prefix("VmHWM:").and_then(|s|s.split_whitespace().next()).and_then(|s|s.parse::<u64>().ok()).and_then(|n|n.checked_mul(1024)))
}
fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len()!=7 { return Err("SAVED OUT vocab=N max_bytes=N codec_bytes=N warmup=N repeats=N".into()); }
    let mut options=BTreeMap::new();
    for arg in &args[2..] {
        let (key,value)=arg.split_once('=').ok_or("expected KEY=VALUE")?;
        if !["vocab","max_bytes","codec_bytes","warmup","repeats"].contains(&key) || options.insert(key,value).is_some() {return Err("unknown/duplicate option".into());}
    }
    let number=|key|->Result<usize,String>{options.get(key).ok_or("missing option")?.parse().map_err(|e|format!("{e}"))};
    let (vocab,max_bytes,codec_bytes,warmup,repeats)=(number("vocab")?,number("max_bytes")?,number("codec_bytes")?,number("warmup")?,number("repeats")?);
    if vocab==0 || warmup>2 || !(1..=8).contains(&repeats) {return Err("positive vocabulary; warmup0..2/repeats1..8 required".into());}
    let saved=Path::new(&args[0]);let out=Path::new(&args[1]);
    if std::fs::metadata(saved).map_err(|e|e.to_string())?.len()>max_bytes as u64 {return Err("saved input exceeds declared byte budget".into());}
    if out.exists() {return Err("fresh output required".into());}
    std::fs::create_dir_all(out).map_err(|e|e.to_string())?;
    let bytes=std::fs::read(saved).map_err(|e|e.to_string())?;
    let length=u64::from_le_bytes(bytes.get(..8).ok_or("missing length")?.try_into().map_err(|_|"invalid length")?);
    if length==u64::MAX || length.div_ceil(8)!=bytes.len().saturating_sub(8) as u64 {return Err("benchmark requires legacy saved native envelope".into());}
    let message=BitString::from_packed(&bytes[8..],length).map_err(|e|e.to_string())?;
    let declarations=Declarations{domains:vec![Domain{size:vocab}],slots:vec![Slot::Token{domain:0}],parameters:0};
    let started=Instant::now();let reference=Artifact::from_bytes(&bytes,&declarations)?;
    let initial_decode_seconds=started.elapsed().as_secs_f64();
    let started=Instant::now();let codec=NativeOperatorCodec::new(&reference.program,codec_bytes).map_err(|e|e.to_string())?;
    let codec_init_seconds=started.elapsed().as_secs_f64();
    let exact_wire_equal=reference.to_bytes()?==bytes;
    if !exact_wire_equal {return Err("ordinary saved-byte canonical replay mismatch".into());}
    let protocol=json!({"scope":"CPU saved native artifact decode only; no fitting/Local/Run","input":saved,"input_sha256":sha256(saved)?,"bytes":bytes.len(),"bits":length,"declarations":{"vocab":vocab,"token_slot":0,"parameters":0},"max_bytes":max_bytes,"codec_bytes":codec_bytes,"warmup":warmup,"repeats":repeats,"ordering":"alternating ordinary/cached then cached/ordinary per round","timed_scope":"Artifact::decode on existing packed message; excludes disk/outer packing/cache init/equality/reencoding","binary_sha256":sha256(&std::env::current_exe().map_err(|e|e.to_string())?)?,"source":include_str!("mpd_saved_decode_bench_2951.rs"),"exact_ordinary_saved_wire_replay":exact_wire_equal,"initial_from_bytes_seconds":initial_decode_seconds,"codec_init_seconds":codec_init_seconds});
    std::fs::write(out.join("PROTOCOL.json"),serde_json::to_vec_pretty(&protocol).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    let mut journal=std::fs::OpenOptions::new().create_new(true).write(true).open(out.join("TIMINGS.jsonl")).map_err(|e|e.to_string())?;
    for round in 0..warmup+repeats {
        for cached in if round%2==0 {[false,true]} else {[true,false]} {
            let started=Instant::now();
            let decoded=if cached {Artifact::decode_with_native_codec(&message,&declarations,&codec)?} else {Artifact::decode(&message,&declarations)?};
            let seconds=started.elapsed().as_secs_f64();
            if decoded!=reference {return Err("decoded graph/labels/numeric values differ".into());}
            let row=json!({"round":round,"warmup":round<warmup,"cached":cached,"decode_seconds":seconds,"exact_reference_equal":true,"codec_usage":codec.usage(),"peak_rss_bytes":peak_rss_bytes()});
            writeln!(journal,"{row}").map_err(|e|e.to_string())?;journal.flush().map_err(|e|e.to_string())?;
        }
    }
    Ok(())
}
