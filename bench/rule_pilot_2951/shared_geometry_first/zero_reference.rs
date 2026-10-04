use std::{env,fs};
fn main()->Result<(),Box<dyn std::error::Error>> {
 let args:Vec<_>=env::args().skip(1).collect();
 let width:usize=args[0].parse()?;
 for path in &args[1..] {
  let bytes=fs::read(path)?;
  if bytes.len()%(8*width)!=0 {return Err("bad shape".into());}
  let mut squared=Vec::new();
  for row in bytes.chunks_exact(8*width) {
   let mut sum=0.0f64;
   for chunk in row.chunks_exact(8) {let value=f64::from_le_bytes(chunk.try_into()?); if !value.is_finite(){return Err("nonfinite".into());} sum+=value*value;}
   squared.push(sum);
  }
  let rms=(squared.iter().sum::<f64>()/squared.len() as f64).sqrt();
  let maximum=squared.iter().copied().fold(0.0f64,f64::max).sqrt()/rms;
  println!("{{\"file\":\"{}\",\"rows\":{},\"width\":{},\"native_rms\":{},\"zero_prediction_max_normalized_error\":{}}}",path,squared.len(),width,rms,maximum);
 }
 Ok(())
}
