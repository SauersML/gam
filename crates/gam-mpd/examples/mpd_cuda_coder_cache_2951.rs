//! Explicit CUDA cache pilot. No fallback. Prints absolute bound gaps and operation counts.
//! Optional arguments: rows pieces nodes scratch_MiB concurrent_rows. Defaults: 40 18 16 128 2.
use gam_gpu::tensor::{CodeRowsWorkspace, Device};
use gam_gpu::GpuPolicy;
use gam_mpd::sparse_code::Coder;
use ndarray::Array2;

fn noise(seed: usize) -> f64 {
    let mut x = (seed as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
    x ^= x >> 33; x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD); x ^= x >> 33;
    (x >> 11) as f64 / (1u64 << 52) as f64 - 1.0
}

fn main() -> Result<(), String> {
    let args: Vec<usize> = std::env::args().skip(1).map(|a| a.parse().map_err(|e| format!("argument {a}: {e}"))).collect::<Result<_, _>>()?;
    if args.len() > 5 { return Err("expected rows pieces nodes scratch_MiB concurrent_rows".into()); }
    let get = |i, fallback| args.get(i).copied().unwrap_or(fallback);
    let (rows, pieces, nodes, bytes, max_rows) = (get(0,40),get(1,18),get(2,16),get(3,128).checked_mul(1 << 20).ok_or("budget overflow")?,get(4,2));
    if rows == 0 || pieces == 0 || max_rows == 0 { return Err("rows, pieces, and concurrent_rows must be positive".into()); }
    let d = Device::accelerator(GpuPolicy::Required).map_err(|e| e.to_string())?.ok_or("required CUDA device absent")?;
    if !d.float64() { return Err("required float64 CUDA device absent".into()); }
    let m = |r: usize, c: usize, salt: usize| Array2::from_shape_fn((r,c), |(i,j)| noise(salt * 7919 + 131 * i + j));
    let (v,u,x,a) = (m(pieces,7,1),m(pieces,6,2),m(rows,7,3),m(6,6,4));
    let f = a.dot(&a.t()); let gram = u.dot(&f).dot(&u.t());
    let z = x.dot(&v.t()); let y = x.dot(&m(6,7,5).t());
    let weights = y.dot(&f).dot(&u.t());
    let yfy: Vec<f64> = y.outer_iter().zip(y.dot(&f).outer_iter()).map(|(p,q)| p.dot(&q)).collect();
    let up = |a: &Array2<f64>| d.upload(a.view()).map_err(|e| e.to_string());
    let (dz,dw,dy,dk) = (up(&z)?,up(&weights)?,up(&Array2::from_shape_vec((rows,1),yfy.clone()).unwrap())?,up(&gram)?);
    for grouped in [false,true] {
        let ranks = if grouped { let mut r = vec![2; pieces/2]; if pieces%2 != 0 { r.push(1); } r } else { vec![1;pieces] };
        let bits: Vec<f64> = ranks.iter().enumerate().map(|(b,r)| 2.0 + 3.0 * *r as f64 + noise(b+1).abs()).collect();
        let coder = Coder::new(gram.clone(),&ranks,&bits,3.0,nodes)?;
        let cpu = coder.code_rows(z.view(),weights.view(),&yfy,None)?;
        let starts: Vec<u32> = coder.starts().iter().map(|s| u32::try_from(*s).map_err(|e| e.to_string())).collect::<Result<_,_>>()?;
        let ds = d.upload_indices(&starts).map_err(|e|e.to_string())?;
        let db = d.upload_vec(1,bits.len(),bits).map_err(|e|e.to_string())?;
        // Compile/warm the kernel before timing either profile variant, and retain the
        // unchanged default execution as a bitwise reference.
        let mut original = d.zeros(rows,ranks.len()).map_err(|e|e.to_string())?;
        let (original_upper,original_lower) = d.code_rows((&dz,&dw,&dy),(&dk,&ds,&db),None,(coder.kappa(),nodes,1e-12),&mut original).map_err(|e|e.to_string())?;
        let original = d.download(&original).map_err(|e|e.to_string())?;
        let mut previous = Some((original,original_upper.iter().map(|x|x.to_bits()).collect::<Vec<_>>(),original_lower.iter().map(|x|x.to_bits()).collect::<Vec<_>>()));
        for (cached,fused) in [(false,false),(false,true),(true,false),(true,true)] {
            let mut on = d.zeros(rows,ranks.len()).map_err(|e|e.to_string())?;
            let workspace = CodeRowsWorkspace {bytes,max_rows,cache_columns:cached};
            let result = if fused {
                d.code_rows_profiled_fused((&dz,&dw,&dy),(&dk,&ds,&db),None,(coder.kappa(),nodes,1e-12),&mut on,workspace)
            } else {
                d.code_rows_profiled((&dz,&dw,&dy),(&dk,&ds,&db),None,(coder.kappa(),nodes,1e-12),&mut on,workspace)
            };
            let (upper,lower,profile) = result.map_err(|e|e.to_string())?;
            let on = d.download(&on).map_err(|e|e.to_string())?;
            for (r,(sets,code,bound)) in cpu.iter().enumerate() {
                assert_eq!(on.row(r).iter().map(|x|*x == 1.0).collect::<Vec<_>>(),*sets,"CPU masks row{r} grouped={grouped}");
                assert!((upper[r]-code).abs() <= 1e-9*code.abs().max(1.0),"CPU upper row{r}");
                assert!((lower[r]-bound).abs() <= 1e-9*bound.abs().max(1.0),"CPU lower row{r}");
                assert!(upper[r].is_finite() && lower[r].is_finite());
                assert!(lower[r] <= upper[r] + 1e-9*upper[r].abs().max(1.0));
            }
            if let Some((prev_on,prev_upper,prev_lower)) = &previous {
                assert_eq!(&on,prev_on,"cache masks");
                assert_eq!(upper.iter().map(|x|x.to_bits()).collect::<Vec<_>>(),*prev_upper,"cache upper bitwise");
                assert_eq!(lower.iter().map(|x|x.to_bits()).collect::<Vec<_>>(),*prev_lower,"cache lower bitwise");
            }
            for c in &profile.rows {
                assert!(c.explored_nodes <= nodes as u64);
                assert!(c.relaxations <= 1 + 2 * nodes as u64);
                if cached { assert!(c.column_computations <= ranks.len() as u64); }
            }
            let counts = profile.rows.iter().fold([0u64;7],|mut s,c| { for(i,x) in [c.relaxations,c.sweeps,c.column_computations,c.column_cache_hits,c.rounding_flips,c.explored_nodes,c.sweep_limit_hits].iter().enumerate() {s[i]+=x;} s });
            let gaps: Vec<f64> = upper.iter().zip(&lower).map(|(u,l)|u-l).collect();
            println!("{}",serde_json::json!({"device":d.name(),"grouped":grouped,"cached":cached,"fused":fused,"rows":rows,"pieces":pieces,"nodes":nodes,"seconds":profile.elapsed.as_secs_f64(),"workspace_bytes":profile.workspace_bytes,"concurrent_rows":profile.concurrent_rows,"mean_absolute_gap":gaps.iter().sum::<f64>()/rows as f64,"max_absolute_gap":gaps.iter().copied().fold(0.0_f64,f64::max),"relaxations":counts[0],"sweeps":counts[1],"column_computations":counts[2],"column_cache_hits":counts[3],"rounding_flips":counts[4],"explored_nodes":counts[5],"sweep_limit_hits":counts[6]}));
            previous = Some((on,upper.iter().map(|x|x.to_bits()).collect::<Vec<_>>(),lower.iter().map(|x|x.to_bits()).collect::<Vec<_>>()));
        }
    }
    Ok(())
}
