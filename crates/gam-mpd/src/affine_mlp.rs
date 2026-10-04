//! Full-output-width affine MLP proposal baseline. Augmented-design SVD is a
//! numerical fitting convention, never a structural discovery/rank certificate.
use crate::{artifact::{Argument,Artifact,Callee},operator_program::{Interface,Node,Operator,Provenance,Rule,exact_precision},run_check::LayerNodes};
use gam_linalg::{decompose::svd,faer_ndarray::{fast_ab,fast_abt}};
use ndarray::{Array1,Array2,Axis,s};
#[derive(Clone,Debug)]
pub struct AffineFit {
    pub weights:Array2<f64>,
    pub offset:Array1<f64>,
    pub training_rows:usize,
    pub augmented_singular_values:Array1<f64>,
    pub numerical_resolution:f64,
    pub numerical_rank:usize,
}
impl AffineFit {
    /// Minimum joint coefficient norm for [X,ones]^+Y at the fixed
    /// max(rows,columns)*epsilon*sigma_max SVD resolution convention.
    /// All independent coefficients are projected once to f32 before returning.
    pub fn fit(inputs:&Array2<f64>,outputs:&Array2<f64>)->Result<Self,String>{
        if inputs.nrows()==0 || inputs.ncols()==0 || outputs.ncols()==0 || outputs.nrows()!=inputs.nrows() || inputs.iter().chain(outputs.iter()).any(|v|!v.is_finite()) {return Err("finite nonempty aligned affine training arrays required".into());}
        let columns=inputs.ncols().checked_add(1).ok_or("augmented width overflow")?;
        let mut design=Array2::ones((inputs.nrows(),columns));design.slice_mut(s![..,..inputs.ncols()]).assign(inputs);
        let factor=svd(design.view(),false).map_err(|e|e.to_string())?;
        if !factor.band.is_finite() || factor.band<0.0 || factor.singular_values.iter().any(|v|!v.is_finite() || *v<0.0) || factor.u.iter().chain(factor.vt.iter()).any(|v|!v.is_finite()){return Err("nonfinite affine SVD factors/resolution".into());}
        let mut projected=fast_ab(&factor.u.t().to_owned(),outputs);let mut numerical_rank=0;
        for (index,mut row) in projected.axis_iter_mut(Axis(0)).enumerate(){
            let sigma=factor.singular_values[index];if sigma>factor.band {numerical_rank+=1;row.mapv_inplace(|v|v/sigma);}else{row.fill(0.0);}
        }
        let coefficients=fast_ab(&factor.vt.t().to_owned(),&projected);
        let weights=coefficients.slice(s![..inputs.ncols(),..]).t().mapv(|v|f64::from(v as f32));
        let offset=coefficients.row(inputs.ncols()).mapv(|v|f64::from(v as f32));
        if weights.iter().chain(offset.iter()).any(|v|!v.is_finite()){return Err("affine coefficient overflow in f32 projection".into());}
        Ok(Self{weights,offset,training_rows:inputs.nrows(),augmented_singular_values:factor.singular_values,numerical_resolution:factor.band,numerical_rank})
    }
    pub fn predict(&self,inputs:&Array2<f64>)->Result<Array2<f64>,String>{
        if inputs.ncols()!=self.weights.ncols() || inputs.iter().any(|v|!v.is_finite()){return Err("affine prediction input mismatch/nonfinite".into());}
        let mut predicted=fast_abt(inputs,&self.weights);predicted+=&self.offset;
        if predicted.iter().any(|v|!v.is_finite()){return Err("affine prediction overflow".into());}Ok(predicted)
    }
    /// Actual Rule/Call replacement, preserving native input/output lineage and
    /// adding the independently serialized paid full-source-scale response binding.
    pub fn candidate(&self,base:&Artifact,native:&crate::operator_program::OperatorProgram,layer:&LayerNodes,name:&str)->Result<Artifact,String>{
        let interfaces=base.program.interfaces().map_err(|e|e.to_string())?;
        let input=&interfaces[base.place(layer.normed).ok_or("affine native input absent")?];
        let output=&interfaces[base.place(layer.mlp).ok_or("affine native MLP write absent")?];
        if self.weights.dim()!=(output.width(),input.width()) || self.offset.len()!=output.width(){return Err("affine fitted dimensions disagree with native boundary".into());}
        if self.weights.iter().chain(self.offset.iter()).any(|v|!v.is_finite() || *v!=f64::from(*v as f32)){return Err("affine candidate requires finite f32 coefficient literals".into());}
        let first=base.program.operators.len();let bias=self.offset.clone().insert_axis(Axis(1));
        let operators=vec![Operator::dense(format!("{name} full affine"),output.clone(),input.clone(),self.weights.clone(),exact_precision(self.weights.iter().copied()).map_err(|e|e.to_string())?,Provenance::default()).map_err(|e|e.to_string())?,Operator::dense(format!("{name} offset"),output.clone(),Interface::constant(),bias.clone(),exact_precision(bias.iter().copied()).map_err(|e|e.to_string())?,Provenance::default()).map_err(|e|e.to_string())?];
        let rule=Rule{name:name.into(),inputs:vec![input.clone()],nodes:vec![Node::Param{index:0},Node::Affine{terms:vec![(0,first)],bias:Some(first+1)}],output:1};
        base.replace_block(name,Callee::New(rule),vec![Argument::Native(layer.normed)],layer.mlp,operators)?.with_uniform_scale_control(native,layer.active,layer.mlp)
    }
}
#[derive(Clone,Debug,serde::Serialize)]
pub struct ResidualSpectrum {
    /// Raw response norms are diagnostics, not arithmetic certificates.
    pub native_frobenius:f64,pub residual_frobenius:f64,
    /// Validated centered residual spectrum and ideal rank-plus-offset floors,
    /// conditional on this fixed f32-coefficient matrix response.
    pub fixed_base_curve:crate::native_mlp_rank::FixedBaseRankCurve,
}
/// CPU fixed-response diagnostic, outside decoded neural Local/Run evidence.
/// Uses one validated centered-residual SVD for the complete declared rank list.
pub fn residual_spectrum(fit:&AffineFit,inputs:&Array2<f64>,outputs:&Array2<f64>,ranks:&[usize])->Result<ResidualSpectrum,String>{
    if outputs.nrows()!=inputs.nrows() || outputs.ncols()!=fit.weights.nrows() || outputs.iter().any(|v|!v.is_finite()){return Err("residual target mismatch/nonfinite".into());}
    let predicted=fit.predict(inputs)?;
    let native_frobenius=outputs.iter().fold(0.0_f64,|norm,v|norm.hypot(*v));
    let residual_frobenius=(outputs-&predicted).iter().fold(0.0_f64,|norm,v|norm.hypot(*v));
    if !native_frobenius.is_finite() || !residual_frobenius.is_finite(){return Err("residual response norm unresolved".into());}
    let fixed_base_curve=crate::native_mlp_rank::measured_fixed_base_rank_curve(outputs,&predicted,ranks)?;
    Ok(ResidualSpectrum{native_frobenius,residual_frobenius,fixed_base_curve})
}
#[cfg(test)]
#[path="affine_mlp_tests.rs"]
mod tests;
