use crate::*;

/// Validated probability mass. It deliberately has no Deserialize implementation.
#[derive(Debug,Clone,PartialEq,Eq)]
pub struct Distribution(Vec<u32>);
impl Distribution {
    pub fn new(mass:Vec<u32>)->Result<Self> {
        if !(2..=7).contains(&mass.len()) || mass.iter().any(|&x|x>PPM)
            || mass.iter().map(|&x|x as u64).sum::<u64>()!=PPM as u64 { return Err(Error::Numeric); }
        Ok(Self(mass))
    }
    pub fn as_slice(&self)->&[u32]{&self.0}
    pub fn into_vec(self)->Vec<u32>{self.0}
    pub fn argmax(&self)->usize {
        // Stable first-index tie break, not Iterator::max_by's last-equal semantics.
        let mut best=0; for i in 1..self.0.len(){ if self.0[i]>self.0[best]{best=i;} } best
    }
    pub fn diagnostics(&self)->Diagnostics {
        let mut x=self.0.clone();x.sort_unstable_by(|a,b|b.cmp(a));
        Diagnostics{top1_ppm:x[0],margin_ppm:x[0]-x[1],tied:x[0]==x[1]}
    }
    pub fn expected_level_microunits(&self)->u64 {
        self.0.iter().enumerate().map(|(i,&x)|i as u64*x as u64).sum()
    }
    pub fn mean_ppm(&self)->u32 {
        let n=self.expected_level_microunits(); let d=(self.0.len()-1) as u64;
        ((n+d/2)/d) as u32
    }
    pub fn tail(&self,from:usize)->Result<u32> {
        if from>=self.0.len(){return Err(Error::Invalid("tail bin".into()));}
        Ok(self.0[from..].iter().sum())
    }
    pub fn cdf(&self,through:usize)->Result<u32> {
        if through>=self.0.len(){return Err(Error::Invalid("cdf bin".into()));}
        Ok(self.0[..=through].iter().sum())
    }
}

/// Converts an already normalized distribution. This is NOT softmax.
pub fn apportion(probabilities:&[f64])->Result<Distribution> {
    if !(2..=7).contains(&probabilities.len()) || probabilities.iter().any(|p|!p.is_finite() || *p<0.0 || *p>1.0) {return Err(Error::Numeric);}
    let sum:f64=probabilities.iter().sum();
    if (sum-1.0).abs()>1e-8 {return Err(Error::Numeric);}
    let scaled:Vec<f64>=probabilities.iter().map(|p|p*PPM as f64).collect();
    let mut mass:Vec<u32>=scaled.iter().map(|p|p.floor() as u32).collect();
    let used:u64=mass.iter().map(|&x|x as u64).sum();
    if used>PPM as u64 {return Err(Error::Numeric);}
    let left=PPM as usize-used as usize;
    if left>mass.len(){return Err(Error::Numeric);}
    let mut order:Vec<usize>=(0..mass.len()).collect();
    order.sort_by(|&a,&b| (scaled[b]-mass[b] as f64).total_cmp(&(scaled[a]-mass[a] as f64)).then(a.cmp(&b)));
    for i in order.into_iter().take(left){mass[i]+=1;}
    Distribution::new(mass)
}

pub fn from_logits(logits:&[f32],temperature:f64)->Result<Distribution> {
    if !(2..=7).contains(&logits.len()) || logits.iter().any(|x|!x.is_finite())
        || !temperature.is_finite() || !(1e-6..=1e6).contains(&temperature) {return Err(Error::Numeric);}
    let max=logits.iter().copied().fold(f32::NEG_INFINITY,f32::max) as f64;
    let p:Vec<f64>=logits.iter().map(|&x|((x as f64-max)/temperature).exp()).collect();
    let total:f64=p.iter().sum();
    if !total.is_finite() || total<=0.0 {return Err(Error::Numeric);}
    apportion(&p.iter().map(|x|x/total).collect::<Vec<_>>())
}

pub fn value(schema:&Schema,d:&Distribution)->Result<DecisionValue> {
    if !schema.primitive.count_valid(d.as_slice().len()) || schema.options.len()!=d.as_slice().len(){return Err(Error::BindingMismatch);}
    let ids:Vec<String>=schema.options.iter().map(|o|o.id.clone()).collect();
    Ok(match schema.primitive {
        Primitive::Choice=>DecisionValue::Choice{selected_id:ids[d.argmax()].clone(),option_ids:ids,mass_ppm:d.as_slice().to_vec()},
        Primitive::Noul=>DecisionValue::Noul{false_ppm:d.as_slice()[0],true_ppm:d.as_slice()[1]},
        Primitive::Score=>DecisionValue::Score{bin_ids:ids,mass_ppm:d.as_slice().to_vec(),expected_level_microunits:d.expected_level_microunits(),mean_ppm:d.mean_ppm()},
    })
}
/// Recomputes every redundant field. Call this after decoding untrusted DTOs.
pub fn validate_value(schema:&Schema,v:&DecisionValue)->Result<Distribution> {
    let d=Distribution::new(v.masses())?;
    if &value(schema,&d)?!=v {return Err(Error::BindingMismatch);}
    Ok(d)
}
