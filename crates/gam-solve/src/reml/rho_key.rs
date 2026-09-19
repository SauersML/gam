use ndarray::Array1;

/// Sanitized memoization key for rho vectors.
///
/// - Rejects NaN entries (returns `None` so callers skip caching).
/// - Canonicalizes ±0.0 to +0.0 to avoid key drift.
pub(super) fn sanitized_rhokey(rho: &Array1<f64>) -> Option<Vec<u64>> {
    let mut key = Vec::with_capacity(rho.len());
    for &v in rho {
        if v.is_nan() {
            return None;
        }
        key.push(if v == 0.0 {
            0.0f64.to_bits()
        } else {
            v.to_bits()
        });
    }
    Some(key)
}

/// Cache identity for an evaluation whose inner solve is controlled by the
/// outer iteration cap.
///
/// The cap suffix is part of the mathematical input: a partial mode computed
/// under a search cap must not alias the uncapped stationary mode at identical
/// rho.  Keeping the construction here makes bundle, PIRLS, and outer-eval
/// caches share one identity convention.
pub(super) fn sanitized_eval_state_key(rho: &Array1<f64>, outer_cap: usize) -> Option<Vec<u64>> {
    let mut key = sanitized_rhokey(rho)?;
    key.push(outer_cap as u64);
    Some(key)
}

/// The uncapped identity that may answer a request made under an outer
/// iteration cap.
///
/// The outer cap is only a budget on how far the inner solve descends toward
/// the stationary mode at rho; a mode certified with no cap at the same rho is
/// that destination, certified at the uncapped (tighter) tolerance, and it is
/// the mode the criterion's value at rho was scored on.  It is not bitwise what
/// a capped solve would return (that one stops at the looser tolerance, from
/// its own warm start, or on its budget), but it is the quantity the capped
/// solve approximates, so answering from it is at least as accurate and keeps
/// value and gradient on one beta.  The converse
/// never holds (#2309): a capped entry is a partial solve and must not stand in
/// for the uncapped mode.  Screening keys have no stand-in, since a screening
/// solve carries its own cache and KKT semantics.
pub(super) fn uncapped_stand_in_key(key: &[u64]) -> Option<Vec<u64>> {
    let [rho @ .., screening_cap, outer_cap] = key else {
        return None;
    };
    if *screening_cap != 0 || *outer_cap == 0 {
        return None;
    }
    let mut uncapped = rho.to_vec();
    uncapped.extend([0, 0]);
    Some(uncapped)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn evaluation_cache_identity_includes_inner_fidelity_2309() {
        let rho = array![0.25, -1.5];
        let capped = sanitized_eval_state_key(&rho, 3).expect("finite capped key");
        let finalized = sanitized_eval_state_key(&rho, 0).expect("finite terminal key");

        assert_ne!(capped, finalized);
        assert_eq!(&capped[..rho.len()], &finalized[..rho.len()]);
    }

    #[test]
    fn only_an_outer_capped_key_has_an_uncapped_stand_in() {
        let rho = array![0.25, -1.5];
        let capped = sanitized_eval_state_key(&rho, 0, 3).expect("finite capped key");
        let finalized = sanitized_eval_state_key(&rho, 0, 0).expect("finite terminal key");
        let screened = sanitized_eval_state_key(&rho, 3, 0).expect("finite screening key");
        let screened_capped = sanitized_eval_state_key(&rho, 3, 5).expect("finite key");

        assert_eq!(uncapped_stand_in_key(&capped), Some(finalized.clone()));
        assert_eq!(uncapped_stand_in_key(&finalized), None);
        assert_eq!(uncapped_stand_in_key(&screened), None);
        assert_eq!(uncapped_stand_in_key(&screened_capped), None);
    }
}
