//! The `CustomFamily` trait itself plus the evaluation result structs it returns
//! (`FamilyEvaluation`, joint-gradient/batched-term carriers) and the eval-scope /
//! outer-eval-context enums that parameterize trait calls.

pub mod families {
    pub mod custom_family {
        pub mod family_trait;
    }
}
