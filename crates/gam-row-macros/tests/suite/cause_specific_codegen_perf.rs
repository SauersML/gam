use gam_row_macros::row_atom;

row_atom! {
    fn generated_cause_specific [order2, third, fourth](
        eta_exit,
        eta_entry,
        derivative;
        weight: scale,
        entry_active: bool,
        event: bool
    ) {
        weight
            * (exp(eta_exit)
                - entry_active * exp(eta_entry)
                - event * (eta_exit + ln(derivative)))
    }
}

