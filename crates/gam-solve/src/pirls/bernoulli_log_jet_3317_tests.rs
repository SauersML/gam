//! #3317: the Bernoulli observed information from the two sides'
//! log-probability jets.
//!
//! The ratio tower divided by `V = μ(1−μ)`; on a saturated cloglog row
//! (`η = 6.6547`) `μ'` and `1 − μ` underflow together and it evaluated `0/0`,
//! refusing the whole fit. `−∂² [y log μ + (1−y) log(1−μ)]` needs no such
//! division, and each side's jet is built so that side's own small
//! probability never appears in a denominator.

use super::*;
use gam_problem::{InverseLink, StandardLink};

/// `(link, η, ∂^{1..5} log μ, ∂^{1..5} log(1−μ))` from mpmath at 400 decimal
/// digits, differentiating the closed-form `μ` and `1−μ` of each link with no
/// gam code involved.
const REFERENCE: [(StandardLink, f64, [f64; 5], [f64; 5]); 32] = [
    (StandardLink::Logit, -40.0, [1.0, -4.248354255291589e-18, -4.2483542552915889e-18, -4.2483542552915889e-18, -4.2483542552915887e-18], [-4.248354255291589e-18, -4.248354255291589e-18, -4.2483542552915889e-18, -4.2483542552915889e-18, -4.2483542552915887e-18]),
    (StandardLink::Logit, -5.0, [9.9330714907571514e-1, -6.6480566707901549e-3, -6.5590677663225629e-3, -6.38287672580193e-3, -6.0358071157057782e-3], [-6.6928509242848556e-3, -6.6480566707901549e-3, -6.5590677663225629e-3, -6.38287672580193e-3, -6.0358071157057782e-3]),
    (StandardLink::Logit, 0.0, [5.0e-1, -2.5e-1, 0.0, 1.25e-1, 0.0], [-5.0e-1, -2.5e-1, 0.0, 1.25e-1, 0.0]),
    (StandardLink::Logit, 0.75, [3.2082130082460703e-1, -2.1789499376181403e-1, 7.8084283078144409e-2, 6.6974376076951819e-2, -1.2608580941235155e-1], [-6.7917869917539297e-1, -2.1789499376181403e-1, 7.8084283078144409e-2, 6.6974376076951819e-2, -1.2608580941235155e-1]),
    (StandardLink::Logit, 4.0, [1.7986209962091558e-2, -1.7662706213291116e-2, 1.702733592838913e-2, -1.5790879068628921e-2, 1.3418349943211285e-2], [-9.8201379003790844e-1, -1.7662706213291116e-2, 1.702733592838913e-2, -1.5790879068628921e-2, 1.3418349943211285e-2]),
    (StandardLink::Logit, 40.0, [4.248354255291589e-18, -4.248354255291589e-18, 4.2483542552915889e-18, -4.2483542552915889e-18, 4.2483542552915887e-18], [-1.0, -4.248354255291589e-18, 4.2483542552915889e-18, -4.2483542552915889e-18, 4.2483542552915887e-18]),
    (StandardLink::Probit, -30.0, [3.0033259667433677e+1, -9.9889622848810991e-1, 7.3099930157844385e-5, 7.2459372109803421e-6, 9.555654977815648e-7], [-1.4736461348785475e-196, -4.4209384046356426e-195, -1.3248078752558142e-193, -3.9655817489581714e-192, -1.185700101061684e-190]),
    (StandardLink::Probit, -8.5, [8.6145953201651729, -9.8719230880772791e-1, 2.7945420853499121e-3, 8.9321786174527929e-4, 3.718074652091409e-4], [-8.1662356316695501e-17, -6.9413002869191177e-16, -5.8184428875645546e-15, -4.8068504486914893e-14, -3.9112695947608296e-13]),
    (StandardLink::Probit, -2.0, [2.3732155328228409, -8.8572089958591874e-1, 5.9355861291565813e-2, 3.9421993865946813e-2, 2.9098988655348834e-2], [-5.5247862678989959e-2, -1.1354805168857645e-1, -1.8439481503247759e-1, -1.8785468561160969e-1, 3.1091902195183747e-2]),
    (StandardLink::Probit, 0.5, [5.0916043383703349e-1, -5.138245643036329e-1, 2.7099012446870783e-1, 8.8167801929197554e-2, -1.113890911178517e-1], [-1.1410777703680645, -7.3151959284412105e-1, -1.6260392517612297e-1, 1.0301187006051434e-1, -4.2292792319855707e-2]),
    (StandardLink::Probit, 3.0, [4.4378390421256638e-3, -1.3333211541740806e-2, 3.5680136876570471e-2, -8.1046222015181884e-2, 1.3967198046926724e-1], [-3.2830986549304365, -9.2944081321473188e-1, -3.1470672830842488e-2, 1.8930547102077378e-2, -1.3578681847727706e-2]),
    (StandardLink::Probit, 9.0, [1.0279773571668915e-18, -9.2517962145020233e-18, 8.2238188573351318e-17, -7.2164010473115782e-16, 6.2480463768603664e-15], [-9.1085231050028688, -9.8848520934528287e-1, -2.3907474397987586e-3, 7.2877087271227609e-4, -2.8995377143551576e-4]),
    (StandardLink::Probit, 30.0, [1.4736461348785475e-196, -4.4209384046356426e-195, 1.3248078752558142e-193, -3.9655817489581714e-192, 1.185700101061684e-190], [-3.0033259667433677e+1, -9.9889622848810991e-1, -7.3099930157844385e-5, 7.2459372109803421e-6, -9.555654977815648e-7]),
    (StandardLink::CLogLog, -30.0, [9.9999999999995321e-1, -4.6788114844199414e-14, -4.6788114844197954e-14, -4.6788114844195035e-14, -4.6788114844189198e-14], [-9.3576229688401746e-14, -9.3576229688401746e-14, -9.3576229688401746e-14, -9.3576229688401746e-14, -9.3576229688401746e-14]),
    (StandardLink::CLogLog, -5.0, [9.9663480982507476e-1, -3.3614068560331543e-3, -3.3538402354252077e-3, -3.3387070629140639e-3, -3.3084409927100313e-3], [-6.7379469990854671e-3, -6.7379469990854671e-3, -6.7379469990854671e-3, -6.7379469990854671e-3, -6.7379469990854671e-3]),
    (StandardLink::CLogLog, -0.75, [7.8234212025904077e-1, -1.9926932328323009e-1, -1.6289984062778239e-1, -9.1777041256881869e-2, 4.4089355395718755e-2], [-4.7236655274101471e-1, -4.7236655274101471e-1, -4.7236655274101471e-1, -4.7236655274101471e-1, -4.7236655274101471e-1]),
    (StandardLink::CLogLog, 1.25, [1.0976889901732219e-1, -2.8541141580760343e-1, 3.9029979946257947e-1, 3.886510786687301e-1, -1.866242365825666], [-3.4903429574618414, -3.4903429574618414, -3.4903429574618414, -3.4903429574618414, -3.4903429574618414]),
    (StandardLink::CLogLog, 3.679931643574629, [2.4051422563565906e-16, -9.2943557520704403e-15, 3.496332774270322e-13, -1.2783727780206083e-11, 4.535239367531583e-10], [-3.9643684079418722e+1, -3.9643684079418722e+1, -3.9643684079418722e+1, -3.9643684079418722e+1, -3.9643684079418722e+1]),
    (StandardLink::CLogLog, 5.259300664374346, [5.6088690242878139e-82, -1.0732398640298857e-79, 2.0428230353402749e-77, -3.8677001533693986e-75, 7.2834644767827949e-73], [-1.9234692919062418e+2, -1.9234692919062418e+2, -1.9234692919062418e+2, -1.9234692919062418e+2, -1.9234692919062418e+2]),
    (StandardLink::CLogLog, 6.654659400973018, [5.089738001893451e-335, -3.9465494539041362e-332, 3.0561769148352035e-329, -2.3636153302233611e-326, 1.825622620999198e-323], [-7.7639343919784609e+2, -7.7639343919784609e+2, -7.7639343919784609e+2, -7.7639343919784609e+2, -7.7639343919784609e+2]),
    (StandardLink::LogLog, -6.654659400973018, [7.7639343919784609e+2, -7.7639343919784609e+2, 7.7639343919784609e+2, -7.7639343919784609e+2, 7.7639343919784609e+2], [-5.089738001893451e-335, -3.9465494539041362e-332, -3.0561769148352035e-329, -2.3636153302233611e-326, -1.825622620999198e-323]),
    (StandardLink::LogLog, -5.259300664374346, [1.9234692919062418e+2, -1.9234692919062418e+2, 1.9234692919062418e+2, -1.9234692919062418e+2, 1.9234692919062418e+2], [-5.6088690242878139e-82, -1.0732398640298857e-79, -2.0428230353402749e-77, -3.8677001533693986e-75, -7.2834644767827949e-73]),
    (StandardLink::LogLog, -3.0, [2.0085536923187668e+1, -2.0085536923187668e+1, 2.0085536923187668e+1, -2.0085536923187668e+1, 2.0085536923187668e+1], [-3.8005425112356633e-8, -7.2535394570773867e-7, -1.3080410198943996e-5, -2.2127176632050637e-4, -3.4778533648589075e-3]),
    (StandardLink::LogLog, -0.4, [1.4918246976412703, -1.4918246976412703, 1.4918246976412703, -1.4918246976412703, 1.4918246976412703], [-4.3301550407510199e-1, -4.0047014613513395e-1, 1.0220255051208411e-1, 3.6690292637226551e-1, -8.5993975354403667e-1]),
    (StandardLink::LogLog, 2.0, [1.3533528323661269e-1, -1.3533528323661269e-1, 1.3533528323661269e-1, -1.3533528323661269e-1, 1.3533528323661269e-1], [-9.338581959051936e-1, -6.4616897599770872e-2, 6.1569876071870879e-2, -5.5486990753101406e-2, 4.3365802424947808e-2]),
    (StandardLink::LogLog, 5.0, [6.7379469990854671e-3, -6.7379469990854671e-3, 6.7379469990854671e-3, -6.7379469990854671e-3, 6.7379469990854671e-3], [-9.9663480982507476e-1, -3.3614068560331543e-3, 3.3538402354252077e-3, -3.3387070629140639e-3, 3.3084409927100313e-3]),
    (StandardLink::LogLog, 30.0, [9.3576229688401746e-14, -9.3576229688401746e-14, 9.3576229688401746e-14, -9.3576229688401746e-14, 9.3576229688401746e-14], [-9.9999999999995321e-1, -4.6788114844199414e-14, 4.6788114844197954e-14, -4.6788114844195035e-14, 4.6788114844189198e-14]),
    (StandardLink::Cauchit, -1000.0, [9.9999933333391111e-4, 9.9999800000288889e-7, 1.9999920000173333e-9, 5.9999600001213331e-12, 2.3999760000970664e-14], [-3.1841092118452029e-7, -6.3692259106256461e-10, -1.9110694464058054e-12, -7.6454821863032887e-15, -3.8233421470248636e-17]),
    (StandardLink::Cauchit, -5.0, [1.9484500305269503e-1, 3.6975810574893353e-2, 1.3647509902799769e-2, 7.3251021926623434e-3, 5.0546731421167184e-3], [-1.306350670045737e-2, -5.1950808613349599e-3, -3.0614262454619924e-3, -2.3762318727302966e-3, -2.2770346757359458e-3]),
    (StandardLink::Cauchit, 0.2, [5.4379757581798629e-1, -5.0486871724166711e-1, -2.2204809555841153e-1, 2.754321883530863, -2.4852611667184961], [-7.001149880506487e-1, -2.2088600108906432e-1, 1.0184726253347992, 2.4716263265262138e-1, -1.0687445609603159e+1]),
    (StandardLink::Cauchit, 7.0, [6.6673431820092537e-3, -1.9113095560692763e-3, 8.1667936736328796e-4, -4.6235547257726236e-4, 3.2513946223025641e-4], [-1.4094725261064796e-1, 1.9599102712491621e-2, -5.3752438978273513e-3, 2.1790984973197073e-3, -1.1593117753743213e-3]),
    (StandardLink::Cauchit, 1000.0, [3.1841092118452029e-7, -6.3692259106256461e-10, 1.9110694464058054e-12, -7.6454821863032887e-15, 3.8233421470248636e-17], [-9.9999933333391111e-4, 9.9999800000288889e-7, -1.9999920000173333e-9, 5.9999600001213331e-12, -2.3999760000970664e-14]),
];

/// Whether a side's log-probability is a closed form in `η` (so its jet is
/// exact to rounding whatever that side's probability): logit's two sides are
/// both `−log(1+e^{∓η})`, cloglog's complement is `−e^η` and loglog's mean is
/// `−e^{−η}`.
fn side_is_closed_form(link: StandardLink, mean_side: bool) -> bool {
    match link {
        StandardLink::Logit => true,
        StandardLink::CLogLog => !mean_side,
        StandardLink::LogLog => mean_side,
        _ => false,
    }
}

#[test]
fn bernoulli_log_jets_match_a_400_digit_reference_3317() {
    let mut failures = Vec::new();
    for &(link, eta, log_mu, log_complement) in REFERENCE.iter() {
        let inverse_link = InverseLink::Standard(link);
        let jet = crate::mixture_link::bernoulli_log_jet5_for_inverse_link(&inverse_link, eta)
            .expect("the standard-link log jets are defined at every finite eta");
        let mu = crate::mixture_link::inverse_link_jet_for_inverse_link(&inverse_link, eta)
            .expect("the standard-link inverse-link jet is defined at every finite eta")
            .mu;
        let complement =
            crate::mixture_link::inverse_link_complement_for_inverse_link(&inverse_link, eta, mu);
        for (mean_side, got, want, probability) in [
            (true, jet.log_mu, log_mu, mu),
            (false, jet.log_complement, log_complement, complement),
        ] {
            // Where a side's own probability is small and it has no closed
            // form, its derivatives are built from density ratios of size
            // `ρ = 1 + |∂ log P|` and order `m` sums terms of size `ρ^m` to a
            // result of size `~ρ²`: the loss is absolute, `~ε·ρ^m` (the
            // module docs of `BernoulliLogJet`). Measured worst case
            // `4.8e-8` at probit `log Φ(−30)` order 5 against the `64ε·31^5
            // = 4.1e-7` allowance. Everywhere else the construction is
            // relative to rounding: measured worst `7e-14` (cloglog η = 5.26,
            // the mean side, `u·ε` conditioning of `1 − e^{−u}`).
            let rho = 1.0 + want[0].abs();
            let own_tail = probability < 0.5 && !side_is_closed_form(link, mean_side);
            for order in 0..5 {
                let cancellation = if own_tail {
                    64.0 * f64::EPSILON * rho.powi(order as i32 + 1)
                } else {
                    0.0
                };
                let band = 1e-12 * want[order].abs() + cancellation + f64::MIN_POSITIVE;
                let error = (got[order] - want[order]).abs();
                if !(error <= band) {
                    failures.push(format!(
                        "{link:?} eta={eta} {} order {}: got {:.17e}, want {:.17e} \
                         (error {error:.3e}, band {band:.3e})",
                        if mean_side { "log mu" } else { "log(1-mu)" },
                        order + 1,
                        got[order],
                        want[order],
                    ));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "#3317: Bernoulli log jets disagree with the 400-digit reference:\n  {}",
        failures.join("\n  ")
    );
}

#[test]
fn saturated_cloglog_row_has_representable_information_3317() {
    // The row the #3317 cloglog fit refused: `μ'` and `1 − μ` both underflow,
    // so the ratio tower's `μ'/(μ(1−μ))` was `0/0`. `−∂^k log μ`, `k = 2..5`,
    // from the 400-digit reference; the fifth is subnormal.
    const ETA: f64 = 6.654659400973018;
    const Y1_REFERENCE: [f64; 4] = [
        3.9465494539041362e-332,
        -3.0561769148352035e-329,
        2.3636153302233611e-326,
        -1.825622620999198e-323,
    ];
    let link = InverseLink::Standard(StandardLink::CLogLog);

    let one = bernoulli_observed_information_jet(&link, ETA, 1.0, 1.0, 1.0)
        .expect("the saturated cloglog row has an observed information");
    for (k, (&value, &reference)) in one.iter().zip(Y1_REFERENCE.iter()).enumerate() {
        assert!(
            value.is_finite() && (value - reference).abs() <= f64::MIN_POSITIVE,
            "#3317: y=1 order {k} at eta={ETA} is {value:e}, reference {reference:e}"
        );
    }

    // y = 0: −log(1−μ) = e^η, so every derivative is e^η, exactly.
    let zero = bernoulli_observed_information_jet(&link, ETA, 0.0, 1.0, 1.0)
        .expect("the saturated cloglog row has an observed information");
    assert_eq!(zero, [ETA.exp(); 4], "#3317: y=0 must be e^eta exactly");

    // A fractional response mixes the two sides by `y` and `1 − y` and scales
    // by `prior_weight/φ`.
    let (y, phi, prior_weight) = (0.25, 2.0, 3.0);
    let mixed = bernoulli_observed_information_jet(&link, ETA, y, phi, prior_weight)
        .expect("the saturated cloglog row has an observed information");
    for k in 0..4 {
        let want = prior_weight / phi * (y * one[k] + (1.0 - y) * zero[k]);
        assert!(
            (mixed[k] - want).abs() <= 4.0 * f64::EPSILON * want.abs(),
            "#3317: fractional y order {k}: {} against {want}",
            mixed[k]
        );
    }
}

#[test]
fn closed_form_log_jets_agree_with_the_generic_construction_3317() {
    // In the interior, where neither side's probability is small, the
    // link-specific constructions and the generic one (inverse-link jet
    // normalized by each side's probability) are both relative to rounding,
    // so they must agree.
    let links = [
        StandardLink::Logit,
        StandardLink::Probit,
        StandardLink::Cauchit,
        StandardLink::CLogLog,
        StandardLink::LogLog,
    ];
    let etas = [-3.0, -1.1, -0.2, 0.0, 0.6, 1.4, 2.5];
    let mut failures = Vec::new();
    for link in links {
        let inverse_link = InverseLink::Standard(link);
        for eta in etas {
            let special =
                crate::mixture_link::bernoulli_log_jet5_for_inverse_link(&inverse_link, eta)
                    .expect("standard-link log jets are defined");
            let generic =
                crate::mixture_link::bernoulli_log_jet5_from_inverse_link_jet(&inverse_link, eta)
                    .expect("standard-link log jets are defined");
            for (side, a, b) in [
                ("log mu", special.log_mu, generic.log_mu),
                ("log(1-mu)", special.log_complement, generic.log_complement),
            ] {
                // The generic route divides the inverse-link derivatives by
                // the side's probability, so its terms are `~ρ^m` with
                // `ρ = 1 + |∂ log P|` and it loses `~ε·ρ^m` absolutely where
                // that probability is small (loglog `η = −3`: `μ = 1.9e-9`,
                // `ρ = 21`, measured `9e-10` at order 5 against `ε·21⁵ =
                // 9.1e-10`). The link-specific route is the accurate one there.
                // Measured worst over this grid: 5.2 units of the model.
                let rho = 1.0 + a[0].abs();
                for order in 0..5 {
                    let model = f64::EPSILON * (a[order].abs() + rho.powi(order as i32 + 1));
                    let ratio = (a[order] - b[order]).abs() / model;
                    if !(ratio <= 64.0) {
                        failures.push(format!(
                            "{link:?} eta={eta} {side} order {}: {:e} against {:e} \
                             ({ratio:.1} units of the error model)",
                            order + 1,
                            a[order],
                            b[order],
                        ));
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "#3317:\n  {}", failures.join("\n  "));
}

#[test]
fn observed_information_matches_the_ratio_tower_3317() {
    use gam_problem::{LinkComponent, MixtureLinkState, SasLinkState};
    use ndarray::array;

    // Away from underflow the ratio tower it replaced for Binomial rows
    // (`observed_weight_dispatch` for `W, W', W''`, `e_obs_from_jets` for
    // `W'''`) is a correct evaluation of the same derivatives, so the two must
    // agree to the tower's error model on every link, including the flexible
    // ones that go through the generic construction.
    let sas = SasLinkState {
        epsilon: -0.3,
        log_delta: 0.25,
        delta: 0.25_f64.exp(),
    };
    let beta_logistic = SasLinkState {
        epsilon: 0.35,
        log_delta: 0.6,
        delta: 0.6_f64.exp(),
    };
    let mixture = MixtureLinkState {
        components: vec![LinkComponent::CLogLog, LinkComponent::Probit],
        rho: array![0.4],
        pi: array![0.6, 0.4],
    };
    let links = [
        InverseLink::Standard(StandardLink::Logit),
        InverseLink::Standard(StandardLink::Probit),
        InverseLink::Standard(StandardLink::Cauchit),
        InverseLink::Standard(StandardLink::CLogLog),
        InverseLink::Standard(StandardLink::LogLog),
        InverseLink::Sas(sas),
        InverseLink::BetaLogistic(beta_logistic),
        InverseLink::Mixture(mixture),
        InverseLink::LatentCLogLog(
            gam_problem::LatentCLogLogState::new(0.8).expect("0.8 is a valid latent SD"),
        ),
    ];
    let (phi, prior_weight) = (1.0, 1.7);
    let mut failures = Vec::new();
    for link in &links {
        for eta in [-2.5, -0.7, 0.3, 1.8] {
            let jet = crate::mixture_link::inverse_link_jet_for_inverse_link(link, eta)
                .expect("the inverse-link jet is defined");
            let h4 = crate::mixture_link::inverse_link_pdfthird_derivative_for_inverse_link(
                link, eta,
            )
            .expect("h4 is defined");
            let h5 = crate::mixture_link::inverse_link_pdffourth_derivative_for_inverse_link(
                link, eta,
            )
            .expect("h5 is defined");
            let one_minus_mu =
                crate::mixture_link::inverse_link_complement_for_inverse_link(link, eta, jet.mu);
            for y in [0.0, 0.35, 1.0] {
                let vj =
                    variance_jet_for_weight_family(WeightFamily::Binomial, jet.mu, one_minus_mu);
                let (w, c, d) = observed_weight_dispatch(
                    WeightFamily::Binomial,
                    WeightLink::Other,
                    y,
                    jet.mu,
                    one_minus_mu,
                    phi,
                    prior_weight,
                    jet,
                    h4,
                );
                let resid =
                    bernoulli_pair_residual(WeightFamily::Binomial, y, jet.mu, one_minus_mu);
                let e = e_obs_from_jets(
                    resid, jet.d1, jet.d2, jet.d3, h4, h5, vj, phi, prior_weight,
                );
                let tower = [w, c, d, e];
                let got = bernoulli_observed_information_jet(link, eta, y, phi, prior_weight)
                    .expect("the observed information is defined");
                // The tower's terms are the same `~ρ^m` density ratios per
                // side, so its error model is the one above, mixed by `y` and
                // `1 − y` and scaled by `prior_weight/φ`; the log-jet route is
                // the accurate one wherever the two differ. Measured worst over
                // this grid: 6.1 units (loglog `η = −2.5`, `μ = 5e-6`).
                let exact = crate::mixture_link::bernoulli_log_jet5_for_inverse_link(link, eta)
                    .expect("the log jets are defined");
                let rho_mu = 1.0 + exact.log_mu[0].abs();
                let rho_complement = 1.0 + exact.log_complement[0].abs();
                for k in 0..4 {
                    let order = k as i32 + 2;
                    let model = f64::EPSILON
                        * (got[k].abs()
                            + prior_weight / phi
                                * (y * rho_mu.powi(order)
                                    + (1.0 - y) * rho_complement.powi(order)));
                    let ratio = (tower[k] - got[k]).abs() / model;
                    if !(ratio <= 64.0) {
                        failures.push(format!(
                            "{link:?} eta={eta} y={y} entry {k}: log-jet {:e} against tower \
                             {:e} ({ratio:.1} units of the error model)",
                            got[k], tower[k],
                        ));
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "#3317:\n  {}", failures.join("\n  "));
}

#[test]
fn latent_cloglog_binomial_takes_the_observed_information_3802() {
    use gam_problem::{GlmLikelihoodSpec, LatentCLogLogState, LikelihoodSpec, ResponseFamily};

    // `fit.rs` upgrades a Binomial cloglog fit with `latent_cloglog` set to
    // this link, so this is the spec the PIRLS working model gates on.
    let link =
        InverseLink::LatentCLogLog(LatentCLogLogState::new(0.8).expect("0.8 is a valid latent SD"));
    let likelihood =
        GlmLikelihoodSpec::canonical(LikelihoodSpec::new(ResponseFamily::Binomial, link.clone()));
    assert!(
        supports_observed_hessian_curvature_for_likelihood(&likelihood, &link),
        "#3802: the latent-cloglog Binomial fit must price its LAML with the observed information"
    );

    // `μ(η) = E[1 − exp(−Z e^η)]` is not the canonical Bernoulli link, so the
    // observed information `W_F − (y − μ)·B` depends on the response through
    // `B ≠ 0`. It is linear in `y` and equals the Fisher weight
    // `μ'²/(μ(1−μ))` at `y = μ`, and `w(1) − w(0) = −B` must stand clear of
    // rounding.
    for eta in [-2.0, -0.4, 0.9] {
        let jet = crate::mixture_link::inverse_link_jet_for_inverse_link(&link, eta)
            .expect("the latent-cloglog jet is defined");
        let complement =
            crate::mixture_link::inverse_link_complement_for_inverse_link(&link, eta, jet.mu);
        let fisher = jet.d1 * jet.d1 / (jet.mu * complement);
        let at = |y: f64| {
            bernoulli_observed_information_jet(&link, eta, y, 1.0, 1.0)
                .expect("the observed information is defined")[0]
        };
        let (w_mean, w0, w1) = (at(jet.mu), at(0.0), at(1.0));
        // Each side is a sum of a few `O(fisher)` density ratios, so the
        // agreement at `y = μ` is to rounding times that scale.
        let rounding = 64.0 * f64::EPSILON * (fisher + w0.abs() + w1.abs());
        assert!(
            (w_mean - fisher).abs() <= rounding,
            "#3802 eta={eta}: observed information at y = mu is {w_mean:e}, Fisher {fisher:e}"
        );
        assert!(
            (w1 - w0).abs() > rounding,
            "#3802 eta={eta}: the observed information does not depend on y \
             (w(0) = {w0:e}, w(1) = {w1:e}), so the link would be canonical"
        );
    }
}
