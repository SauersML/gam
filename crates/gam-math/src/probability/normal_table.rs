//! The table route for the standard normal's `Φ`, `φ` and scaled tail `Q(t) = e^{t²/2}·Φ(−t)` (#2946, #2928).
//!
//! It reads one exponential for both `Φ` and `φ`, calls no `erfc`, and uses no fused multiply-add in its polynomials:
//! generic x86-64 builds lower `mul_add` to a library call.
//! - **φ** is [`normal_pdf`](super::normal_pdf)'s own computation, shared through its parts, so it is bitwise
//!   `normal_pdf(x)`.
//! - **Central, `|x| < 0.75`:** `Φ(x) = ½ + x·P(x²)`, with `P` of degree 10 and its constant term split high and low.
//! - **Otherwise, `t = |x|`:** `Φ(−t) = E·Q(t)·(1 − ½·res)`, where `E = exp(−½·fl(t²))` is `φ`'s exponential and
//!   `res = t² − fl(t²)` is its exact square residual. Then `Φ(t) = 1 − Φ(−t)`.
//! - **`Q` on `[0, 8)`:** 16 pieces of width `½` about their midpoints `c`. Each is `c₀ʰⁱ + (c₀ˡᵒ + v·E(v))` with
//!   `v = t − c`, degree 14, and `E` by Estrin's scheme.
//! - **`Q` on `[8, ∞)`:** `Q = r·K(r²)`, `r = 1/t`, with `K` of degree 12 in `s = r²` on `[0, 1/64]`.
//! - **`N = 1 − t·R` for the Mills owner:** its own 16 pieces at degree 14 and, past 8, `N = s·K_N(s)` with `K_N` of
//!   degree 13, so the owner never forms the cancelling `1 − t·R`. The owner `normal_left_tail_ratios` reads `R = Q·√(2π)`
//!   and `N` from here, with bounds from the proven constants.
//!
//! The coefficients are gam-2928's 48-node Chebyshev interpolants at 50 digits (`phi_fit.py 0.75 16 14 12 10 0.5`),
//! rounded to `f64`. `N`'s come from the same generator fitting `1 − t·M(t)` (fr-kernel's `n_fit.py`, MSI job 1252825). This module's tests PROVE the published constants [`NORMAL_SCALED_TAIL_RELATIVE_ERROR`] and
//! [`NORMAL_CDF_RELATIVE_ERROR`] from IEEE-754 semantics and the cited `libm::exp` contract. They do not rely on a
//! grid of measurements. Each piece's approximation error is bounded by a certified Taylor model of `Q` minus the
//! stored polynomial, over a covering of the piece. The evaluation's rounding is propagated a priori through the same
//! operation tree.

use super::{normal_density_parts, finish_normal_density, square_residual};

/// `|x|` below which `Φ(x) = ½ + x·P(x²)` needs no exponential.
const CENTRAL_REACH: f64 = 0.75;

/// Degree of the central `P` in `y = x²`.
const CENTRAL_DEGREE: usize = 10;

/// `P(y) = (Φ(√y) − ½)/√y ≈ Σ_k CENTRAL[k]·y^k` on `y ∈ [0, 0.5625]`.
const CENTRAL: [f64; CENTRAL_DEGREE + 1] = [
    0.3989422804014327,
    -0.06649038006690544,
    0.009973557010035812,
    -0.0011873282154803186,
    0.00011543468761421972,
    -9.444656243406497e-06,
    6.659692681111987e-07,
    -4.122639550167386e-08,
    2.2729313014180584e-09,
    -1.1220981707762466e-10,
    4.498387632764599e-12,
];

/// The low word of `CENTRAL[0]`.
const CENTRAL_LOW: f64 = -2.492343151805567e-17;

/// Pieces of `Q` on `[0, 8)`.
const PIECE_COUNT: usize = 16;

/// Width of each piece.
const PIECE_WIDTH: f64 = 0.5;

/// `1/PIECE_WIDTH`: `floor(t·PIECE_SCALE)` indexes the piece.
const PIECE_SCALE: f64 = 2.0;

/// Where the pieces end and the reciprocal tail begins.
const TABLE_REACH: f64 = PIECE_COUNT as f64 * PIECE_WIDTH;

/// Degree of each piece in `v = t − c`.
const PIECE_DEGREE: usize = 14;

/// Degree of the tail `K` in `s = 1/t²`.
const TAIL_DEGREE: usize = 12;

/// `K(s) = t·Q(t) ≈ Σ_k TAIL[k]·s^k` on `s ∈ [0, 1/64]`; `K(0) = 1/√(2π)`.
const TAIL: [f64; TAIL_DEGREE + 1] = [
    0.3989422804014327,
    -0.398942280401432,
    1.1968268412017982,
    -5.984134202434688,
    41.88893675765726,
    -376.9992527432712,
    4146.656733325157,
    -53842.68688465505,
    799313.0882046289,
    -12842943.286401203,
    198986140.40721336,
    -2424345162.1884503,
    15967161721.030256,
];

/// The low word of `TAIL[0]`.
const TAIL_LOW: f64 = -2.4954594820767507e-17;

/// `c₀ʰⁱ` of each piece.
const PIECE_CONSTANTS: [f64; PIECE_COUNT] = [
    0.4140321029477354,
    0.30023246233995093,
    0.23076032130563176,
    0.18523166467823896,
    0.15365193742384164,
    0.13072473410074711,
    0.11345206212929865,
    0.10003920963545321,
    0.08935931861967142,
    0.08067539917254936,
    0.07348823085269288,
    0.0674492313514587,
    0.062308486908362076,
    0.057882631723879995,
    0.05403435940923554,
    0.05065898233519691,
];

/// `c₁, …, c₁₄` of each piece: `Q(c + v) ≈ c₀ʰⁱ + (c₀ˡᵒ + v·Σ_{k≥1} c_k v^{k−1})`.
const PIECE_SLOPES: [[f64; PIECE_DEGREE]; PIECE_COUNT] = [
    [
        -0.2954342546644988, 0.17008676964080535, -0.084304187418099, 0.037252680696570055, -0.014998203448817955,
        0.005583854972406679, -0.0019431770987853806, 0.0006372575864395057, -0.00019820704636264343,
        5.877060873589931e-05, -1.6681462354123552e-05, 4.5495267266032806e-06, -1.2134536601501183e-06,
        3.081327789250703e-07,
    ],
    [
        -0.17376793364646947, 0.08495325605254941, -0.03668433053568578, 0.014360002037696253, -0.0051828658014869,
        0.001745475447765703, -0.0005533941733473766, 0.00016630372708469497, -4.7629610026169506e-05,
        1.3058156469670319e-05, -3.4393745448796658e-06, 8.731348112202633e-07, -2.169800003207015e-07,
        5.155834849309245e-08,
    ],
    [
        -0.11049187876939297, 0.04632273642194529, -0.01752948608065378, 0.006102719705282011,
        -0.0019802172898109867, 0.0006045746821701409, -0.00017492841952956934, 4.823926969374819e-05,
        -1.2736594677007818e-05, 3.2318534982440438e-06, -7.905703566309766e-07, 1.8695411489210355e-07,
        -4.332540498525887e-08, 9.639037622548816e-09,
    ],
    [
        -0.0747868672145145, 0.027177323526419297, -0.00907551701442691, 0.00282379218779305, -0.000826776137157957,
        0.00022948899129452958, -6.0738628902340416e-05, 1.539954883388376e-05, -3.754380252519068e-06,
        8.829385217211864e-07, -2.0083089001798794e-07, 4.4287028850279665e-08, -9.581901590101285e-09,
        1.9976257448897437e-09,
    ],
    [
        -0.05322542119778899, 0.016947369864408205, -0.005031279667623509, 0.0014067476530638273,
        -0.00037321948964601014, 9.450063356007237e-05, -2.2941866302790086e-05, 5.360179296054791e-06,
        -1.2090515230303164e-06, 2.639813792468096e-07, -5.591567362616057e-08, 1.1513488996374756e-08,
        -2.3288614953798454e-09, 4.5544522525078e-10,
    ],
    [
        -0.03944926162437811, 0.011119632316853652, -0.0029567575843435245, 0.0007471372399772397,
        -0.00018042603488123023, 4.1827607342314954e-05, -9.342873526577308e-06, 2.0168381426995234e-06,
        -4.2184098071320236e-07, 8.567755526620599e-08, -1.6929343745248352e-08, 3.259961357269259e-09,
        -6.174238132370285e-10, 1.13409212176875e-10,
    ],
    [
        -0.030223078481212095, 0.007613528532679665, -0.0018263702500010619, 0.0004194563050440534,
        -9.262745172157947e-05, 1.9736181158154886e-05, -4.069266136657609e-06, 8.13883276661834e-07,
        -1.5823839293697438e-07, 2.9960852885767776e-08, -5.533124401003551e-09, 9.98130493095167e-10,
        -1.773071324699882e-10, 3.063187144849871e-11,
    ],
    [
        -0.023795244268483163, 0.005403521814320675, -0.0011773458215935434, 0.0002471187458362217,
        -5.013010494154288e-05, 9.855142050906446e-06, -1.8819031786252173e-06, 3.497506413560378e-07,
        -6.337092077527583e-08, 1.1210969705232346e-08, -1.9390400313632804e-09, 3.282820116524108e-10,
        -5.4793832905849955e-11, 8.917083130051817e-12,
    ],
    [
        -0.019165176267829143, 0.003953659740698779, -0.0007873741232864444, 0.00015182992918284763,
        -2.8419384851868537e-05, 5.174590593734534e-06, -9.181964040597195e-07, 1.590319845516588e-07,
        -2.6923385963879282e-08, 4.460759691321144e-09, -7.240959904643995e-10, 1.1527446242064702e-10,
        -1.8112127341714227e-11, 2.7810092143309264e-12,
    ],
    [
        -0.015734134331823208, 0.0029691305481945587, -0.000543588075966352, 9.677179683859657e-05,
        -1.6784408196603713e-05, 2.84097631745487e-06, -4.6996724123833195e-07, 7.607899019376673e-08,
        -1.2065782118121253e-08, 1.876652603971958e-09, -2.8651366998688545e-10, 4.297443648903612e-11,
        -6.3682313767100576e-12, 9.24091113912502e-13,
    ],
    [
        -0.013129068424795096, 0.0022803108112593095, -0.00038581222189457365, 6.369916157819948e-05,
        -1.027832472180528e-05, 1.6229927981203118e-06, -2.5108750452362524e-07, 3.80979249204028e-08,
        -5.674822123024328e-09, 8.305109096227172e-10, -1.1951174583577898e-10, 1.6922280649984088e-11,
        -2.369535015928468e-12, 3.255028333337767e-13,
    ],
    [
        -0.011109200130545225, 0.0017856653004118203, -0.00028054155105908637, 4.313784545551842e-05,
        -6.499787937971102e-06, 9.606774686974363e-07, -1.3941321327970374e-07, 1.988143654201512e-08,
        -2.7883281455322447e-09, 3.8485498254872563e-10, -5.230984005660777e-11, 7.00590304837481e-12,
        -9.287108401818937e-13, 1.2097686583604136e-13,
    ],
    [
        -0.009514237224169695, 0.0014222521286507408, -0.00020838714003418857, 2.9958125859265542e-05,
        -4.229770682755791e-06, 5.870098486736438e-07, -8.013701836349065e-08, 1.076918548758e-08,
        -1.4255121245081872e-09, 1.8597347566189106e-10, -2.3925137031019443e-11, 3.0366969937895575e-12,
        -3.8179683420228297e-13, 4.7240308133848425e-14,
    ],
    [
        -0.008234516265242704, 0.0011498234667458754, -0.00015773595490268152, 2.1276442788193783e-05,
        -2.8239932164746954e-06, 3.6908142949826604e-07, -4.7527652480139715e-08, 6.033721907104344e-09,
        -7.555588475978079e-10, 9.336997052037243e-11, -1.13919094487594e-11, 1.3728474713681907e-12,
        -1.6400233250557688e-13, 1.9306571539871965e-14,
    ],
    [
        -0.007193174684475032, 0.0009419214733957776, -0.00012141466745188136, 1.5416283592409456e-05,
        -1.9293222813825617e-06, 2.3811617539764739e-07, -2.8997144249921003e-08, 3.4858599481888597e-09,
        -4.138510704631069e-10, 4.854396956268267e-11, -5.627915503156202e-12, 6.451171715057977e-13,
        -7.335362719730399e-14, 8.229068153304929e-15,
    ],
    [
        -0.006335167303656634, 0.0007807178659289988, -9.486794756896433e-05, 1.13728180673813e-05,
        -1.345721509351854e-06, 1.572460616507385e-07, -1.8152075936936872e-08, 2.0709341424230745e-09,
        -2.3359292631164656e-10, 2.605889671965958e-11, -2.8760348515060505e-12, 3.1412903374369907e-13,
        -3.405476532687582e-14, 3.6463614365848224e-15,
    ],
];

/// `c₀ˡᵒ`, the low word of each piece's constant term.
const PIECE_LOWS: [f64; PIECE_COUNT] = [
    1.6592883912142788e-17,
    2.3538007808179885e-18,
    1.2757613745174399e-17,
    5.204928156040966e-18,
    -5.693933663763213e-18,
    1.1881942860320672e-19,
    -6.8659539044850795e-18,
    -3.4263544572261513e-18,
    1.3396901271902678e-18,
    3.247075259999749e-18,
    -3.487919548572928e-18,
    -6.488171234801063e-18,
    9.573089039174836e-19,
    1.7786976342870805e-18,
    -1.0044018033117998e-18,
    -1.1978666387357065e-18,
];

/// `c₀ʰⁱ` of each piece of `N(t) = 1 − t·R(t)`.
const RATIO_PIECE_CONSTANTS: [f64; PIECE_COUNT] = [
    0.7405438560365682,
    0.43557161570244396,
    0.27696206744046115,
    0.1874628759309762,
    0.1334163457035221,
    0.09888463460098185,
    0.075758023067398,
    0.05964583208513115,
    0.048039972721227564,
    0.039439625992990404,
    0.03290969413315648,
    0.027846635155759063,
    0.023848656037650524,
    0.020640871298366226,
    0.01803061504846504,
    0.015879909487863556,
];

/// `c₁, …, c₁₄` of each piece of `N`, in `v = t − c`.
const RATIO_PIECE_SLOPES: [[f64; PIECE_DEGREE]; PIECE_COUNT] = [
    [
        -0.8526886118445848, 0.6339577795559952, -0.3735144909592913, 0.18797460416704173, -0.08397989253172607,
        0.0340957586447089, -0.012778943069949074, 0.0044714806849555895, -0.0014731608479843724,
        0.00046000242503473125, -0.0001368455421591401, 3.8962582985689804e-05, -1.0824578250470156e-05,
        2.8515511268338284e-06,
    ],
    [
        -0.4258924672865751, 0.2758619404699783, -0.14398074852579273, 0.06495758980807313, -0.02625154866027486,
        0.0097100743791005, -0.003334892995130857, 0.0010745072631902855, -0.0003273194646617329,
        9.484077778911836e-05, -2.6263179416830465e-05, 6.97966331310171e-06, -1.8109983183051239e-06,
        4.4742565775444494e-07,
    ],
    [
        -0.23222776174705465, 0.131819716348552, -0.06118899906163061, 0.02481834324275774, -0.009092663954724425,
        0.0030693636579876642, -0.0009673433387689989, 0.0002873331256615188, -8.10105572834271e-05,
        2.1799590677833037e-05, -5.623441954905428e-06, 1.3958675517491631e-06, -3.3853646188469985e-07,
        7.848051764869553e-08,
    ],
    [
        -0.1362468951602338, 0.06824684266577165, -0.028312789358416763, 0.010362102210949874,
        -0.0034514615657728715, 0.0010657441521736348, -0.0003088075561654882, 8.469751163993713e-05,
        -2.2131987310755052e-05, 5.537738332735322e-06, -1.332123961669695e-06, 3.091354521475096e-07,
        -7.015274132739039e-08, 1.527137825834088e-08,
    ],
    [
        -0.08496151296550991, 0.03783464361732345, -0.014104773769762362, 0.004677612626949821,
        -0.0014212677603134003, 0.00040254711527260424, -0.00010748781583999962, 2.727578253833789e-05,
        -6.617032028696838e-06, 1.541810621462462e-06, -3.463185353719581e-07, 7.522397992953457e-08,
        -1.5993029367602074e-08, 3.2723249560941693e-09,
    ],
    [
        -0.055745569537851966, 0.022234476486435402, -0.007491181323026864, 0.002261305002564164,
        -0.0006290775793464763, 0.00016393377664573758, -4.044370810972971e-05, 9.516586284307223e-06,
        -2.1476178555130606e-06, 4.6680364269749886e-07, -9.80577140777726e-08, 1.9963740985241516e-08,
        -3.982095237869605e-09, 7.666946694597927e-10,
    ],
    [
        -0.03816857177944944, 0.013734093925792652, -0.004205684136782681, 0.0011609129474616222,
        -0.0002968276183456279, 7.140096288845303e-05, -1.63208226679918e-05, 3.5698033460095873e-06,
        -7.510072170071832e-07, 1.5256749051621764e-07, -3.002320412585076e-08, 5.738129758772154e-09,
        -1.0754980820952524e-09, 1.9510238172093123e-10,
    ],
    [
        -0.027089241124723214, 0.008853504976275125, -0.002477739342017701, 0.0006282876922834468,
        -0.00014821906629184332, 3.302062202429018e-05, -7.013558773482807e-06, 1.429626042515088e-06,
        -2.8101733837370246e-07, 5.346584166917453e-08, -9.874545389507751e-09, 1.7746367168083348e-09,
        -3.1306553792294123e-10, 5.358426621733091e-11,
    ],
    [
        -0.019820710588611658, 0.0059209627204277904, -0.0015223247736997933, 0.00035618416808656626,
        -7.782465055136788e-05, 1.611103947676187e-05, -3.1890725523650573e-06, 6.073822746139914e-07,
        -1.1181466421077397e-07, 1.9965688748401478e-08, -3.4673953943814258e-09, 5.869974224923751e-10,
        -9.763247007506325e-11, 1.579024796347142e-11,
    ],
    [
        -0.014885013166350247, 0.004087719722908567, -0.0009702836885698921, 0.00021036136079277538,
        -4.2727629389336695e-05, 8.24623222533047e-06, -1.5256139833952318e-06, 2.721998723901053e-07,
        -4.704070494412691e-08, 7.900095935830965e-09, -1.292649067626903e-09, 2.0650768168499997e-10,
        -3.24405858698754e-11, 4.965282866700602e-12,
    ],
    [
        -0.011431783108898681, 0.0029012634722974436, -0.000638680477928814, 0.00012881969681757946,
        -2.440943782374517e-05, 4.405681267738958e-06, -7.639786864803164e-07, 1.2802202524025946e-07,
        -2.0817821334039054e-08, 3.2953134451642845e-09, -5.090137140106523e-10, 7.68773411153579e-11,
        -1.1426493923720664e-11, 1.6577180413578226e-12,
    ],
    [
        -0.008951998262079456, 0.002109640152280627, -0.0004325221725018595, 8.146276112211939e-05,
        -1.44483678350276e-05, 2.4461997158557933e-06, -3.986829678114104e-07, 6.29037191487213e-08,
        -9.646883825504584e-09, 1.4423439898774575e-09, -2.1073410408028995e-10, 3.014432153474611e-11,
        -4.246653597494099e-12, 5.848887516805135e-13,
    ],
    [
        -0.007130114798660148, 0.0015670472918375604, -0.0003003755413351566, 5.301231394300466e-05,
        -8.828492905033325e-06, 1.4061160125245637e-06, -2.159547587031108e-07, 3.2159060837406513e-08,
        -4.6616637301254275e-09, 6.596891737135828e-10, -9.134236458040986e-11, 1.239718457222084e-11,
        -1.6582316784736154e-12, 2.1716206704132242e-13,
    ],
    [
        -0.005764360025158898, 0.0011861562134549456, -0.0002133285323058215, 3.5393506218908065e-05,
        -5.550899680929493e-06, 8.339391027359388e-07, -1.2099438346879162e-07, 1.7045146481628325e-08,
        -2.340438083256084e-09, 3.1410946746310854e-10, -4.129458905185399e-11, 5.3270396246589856e-12,
        -6.77683760137331e-13, 8.45187164154769e-14,
    ],
    [
        -0.004722093995391896, 0.0009130243151694181, -0.00015457156936985405, 2.418046890694558e-05,
        -3.5812124273924257e-06, 5.087954316232701e-07, -6.99020408602428e-08, 9.336337130755915e-09,
        -1.2168168675315651e-09, 1.5517856703411363e-10, -1.9404815509964722e-11, 2.3833344664998414e-12,
        -2.8884394518882096e-13, 3.435875068901884e-14,
    ],
    [
        -0.003913938954494406, 0.0007133960391977319, -0.00011402970931972901, 1.6866117925602318e-05,
        -2.364944545248664e-06, 3.1850354750760684e-07, -4.1528496610359394e-08, 5.269775695465885e-09,
        -6.531996735737891e-10, 7.930088737657639e-11, -9.448851840041122e-12, 1.1067604011445319e-12,
        -1.2798648057742356e-13, 1.4542219623299583e-14,
    ],
];

/// `c₀ˡᵒ` of each piece of `N`.
const RATIO_PIECE_LOWS: [f64; PIECE_COUNT] = [
    4.81551362903206e-17,
    1.9801381131246917e-18,
    8.20174279894205e-18,
    -1.5881983541265193e-18,
    3.3760832139636134e-18,
    4.4037949943690605e-18,
    -2.848998229225763e-18,
    2.1555959487936727e-18,
    6.258570530803551e-19,
    -2.8477357119177276e-18,
    -3.202400488811176e-18,
    8.2069754484569645e-19,
    -1.4562393400950021e-18,
    1.1506412831887225e-18,
    7.900464083718822e-20,
    -7.676306021364285e-19,
];

/// Degree of the tail `K_N(s) = t²·N(t)` in `s = 1/t²`.
const RATIO_TAIL_DEGREE: usize = 13;

/// `K_N(s) = t²·N(t) ≈ Σ_k RATIO_TAIL[k]·s^k` on `s ∈ [0, 1/64]`; `K_N(0) = 1`.
const RATIO_TAIL: [f64; RATIO_TAIL_DEGREE + 1] = [
    1.0,
    -2.9999999999999964,
    14.999999999984041,
    -104.99999997334433,
    944.9999766792711,
    -10394.987716488944,
    135130.7793826614,
    -2026030.9334499883,
    34293872.1230369,
    -634861199.2409745,
    12012340174.417112,
    -204771054991.08054,
    2585671197637.3213,
    -16966527494740.004,
];

/// The low word of `RATIO_TAIL[0]`.
const RATIO_TAIL_LOW: f64 = -1.4836642522860837e-19;

/// A bound on the relative error of [`normal_scaled_tail`] at every `t ≥ 0`, relative to the COMPUTED value:
/// `|Q̂ − Q(t)| ≤ NORMAL_SCALED_TAIL_RELATIVE_ERROR·Q̂`. It is derived, not measured.
/// `scaled_tail_constant_covers_its_proof_and_is_within_twice_of_it` recomputes the proof: each piece's certified
/// approximation error plus its a priori evaluation rounding, and the same for the reciprocal tail. It asserts that this
/// constant covers the worst and is within twice of it.
pub const NORMAL_SCALED_TAIL_RELATIVE_ERROR: f64 = 4.0e-16;

/// A bound on the relative error of [`positive_part_ratio`] at every `t ≥ 0` where its value is a normal double,
/// relative to the COMPUTED value. It is derived the way [`NORMAL_SCALED_TAIL_RELATIVE_ERROR`] is, in
/// `positive_part_ratio_constant_covers_its_proof_and_is_within_twice_of_it`. The reciprocal tail's `s = fl(r²)` enters
/// `N = s·K_N` undamped, so it adds about `3u` that `Q`'s `r·K` does not.
pub(super) const POSITIVE_PART_RATIO_RELATIVE_ERROR: f64 = 8.0e-16;

/// A bound on the relative error of [`normal_cdf_and_pdf`]'s `Φ`, relative to the COMPUTED value, wherever that value is
/// a normal double: `|Φ̂ − Φ(x)| ≤ NORMAL_CDF_RELATIVE_ERROR·Φ̂ + NORMAL_CDF_UNDERFLOW_FLOOR`.
///
/// Derived in `normal_cdf_constant_covers_its_proof_and_is_within_twice_of_it`. The terms are the central polynomial's
/// proof, [`NORMAL_SCALED_TAIL_RELATIVE_ERROR`]'s proof, and the cited `libm::exp` contract behind `φ`'s exponential
/// (libm 0.2.16 `src/math/exp.rs:58-60`, pinned by `libm_version_matches_the_cited_error_analysis`). The rest is the
/// rounding of the product, the square-residual correction and the complement.
pub const NORMAL_CDF_RELATIVE_ERROR: f64 = 9.0e-16;

/// The absolute floor under [`NORMAL_CDF_RELATIVE_ERROR`]: where `Φ(−t)` lands among the subnormals or underflows
/// (`t ≳ 37.5`), each rounded step errs by up to `η/2` beyond its relative band, `η = 2⁻¹⁰⁷⁴`. The exponential, the
/// product, the correction and the complement give four such steps.
pub const NORMAL_CDF_UNDERFLOW_FLOOR: f64 = 4.0 * f64::from_bits(1);

/// Estrin's scheme for `Σ_k coefficients[k]·argument^k`: pairs `a + b·p` level by level, with `p` squared between
/// levels. Plain products and sums, so the tests' a priori rounding bound walks the same tree.
#[inline(always)]
fn estrin<const N: usize>(coefficients: &[f64; N], argument: f64) -> f64 {
    let mut terms = *coefficients;
    let mut count = N;
    let mut power = argument;
    while count > 1 {
        let half = count.div_ceil(2);
        let mut index = 0;
        while index < half {
            terms[index] = if 2 * index + 1 < count {
                terms[2 * index] + terms[2 * index + 1] * power
            } else {
                terms[2 * index]
            };
            index += 1;
        }
        count = half;
        power *= power;
    }
    terms[0]
}

/// `Q(t)` for `t ≥ 0` (or `+∞`, giving 0; NaN propagates) from the table.
#[inline]
pub(super) fn scaled_tail(t: f64) -> f64 {
    if t < TABLE_REACH {
        let index = (t * PIECE_SCALE) as usize;
        let offset = t - (index as f64 + 0.5) * PIECE_WIDTH;
        PIECE_CONSTANTS[index] + (PIECE_LOWS[index] + offset * estrin(&PIECE_SLOPES[index], offset))
    } else if t.is_finite() {
        let reciprocal = 1.0 / t;
        let square = reciprocal * reciprocal;
        let mut sum = TAIL[TAIL_DEGREE];
        for &coefficient in TAIL[1..TAIL_DEGREE].iter().rev() {
            sum = sum * square + coefficient;
        }
        reciprocal * (TAIL[0] + (square * sum + TAIL_LOW))
    } else if t.is_nan() {
        t
    } else {
        0.0
    }
}

/// `N(t) = E[(E − t)₊]/φ(t) = 1 − t·R(t)` for `t ≥ 0` (or `+∞`, giving 0; NaN propagates) from its own table, so the
/// cancellation of `1 − t·R` is never formed. The pieces mirror `Q`'s. Past `t = 8`, `N = s·K_N(s)` with `s = r²`,
/// `r = 1/t`, and `K_N` of degree 13 in `s` with `K_N(0) = 1`.
#[inline]
pub(super) fn positive_part_ratio(t: f64) -> f64 {
    if t < TABLE_REACH {
        let index = (t * PIECE_SCALE) as usize;
        let offset = t - (index as f64 + 0.5) * PIECE_WIDTH;
        RATIO_PIECE_CONSTANTS[index]
            + (RATIO_PIECE_LOWS[index] + offset * estrin(&RATIO_PIECE_SLOPES[index], offset))
    } else if t.is_finite() {
        let reciprocal = 1.0 / t;
        let square = reciprocal * reciprocal;
        let mut sum = RATIO_TAIL[RATIO_TAIL_DEGREE];
        for &coefficient in RATIO_TAIL[1..RATIO_TAIL_DEGREE].iter().rev() {
            sum = sum * square + coefficient;
        }
        square * (RATIO_TAIL[0] + (square * sum + RATIO_TAIL_LOW))
    } else if t.is_nan() {
        t
    } else {
        0.0
    }
}

/// The scaled normal tail `Q(t) = e^{t²/2}·Φ(−t) = R(t)/√(2π)` for `t ≥ 0`, where `R` is Mills' ratio. It is 0 at
/// `+∞`, NaN in gives NaN out, and a negative argument (outside the table's domain) gives NaN.
///
/// No libm call. Its relative error is at most [`NORMAL_SCALED_TAIL_RELATIVE_ERROR`] of the computed value,
/// proven in this module's tests.
#[inline]
pub fn normal_scaled_tail(t: f64) -> f64 {
    if t >= 0.0 { scaled_tail(t) } else { f64::NAN }
}

/// `(Φ(x), φ(x))` for the standard normal, from one exponential.
///
/// `φ` is bitwise [`normal_pdf`](super::normal_pdf)`(x)`, and symmetric in `±x`. `Φ` is within
/// [`NORMAL_CDF_RELATIVE_ERROR`] of its computed value, plus [`NORMAL_CDF_UNDERFLOW_FLOOR`], and `φ` within
/// [`normal_pdf_bounded`](super::normal_pdf_bounded)'s bound. `±∞` give `(0, 0)` and `(1, 0)`, and NaN propagates.
#[inline]
pub fn normal_cdf_and_pdf(x: f64) -> (f64, f64) {
    let parts = normal_density_parts(x);
    let density = finish_normal_density(x, parts);
    let magnitude = x.abs();
    if magnitude < CENTRAL_REACH {
        let square = parts.rounded_square;
        let mut sum = CENTRAL[CENTRAL_DEGREE];
        for &coefficient in CENTRAL[1..CENTRAL_DEGREE].iter().rev() {
            sum = sum * square + coefficient;
        }
        let polynomial = CENTRAL[0] + (square * sum + CENTRAL_LOW);
        return (0.5 + x * polynomial, density);
    }
    let lower = parts.exponential * scaled_tail(magnitude);
    if parts.head == 0.0 || parts.head.is_nan() {
        // φ underflowed, x is ±∞, or x is NaN: the correction has nothing to correct, and ±∞ would feed it ∞ − ∞.
        return (if x < 0.0 { lower } else { 1.0 - lower }, density);
    }
    let residual = square_residual(x, parts.rounded_square);
    let lower = lower - 0.5 * residual * lower;
    (if x < 0.0 { lower } else { 1.0 - lower }, density)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::double_double::{BoundedDoubleDouble, DoubleDouble};
    use crate::probability::{normal_left_tail_ratios, normal_pdf, normal_pdf_bounded};
    use crate::roundoff::{UNIT_ROUNDOFF, inflated};

    type Word = BoundedDoubleDouble;

    /// The smallest positive subnormal.
    const ETA: f64 = f64::from_bits(1);

    /// Taylor order of the certified models. Past it, the remainders are below `1e-30` of `Q` on every piece.
    const MODEL_ORDER: usize = 30;

    /// Cells of each piece's covering. Their half-width times the slope bound adds under `1e-18` to the proof.
    const PIECE_CELLS: usize = 8192;

    /// Cells of each reciprocal-tail covering, where the difference polynomial is far flatter.
    const TAIL_CELLS: usize = 1024;

    /// Unit sub-intervals of the reciprocal tail below `ASYMPTOTIC_REACH`.
    const TAIL_INTERVALS: usize = 56;

    /// Where the covering by unit sub-intervals stops and the enveloping asymptotic series takes over.
    const ASYMPTOTIC_REACH: f64 = TABLE_REACH + TAIL_INTERVALS as f64;

    /// Terms of the asymptotic series kept: its first omitted term at `t = 64` is below `1e-35` of either kernel.
    const ASYMPTOTIC_TERMS: usize = 15;

    /// Which table a proof reads: `Q = R/√(2π)`, or `N = 1 − t·R`.
    #[derive(Clone, Copy, Debug, PartialEq)]
    enum Target {
        ScaledTail,
        PositivePartRatio,
    }

    impl Target {
        fn constants(self) -> &'static [f64; PIECE_COUNT] {
            match self {
                Target::ScaledTail => &PIECE_CONSTANTS,
                Target::PositivePartRatio => &RATIO_PIECE_CONSTANTS,
            }
        }

        fn slopes(self) -> &'static [[f64; PIECE_DEGREE]; PIECE_COUNT] {
            match self {
                Target::ScaledTail => &PIECE_SLOPES,
                Target::PositivePartRatio => &RATIO_PIECE_SLOPES,
            }
        }

        fn lows(self) -> &'static [f64; PIECE_COUNT] {
            match self {
                Target::ScaledTail => &PIECE_LOWS,
                Target::PositivePartRatio => &RATIO_PIECE_LOWS,
            }
        }

        fn tail(self) -> &'static [f64] {
            match self {
                Target::ScaledTail => &TAIL,
                Target::PositivePartRatio => &RATIO_TAIL,
            }
        }

        fn tail_low(self) -> f64 {
            match self {
                Target::ScaledTail => TAIL_LOW,
                Target::PositivePartRatio => RATIO_TAIL_LOW,
            }
        }

        /// The stored reciprocal form is `F(t) = Σ_j κ_j·t^{−(2j + power)}`: `Q = r·K(r²)`, `N = r²·K_N(r²)`.
        fn power(self) -> usize {
            match self {
                Target::ScaledTail => 1,
                Target::PositivePartRatio => 2,
            }
        }
    }

    fn exact(value: f64) -> Word {
        Word::exact(value)
    }

    /// An upper bound on `|value|`.
    fn above(value: Word) -> f64 {
        value.value.high.abs() + value.value.low.abs() + value.rounding
    }

    /// A lower bound on `|value|`, when positive.
    fn below(value: Word) -> f64 {
        value.value.high.abs() - value.value.low.abs() - value.rounding
    }

    /// `A/B` in bounded double-double. With `q = fl(a_hi/b_hi)` and the enclosed remainder `R = A − q·B`,
    /// `A/B = q + R/B` exactly, and `R/B − r_hi/b_hi = ((r_lo + ε_R)·b_hi − r_hi·(b_lo + ε_B))/(B·b_hi)`. So replacing it by
    /// the rounded `r_hi/b_hi` errs by at most `(|r_lo| + ρ_R)/|B| + |r_hi|(|b_lo| + ρ_B)/(|B|·|b_hi|) + u|corr| + η/2`,
    /// with `|B|` bounded below by `below(B)`.
    fn quotient(numerator: Word, denominator: Word) -> Word {
        let floor = below(denominator);
        assert!(floor > 0.0, "a quotient by a denominator that may vanish: {denominator:?}");
        let head = numerator.value.high / denominator.value.high;
        let remainder = numerator.sub(denominator.mul_f64(head));
        let correction = remainder.value.high / denominator.value.high;
        let divisor = denominator.value.high.abs();
        Word {
            value: DoubleDouble::two_sum(head, correction),
            rounding: inflated(
                (remainder.value.low.abs() + remainder.rounding) / floor
                    + remainder.value.high.abs() * (denominator.value.low.abs() + denominator.rounding)
                        / (floor * divisor)
                    + UNIT_ROUNDOFF * correction.abs()
                    + ETA,
                12,
            ),
        }
    }

    /// `R(t) = Φ(−t)/φ(t)` for `0 ≤ t ≤ 4`, in bounded double-double with no libm call:
    /// `R(t) = √(π/2)·e^{t²/2} − S(t)`, with `e^{t²/2} = Σ_k (t²/2)^k/k!` and
    /// `S(t) = e^{t²/2}·∫₀ᵗ e^{−s²/2} ds = Σ_n t^{2n+1}/(2n+1)!!`, since `S` solves `S′ = 1 + t·S` with `S(0) = 0`.
    /// Both series have positive terms (see [`positive_series`]). The subtraction cancels by at most
    /// `√(π/2)·e⁸/R(4) ≈ 1.6e4`: its carries scale with the result, and the words absorb the cancellation.
    fn mills_ratio_series(t: f64) -> BoundedDoubleDouble {
        let floor = UNIT_ROUNDOFF * UNIT_ROUNDOFF;
        let square = BoundedDoubleDouble::product(t, t);
        let exponential = positive_series(BoundedDoubleDouble::exact(1.0), square.mul_f64(0.5), 1.0, 1.0, floor);
        let odd = positive_series(BoundedDoubleDouble::exact(t), square, 3.0, 2.0, floor);
        BoundedDoubleDouble::PI
            .mul_f64(0.5)
            .sqrt()
            .mul(exponential)
            .sub(odd)
    }

    /// `Σ_j T_j` with `T_0 = first` and `T_{j+1} = T_j·factor/(offset + j·step)`, for nonnegative terms whose ratios
    /// decrease. When the next ratio's upper bound `ρ` is below one, every later ratio is smaller, so the rest is at most
    /// `T_j·ρ/(1 − ρ)`. The sum stops once that tail falls below `floor` of it, and the tail joins its bound. The test runs
    /// before the next term is formed, so no term below the resolution the bound already carries is ever computed.
    fn positive_series(
        first: BoundedDoubleDouble,
        factor: BoundedDoubleDouble,
        offset: f64,
        step: f64,
        floor: f64,
    ) -> BoundedDoubleDouble {
        let upper = |value: BoundedDoubleDouble| value.value.high.abs() + value.value.low.abs() + value.rounding;
        let factor_upper = upper(factor);
        let mut term = first;
        let mut sum = first;
        let mut divisor = offset;
        loop {
            let ratio = factor_upper / divisor;
            if ratio < 1.0 {
                // `m = 6`: the upper value, the product, the difference, the quotient and the inflation's pair.
                let tail = inflated(upper(term) * ratio / (1.0 - ratio), 6);
                if tail <= floor * sum.value.high {
                    sum.rounding = inflated(sum.rounding + tail, 1);
                    return sum;
                }
            }
            term = term.mul(factor).div_f64(divisor);
            sum = sum.add(term);
            divisor += step;
        }
    }

    /// `M_0(c), …, M_order(c)`, `M_n(c) = ∫₀^∞ sⁿ e^{−cs − s²/2} ds`, in bounded double-double, `order ≥ 1`.
    /// - Below `c = 4`: the owner's series gives `M_0 = R(c)`, then `M_1 = 1 − c·M_0` and `M_{n+1} = n·M_{n−1} − c·M_n`.
    ///   Near `n ≈ c²` that recurrence cancels by about `c²/(n + 1)` per step, at most about `2¹³` in all at `c = 3.75`.
    /// - From `c = 4`: the ratios `ρ_n = M_n/M_{n−1}` satisfy `ρ_n = n/(c + ρ_{n+1})`, dividing the recurrence by `M_n`. That
    ///   is a continued fraction of positive terms whose tail below depth `L` lies in `(0, (L + 1)/c)`. Every level decreases
    ///   in the one below it, so the chains started from both ends enclose every `ρ_n`. Then `M_0 = 1/(c + ρ_1)` and
    ///   `M_n = M_{n−1}·ρ_n`, all positive, so nothing cancels.
    fn moments(center: f64, order: usize) -> Vec<Word> {
        if center < 4.0 {
            let first = mills_ratio_series(center);
            let mut moments = vec![first, exact(1.0).sub(first.mul_f64(center))];
            for index in 1..order {
                let next = moments[index - 1]
                    .mul_f64(index as f64)
                    .sub(moments[index].mul_f64(center));
                moments.push(next);
            }
            return moments;
        }
        const DEPTH: usize = 400;
        let base = exact(center);
        let mut from_zero = exact(0.0);
        let mut from_upper = exact(inflated((DEPTH + 1) as f64 / center, 1));
        let mut ratios = vec![exact(0.0); order + 1];
        for level in (1..=DEPTH).rev() {
            let numerator = exact(level as f64);
            from_zero = quotient(numerator, base.add(from_zero));
            from_upper = quotient(numerator, base.add(from_upper));
            if level <= order {
                ratios[level] = Word {
                    value: from_zero.value,
                    rounding: from_zero.rounding + above(from_zero.sub(from_upper)),
                };
            }
        }
        let mut moments = vec![quotient(exact(1.0), base.add(ratios[1]))];
        for index in 1..=order {
            let next = moments[index - 1].mul(ratios[index]);
            moments.push(next);
        }
        moments
    }

    fn inverse_root_two_pi() -> Word {
        quotient(exact(1.0), Word::PI.mul_f64(2.0).sqrt())
    }

    /// `a_k = Q^{(k)}(c)/k! = (−1)ᵏ M_k(c)/(k!·√(2π))`, `k ≤ order`: `Q = R/√(2π)` and `M_k′ = −M_{k+1}`.
    fn scaled_tail_taylor(center: f64, order: usize) -> Vec<Word> {
        let scale = inverse_root_two_pi();
        let mut factorial = exact(1.0);
        moments(center, order)
            .into_iter()
            .enumerate()
            .map(|(degree, moment)| {
                if degree > 0 {
                    factorial = factorial.mul_f64(degree as f64);
                }
                let coefficient = quotient(moment, factorial).mul(scale);
                if degree % 2 == 1 { coefficient.negated() } else { coefficient }
            })
            .collect()
    }

    /// A bound on `Σ_{k>order} |a_k|·h^k` for `Q` about `c ≥ 0`, `h` the reach, the smaller of two majorants:
    /// - `M_k(c) ≤ M_k(0) = 2^{(k−1)/2}·Γ((k + 1)/2)`, and Wendel's `Γ(y + ½) ≤ √y·Γ(y)` gives
    ///   `M_{k+1}(0) ≤ √(k + 1)·M_k(0)`, so past `k = order + 1` the terms fall by at least `h/√(order + 2)`;
    /// - `M_k(c) ≤ ∫ sᵏ e^{−cs} ds = k!/c^{k+1}` when `c > 0`, so `|a_k|·h^k ≤ (h/c)^k/(c√(2π))`, geometric.
    fn scaled_tail_taylor_remainder(center: f64, order: usize, reach: f64) -> f64 {
        const INVERSE_ROOT_TWO_PI_UPPER: f64 = 0.398_942_280_401_433;
        const HALF_PI_ROOT_UPPER: f64 = 1.253_314_137_315_501;
        let next = order + 1;
        // M_next(0) = (next − 1)!!·(√(π/2) if next is even, else 1).
        let mut moment = if next % 2 == 0 { HALF_PI_ROOT_UPPER } else { 1.0 };
        let mut factor = next as f64 - 1.0;
        while factor > 0.0 {
            moment *= factor;
            factor -= 2.0;
        }
        let mut term = moment * INVERSE_ROOT_TWO_PI_UPPER;
        for degree in 1..=next {
            term = term * reach / degree as f64;
        }
        let decay = reach / ((next + 1) as f64).sqrt();
        let origin = inflated(term / (1.0 - decay), 3 * next + 4);
        if center <= reach {
            return origin;
        }
        let ratio = reach / center;
        let geometric = inflated(
            ratio.powi(next as i32) * INVERSE_ROOT_TWO_PI_UPPER / (center * (1.0 - ratio)),
            next + 6,
        );
        origin.min(geometric)
    }

    /// `b_k = N^{(k)}(c)/k! = (−1)ᵏ M_{k+1}(c)/k!`, `k ≤ order`: `N = M_1` and `M_k′ = −M_{k+1}`.
    fn positive_part_ratio_taylor(center: f64, order: usize) -> Vec<Word> {
        let moments = moments(center, order + 1);
        let mut factorial = exact(1.0);
        (0..=order)
            .map(|degree| {
                if degree > 0 {
                    factorial = factorial.mul_f64(degree as f64);
                }
                let coefficient = quotient(moments[degree + 1], factorial);
                if degree % 2 == 1 { coefficient.negated() } else { coefficient }
            })
            .collect()
    }

    /// A bound on `Σ_{k>order} |b_k|·h^k` for `N` about `c ≥ 0`, the smaller of two majorants:
    /// - `M_{k+1}(c) ≤ M_{k+1}(0)` and Wendel's inequality, so past `k = order + 1` the terms fall by at least
    ///   `h·√(order + 3)/(order + 2)`;
    /// - `M_{k+1}(c) ≤ (k + 1)!/c^{k+2}`, so `|b_k|·h^k ≤ (k + 1)ρᵏ/c` with `ρ = h/c`, and
    ///   `Σ_{k≥n} (k + 1)ρᵏ ≤ (n + 1)ρⁿ/(1 − ρ)²`.
    fn positive_part_ratio_taylor_remainder(center: f64, order: usize, reach: f64) -> f64 {
        const HALF_PI_ROOT_UPPER: f64 = 1.253_314_137_315_501;
        let next = order + 1;
        let index = next + 1;
        // M_index(0) = (index − 1)!!·(√(π/2) if index is even, else 1).
        let mut moment = if index % 2 == 0 { HALF_PI_ROOT_UPPER } else { 1.0 };
        let mut factor = index as f64 - 1.0;
        while factor > 0.0 {
            moment *= factor;
            factor -= 2.0;
        }
        let mut term = moment;
        for degree in 1..=next {
            term = term * reach / degree as f64;
        }
        let decay = reach * ((next + 2) as f64).sqrt() / (next + 1) as f64;
        let origin = inflated(term / (1.0 - decay), 3 * next + 4);
        if center <= reach {
            return origin;
        }
        let ratio = reach / center;
        let geometric = inflated(
            (next + 1) as f64 * ratio.powi(next as i32) / (center * (1.0 - ratio) * (1.0 - ratio)),
            next + 8,
        );
        origin.min(geometric)
    }

    fn taylor(target: Target, center: f64, order: usize) -> Vec<Word> {
        match target {
            Target::ScaledTail => scaled_tail_taylor(center, order),
            Target::PositivePartRatio => positive_part_ratio_taylor(center, order),
        }
    }

    fn taylor_remainder(target: Target, center: f64, order: usize, reach: f64) -> f64 {
        match target {
            Target::ScaledTail => scaled_tail_taylor_remainder(center, order, reach),
            Target::PositivePartRatio => positive_part_ratio_taylor_remainder(center, order, reach),
        }
    }

    /// `sup_{v ∈ [lower, upper]} |Σ_k coefficients[k]·vᵏ|`, rigorously. Each of `cells` equal cells contributes the
    /// polynomial's bounded double-double value at its midpoint, plus its half-width times a bound on `|p′|` over the
    /// interval. `lower`, `upper` and `cells` are dyadic, so every midpoint is exact.
    fn polynomial_sup(coefficients: &[Word], lower: f64, upper: f64, cells: usize) -> f64 {
        let reach = lower.abs().max(upper.abs());
        let slope = coefficients
            .iter()
            .enumerate()
            .skip(1)
            .fold(0.0, |total, (degree, &coefficient)| {
                total + degree as f64 * above(coefficient) * reach.powi(degree as i32 - 1)
            });
        let half_width = (upper - lower) / (2 * cells) as f64;
        let mut largest = 0.0_f64;
        for cell in 0..cells {
            let midpoint = lower + (2 * cell + 1) as f64 * half_width;
            let mut sum = exact(0.0);
            for &coefficient in coefficients.iter().rev() {
                sum = sum.mul_f64(midpoint).add(coefficient);
            }
            largest = largest.max(above(sum));
        }
        inflated(largest + half_width * inflated(slope, 3 * coefficients.len()), 2)
    }

    /// `Σ_k (k + 1)·|coefficients[k]|·reach^k`: a bound on `|p′|` over `|v| ≤ reach` for `p(v) = c + v·Σ_k coefficients[k]·vᵏ`.
    fn slope_bound(coefficients: &[f64], reach: f64) -> f64 {
        let total = coefficients
            .iter()
            .enumerate()
            .fold(0.0, |total, (degree, &coefficient)| {
                total + (degree + 1) as f64 * coefficient.abs() * reach.powi(degree as i32)
            });
        inflated(total, 3 * coefficients.len())
    }

    /// Production's [`estrin`] at `|argument| ≤ reach`, walked on magnitudes: returns `sup |value|` and a bound on the
    /// rounding of the computed value. Each node `a + b·p` has an exact value below `A + B·P`, and its computed value
    /// errs by the inputs' propagated errors plus `u` of the product and `u` of the sum. The power `p` is squared
    /// between levels with its own error. The walk mirrors production's in-place indexing step for step.
    fn estrin_rounding<const N: usize>(coefficients: &[f64; N], reach: f64) -> (f64, f64) {
        let u = UNIT_ROUNDOFF;
        let mut magnitudes = (*coefficients).map(f64::abs);
        let mut errors = [0.0_f64; N];
        let mut count = N;
        let mut power = reach;
        let mut power_error = 0.0_f64;
        while count > 1 {
            let half = count.div_ceil(2);
            let mut index = 0;
            while index < half {
                if 2 * index + 1 < count {
                    let (low, low_error) = (magnitudes[2 * index], errors[2 * index]);
                    let (high, high_error) = (magnitudes[2 * index + 1], errors[2 * index + 1]);
                    let computed_power = power + power_error;
                    let product = (high + high_error) * computed_power;
                    let product_error = high_error * computed_power + high * power_error + u * product;
                    magnitudes[index] = low + high * power;
                    errors[index] = low_error + product_error + u * (low + low_error + product + product_error);
                } else {
                    magnitudes[index] = magnitudes[2 * index];
                    errors[index] = errors[2 * index];
                }
                index += 1;
            }
            count = half;
            let computed_power = power + power_error;
            power_error = 2.0 * power * power_error + power_error * power_error + u * computed_power * computed_power;
            power *= power;
        }
        (magnitudes[0], inflated(errors[0], 8 * N) + N as f64 * ETA)
    }

    /// Production's Horner loops at `0 ≤ argument ≤ reach`, walked on magnitudes: `sum = c[last]`, then
    /// `sum = sum·v + c[k]` down to `k = 0` of the slice. Returns `sup |value|` and a bound on the computed value's rounding.
    fn horner_rounding(coefficients: &[f64], reach: f64) -> (f64, f64) {
        let u = UNIT_ROUNDOFF;
        let last = coefficients.len() - 1;
        let mut magnitude = coefficients[last].abs();
        let mut error = 0.0_f64;
        for &coefficient in coefficients[..last].iter().rev() {
            let product = (magnitude + error) * reach;
            let product_error = error * reach + u * product;
            magnitude = magnitude * reach + coefficient.abs();
            error = product_error + u * (coefficient.abs() + product + product_error);
        }
        (magnitude, inflated(error, 4 * coefficients.len()) + coefficients.len() as f64 * ETA)
    }

    /// One piece's proof: absolute bounds on the approximation error `|P(v) − Q(c + v)|` and on the evaluation's
    /// rounding, and a lower bound on `Q` over the piece.
    struct PieceProof {
        approximation: f64,
        rounding: f64,
        minimum: f64,
    }

    impl PieceProof {
        /// The proven bound on `|Q̂ − Q|/Q` over the piece.
        fn relative(&self) -> f64 {
            inflated((self.approximation + self.rounding) / self.minimum, 2)
        }
    }

    /// The proof for piece `index` of the stored polynomial `constant + low + v·Σ_k slopes[k]·vᵏ`.
    /// - **Approximation.** The certified Taylor model of `Q` about the midpoint `c`, subtracted coefficient by coefficient
    ///   from the stored polynomial, leaves a polynomial whose supremum over the piece [`polynomial_sup`] bounds. The
    ///   model's own remainder is then added.
    /// - **Rounding.** `E` from [`estrin_rounding`], then the product `v·E`, the sum with `c₀ˡᵒ`, and the sum with `c₀ʰⁱ`,
    ///   each `u` of its result. Piece 0's offset `t − ¼` is inexact below `t = ⅛`. It errs there by at most `u·¼`,
    ///   which the polynomial's slope carries.
    /// - **Minimum.** `Q` decreases (`Q′ = tQ − 1/√(2π) < 0`, since `t·R(t) < 1`), so its minimum on the piece is at
    ///   `c + h`, bounded below from the model.
    fn piece_proof(
        target: Target,
        index: usize,
        constant: f64,
        slopes: &[f64; PIECE_DEGREE],
        low: f64,
    ) -> PieceProof {
        let u = UNIT_ROUNDOFF;
        let center = (index as f64 + 0.5) * PIECE_WIDTH;
        let reach = 0.5 * PIECE_WIDTH;
        let taylor = taylor(target, center, MODEL_ORDER);
        let remainder = taylor_remainder(target, center, MODEL_ORDER, reach);
        let mut difference: Vec<Word> = taylor.iter().map(|coefficient| coefficient.negated()).collect();
        difference[0] = difference[0].add(exact(constant)).add(exact(low));
        for (degree, &slope) in slopes.iter().enumerate() {
            difference[degree + 1] = difference[degree + 1].add(exact(slope));
        }
        let approximation = inflated(polynomial_sup(&difference, -reach, reach, PIECE_CELLS) + remainder, 1);
        let mut at_end = exact(0.0);
        for &coefficient in taylor.iter().rev() {
            at_end = at_end.mul_f64(reach).add(coefficient);
        }
        let minimum = below(at_end) - remainder;
        let (slope_magnitude, slope_error) = estrin_rounding(slopes, reach);
        let product = reach * (slope_magnitude + slope_error);
        let product_error = reach * slope_error + u * product;
        let inner = low.abs() + product;
        let inner_error = product_error + u * (inner + product_error);
        let total = constant.abs() + inner + inner_error;
        let offset_error = if index == 0 { u * reach * slope_bound(slopes, reach) } else { 0.0 };
        PieceProof {
            approximation,
            rounding: inflated(inner_error + u * total + offset_error + 2.0 * ETA, 8),
            minimum,
        }
    }

    /// The proof of the reciprocal tail `t ≥ 8`: the largest proven `|Q̂ − Q|/Q` there.
    /// - **Approximation on `[8, 64)`**, over 56 unit sub-intervals about `c = 8.5, …, 63.5`, `h = ½`. The stored function
    ///   is `F(t) = Σ_j κ_j t^{−(2j+1)}`, with `κ_0 = TAIL[0] + TAIL_LOW`. Its Taylor coefficients about `c` are
    ///   `b_k = (−1)ᵏ Σ_j κ_j C(2j + k, k)·c^{−(2j+1)−k}`, from `(c + v)^{−m} = c^{−m} Σ_k C(m + k − 1, k)(−v/c)ᵏ`. Their
    ///   remainder past the model order is geometric with ratio at most `(2·12 + D + 2)ρ/(D + 2)`, `ρ = h/c`. Subtracting
    ///   `Q`'s certified model leaves a polynomial that [`polynomial_sup`] bounds.
    /// - **Approximation on `[64, ∞)`**, in `s = 1/t² ∈ (0, 1/4096]`. `K(s) = t·Q(t)` is enveloped by its asymptotic
    ///   series `Σ_{m<M} (−1)ᵐ(2m − 1)!!·sᵐ/√(2π)`, whose remainder is at most the first omitted term. That follows from
    ///   integrating `e^{−u} = Σ_{m<M}(−u)ᵐ/m! + θ(−u)ᴹ/M!`, `θ ∈ [0, 1]`, against `e^{−ts}`. `K ≥ 1/((1 + s)√(2π))`
    ///   follows from Mills' `R(t) > t/(t² + 1)`.
    /// - **Rounding.**
    ///   - `r = fl(1/t)` and `s = fl(r²)` put `s` within `3u` of `1/t²`, which `K`'s slope carries.
    ///   - The Horner sum comes from [`horner_rounding`], then the product with `s`, the sums with `TAIL_LOW` and
    ///     `TAIL[0]`, and the product with `r`, which also carries `r`'s own `u`.
    fn tail_proof(target: Target) -> f64 {
        let u = UNIT_ROUNDOFF;
        let scale = inverse_root_two_pi();
        let stored_tail = target.tail();
        let degree_of_tail = stored_tail.len() - 1;
        let power = target.power();
        let mut kappa: Vec<Word> = stored_tail.iter().map(|&coefficient| exact(coefficient)).collect();
        kappa[0] = kappa[0].add(exact(target.tail_low()));
        let mut worst = 0.0_f64;
        for interval in 0..TAIL_INTERVALS {
            let center = TABLE_REACH + interval as f64 + 0.5;
            let reach = 0.5;
            let model = taylor(target, center, MODEL_ORDER);
            let model_remainder = taylor_remainder(target, center, MODEL_ORDER, reach);
            let inverse = quotient(exact(1.0), exact(center));
            // inverse_powers[m] = c^{−m}.
            let highest = 2 * degree_of_tail + power + MODEL_ORDER;
            let mut inverse_powers = vec![exact(1.0)];
            for exponent in 1..=highest {
                let next = inverse_powers[exponent - 1].mul(inverse);
                inverse_powers.push(next);
            }
            // (c + v)^{−m} = c^{−m} Σ_k C(m + k − 1, k)(−v/c)ᵏ with m = 2j + power.
            let mut stored = vec![exact(0.0); MODEL_ORDER + 1];
            for (index, &coefficient) in kappa.iter().enumerate() {
                let exponent = 2 * index + power;
                let mut binomial = exact(1.0);
                for (degree, slot) in stored.iter_mut().enumerate() {
                    if degree > 0 {
                        binomial = binomial.mul_f64((exponent + degree - 1) as f64).div_f64(degree as f64);
                    }
                    let term = coefficient.mul(binomial).mul(inverse_powers[exponent + degree]);
                    *slot = if degree % 2 == 1 { slot.sub(term) } else { slot.add(term) };
                }
            }
            let ratio = reach / center;
            let order = MODEL_ORDER as f64;
            let largest_exponent = (2 * degree_of_tail + power) as f64;
            let decay = (largest_exponent + order + 1.0) / (order + 2.0) * ratio;
            let mut stored_remainder = 0.0;
            for (index, &coefficient) in stored_tail.iter().enumerate() {
                let exponent = 2 * index + power;
                let mut binomial = 1.0_f64;
                for degree in 1..=MODEL_ORDER + 1 {
                    binomial = binomial * (exponent + degree - 1) as f64 / degree as f64;
                }
                stored_remainder += (coefficient.abs() + target.tail_low().abs())
                    * binomial
                    * ratio.powi(MODEL_ORDER as i32 + 1)
                    / center.powi(exponent as i32);
            }
            let stored_remainder = inflated(stored_remainder / (1.0 - decay), 4 * MODEL_ORDER);
            let difference: Vec<Word> = stored.iter().zip(&model).map(|(&b, &a)| b.sub(a)).collect();
            let approximation = polynomial_sup(&difference, -reach, reach, TAIL_CELLS)
                + stored_remainder
                + model_remainder;
            let mut at_end = exact(0.0);
            for &coefficient in model.iter().rev() {
                at_end = at_end.mul_f64(reach).add(coefficient);
            }
            let minimum = below(at_end) - model_remainder;
            worst = worst.max(inflated(approximation, 2) / minimum);
        }
        // [64, ∞) in s = 1/t²: the enveloping asymptotic series of the kernel, K = t·Q or K_N = t²·N.
        // - K(s) = Σ_m (−1)ᵐ(2m − 1)!!·sᵐ/√(2π), with K ≥ 1/((1 + s)√(2π)) from Mills' R(t) > t/(t² + 1).
        // - K_N(s) = Σ_n (−1)ⁿ(2n + 1)!!·sⁿ, with K_N ≥ 1/(1 + 3s) from the continued fraction's third convergent
        //   R(t) < (t² + 2)/(t³ + 3t).
        // Either remainder is at most its first omitted term.
        let top = 1.0 / (ASYMPTOTIC_REACH * ASYMPTOTIC_REACH);
        let mut series = Vec::with_capacity(ASYMPTOTIC_TERMS);
        let mut double_factorial = exact(1.0);
        for term in 0..ASYMPTOTIC_TERMS {
            let factor = match target {
                Target::ScaledTail => 2 * term as i64 - 1,
                Target::PositivePartRatio => 2 * term as i64 + 1,
            };
            if factor > 1 {
                double_factorial = double_factorial.mul_f64(factor as f64);
            }
            let coefficient = match target {
                Target::ScaledTail => double_factorial.mul(scale),
                Target::PositivePartRatio => double_factorial,
            };
            series.push(if term % 2 == 1 { coefficient.negated() } else { coefficient });
        }
        let next_factor = match target {
            Target::ScaledTail => 2 * ASYMPTOTIC_TERMS - 1,
            Target::PositivePartRatio => 2 * ASYMPTOTIC_TERMS + 1,
        };
        let omitted_coefficient = match target {
            Target::ScaledTail => double_factorial.mul_f64(next_factor as f64).mul(scale),
            Target::PositivePartRatio => double_factorial.mul_f64(next_factor as f64),
        };
        let mut difference: Vec<Word> = series.iter().map(|coefficient| coefficient.negated()).collect();
        for (index, &coefficient) in kappa.iter().enumerate() {
            difference[index] = difference[index].add(coefficient);
        }
        let first_omitted =
            inflated(above(omitted_coefficient) * top.powi(ASYMPTOTIC_TERMS as i32), ASYMPTOTIC_TERMS + 2);
        let kernel_floor = |s: f64| match target {
            Target::ScaledTail => below(scale) / (1.0 + s),
            Target::PositivePartRatio => 1.0 / inflated(1.0 + 3.0 * s, 1),
        };
        let asymptotic = inflated(polynomial_sup(&difference, 0.0, top, TAIL_CELLS) + first_omitted, 1);
        worst = worst.max(inflated(asymptotic / kernel_floor(top), 2));
        // Rounding, over all of s ∈ [0, 1/64]. r = fl(1/t) and s = fl(r²) put s within 3u of 1/t², which the kernel's
        // slope carries. Q = r·K then adds r's u and the product's; N = s·K_N adds s's own 3u and the product's.
        let square_top = inflated(1.0 / (TABLE_REACH * TABLE_REACH), 3);
        let (sum_magnitude, sum_error) = horner_rounding(&stored_tail[1..], square_top);
        let product = square_top * (sum_magnitude + sum_error);
        let product_error = square_top * sum_error + u * product;
        let inner = target.tail_low().abs() + product;
        let inner_error = product_error + u * (inner + product_error);
        let total = stored_tail[0].abs() + inner + inner_error;
        let argument_error = slope_bound(&stored_tail[1..], square_top) * 3.01 * u * square_top;
        let kernel_relative = (inner_error + u * total + argument_error) / kernel_floor(square_top);
        let outer = match target {
            Target::ScaledTail => 2.01 * u,
            Target::PositivePartRatio => 4.02 * u,
        };
        let rounding = inflated(kernel_relative + outer, 8);
        inflated(worst + rounding + worst * rounding, 2)
    }

    /// The proven relative error of `target`'s table over the pieces `from..PIECE_COUNT` and the reciprocal tail.
    fn table_proof(target: Target, from: usize) -> f64 {
        let pieces = (from..PIECE_COUNT)
            .map(|index| {
                piece_proof(
                    target,
                    index,
                    target.constants()[index],
                    &target.slopes()[index],
                    target.lows()[index],
                )
                .relative()
            })
            .fold(0.0, f64::max);
        pieces.max(tail_proof(target))
    }

    /// Bounds on `Φ(−0.75) = φ(0.75)·R(0.75)`: `φ` from `normal_pdf_bounded`, `R = M_0(0.75)` from the certified moments
    /// (the owner itself reads this table).
    fn central_edge() -> (f64, f64) {
        let (density, density_rounding) = normal_pdf_bounded(CENTRAL_REACH);
        let ratio = moments(CENTRAL_REACH, 1)[0];
        (
            (density - density_rounding) * below(ratio),
            inflated((density + density_rounding) * above(ratio), 2),
        )
    }

    /// The proven `|Φ̂ − Φ|/Φ` on `|x| < 0.75`.
    /// - **Approximation.** `P(y) = (Φ(√y) − ½)/√y = Σ_n (−1)ⁿ yⁿ/(2ⁿ n! (2n + 1)√(2π))`. The terms alternate and decrease on
    ///   `y ≤ 9/16`, so the series past the model order is within its first omitted term. The stored `P` minus that series
    ///   is bounded by [`polynomial_sup`] over `[0, 9/16]`, and `Φ` sees it times `|x| ≤ ¾`.
    /// - **Rounding.**
    ///   - `y = fl(x²)` within `u·y`, carried by `P`'s slope.
    ///   - The Horner sum from [`horner_rounding`], then the product with `y`, the sums with `CENTRAL_LOW` and `CENTRAL[0]`,
    ///     the product with `x`, and the sum with `½`, each `u` of its result.
    /// - **Relative.** Everything divided by `Φ(−0.75)`, the smallest `Φ` on the interval, bounded below by the owner.
    fn central_proof() -> f64 {
        let u = UNIT_ROUNDOFF;
        let top = CENTRAL_REACH * CENTRAL_REACH;
        let scale = inverse_root_two_pi();
        let mut series = Vec::with_capacity(MODEL_ORDER + 1);
        let mut term = exact(1.0);
        for degree in 0..=MODEL_ORDER {
            if degree > 0 {
                term = term.negated().div_f64(2.0 * degree as f64);
            }
            series.push(term.div_f64((2 * degree + 1) as f64).mul(scale));
        }
        let first_omitted = {
            let next = MODEL_ORDER + 1;
            above(term.div_f64(2.0 * next as f64).div_f64((2 * next + 1) as f64).mul(scale))
                * top.powi(next as i32)
        };
        let mut difference: Vec<Word> = series.iter().map(|coefficient| coefficient.negated()).collect();
        difference[0] = difference[0].add(exact(CENTRAL[0])).add(exact(CENTRAL_LOW));
        for (degree, &coefficient) in CENTRAL.iter().enumerate().skip(1) {
            difference[degree] = difference[degree].add(exact(coefficient));
        }
        let approximation =
            inflated(polynomial_sup(&difference, 0.0, top, PIECE_CELLS) + inflated(first_omitted, 2), 1);
        let (sum_magnitude, sum_error) = horner_rounding(&CENTRAL[1..], top);
        let product = top * (sum_magnitude + sum_error);
        let product_error = top * sum_error + u * product;
        let inner = CENTRAL_LOW.abs() + product;
        let inner_error = product_error + u * (inner + product_error);
        let polynomial = CENTRAL[0].abs() + inner + inner_error;
        let polynomial_error = inner_error + u * polynomial + slope_bound(&CENTRAL[1..], top) * u * top;
        let scaled = CENTRAL_REACH * (polynomial + polynomial_error);
        let absolute = CENTRAL_REACH * (approximation + polynomial_error)
            + u * scaled
            + u * (0.5 + scaled + u * scaled);
        let (lowest, _) = central_edge();
        inflated(absolute / (lowest - absolute), 16)
    }

    /// The proven `|Φ̂ − Φ|/Φ` wherever `Φ̂` is a normal double, given the proven scaled-tail bound over `t ≥ 0.75`.
    /// - **Left tail**, `Φ(−t) = exp(−½t²)·Q(t)`:
    ///   - `exp(−½t²) = exp(−½·fl(t²))·exp(−½·res)` with `res = t² − fl(t²)` exact, and `|res| ≤ u·t²`;
    ///   - `libm::exp` errs by under one ulp (`2u` relative, the cited contract);
    ///   - `1 − ½res` stands for `exp(−½res)` within `(u·t²)²/8 ≤ 3e-27` at `t ≤ 38.6`;
    ///   - the product `E·Q̂` rounds by `u`, the product `½res·L` by `u` of a term below `u·t²/2` of `L`, and the
    ///     difference by `u`.
    /// - **Right tail:** `Φ(t) = 1 − Φ(−t)`, so its error is the left error times `L/(1 − L)` with `L ≤ Φ(−0.75)`, plus the
    ///   complement's `u`.
    fn normal_cdf_proof(scaled_tail: f64) -> f64 {
        let u = UNIT_ROUNDOFF;
        let largest_square = 38.6_f64 * 38.6;
        let exponential = 2.0 * u / (1.0 - 2.0 * u);
        let correction = (u * largest_square) * (u * largest_square) / 8.0 + u * u * largest_square / 2.0;
        let left = inflated(
            (1.0 + scaled_tail) * (1.0 + exponential) * (1.0 + u) * (1.0 + correction) * (1.0 + u) - 1.0,
            8,
        );
        let (_, edge) = central_edge();
        let right = inflated((left * edge / (1.0 - edge) + u) / (1.0 - u), 6);
        central_proof().max(left).max(right)
    }

    /// A proven relative bound converted to one relative to the computed value: `ε/(1 − ε)`.
    fn relative_to_computed(bound: f64) -> f64 {
        inflated(bound / (1.0 - bound), 2)
    }

    #[test]
    fn scaled_tail_constant_covers_its_proof_and_is_within_twice_of_it() {
        let proven = relative_to_computed(table_proof(Target::ScaledTail, 0));
        assert!(
            proven <= NORMAL_SCALED_TAIL_RELATIVE_ERROR && NORMAL_SCALED_TAIL_RELATIVE_ERROR <= 2.0 * proven,
            "the proof gives {proven:e} ({:.3}u); the published constant is {NORMAL_SCALED_TAIL_RELATIVE_ERROR:e}",
            proven / UNIT_ROUNDOFF
        );
        // Each piece, and the tail, is certified on its own: none is left to the others' margins.
        for index in 0..PIECE_COUNT {
            let proof = piece_proof(
                Target::ScaledTail,
                index,
                PIECE_CONSTANTS[index],
                &PIECE_SLOPES[index],
                PIECE_LOWS[index],
            );
            assert!(
                proof.minimum > 0.0 && proof.relative() <= proven,
                "piece {index}: approximation {:e}, rounding {:e}, minimum {:e}",
                proof.approximation,
                proof.rounding,
                proof.minimum
            );
        }
    }

    #[test]
    fn positive_part_ratio_constant_covers_its_proof_and_is_within_twice_of_it() {
        let proven = relative_to_computed(table_proof(Target::PositivePartRatio, 0));
        assert!(
            proven <= POSITIVE_PART_RATIO_RELATIVE_ERROR && POSITIVE_PART_RATIO_RELATIVE_ERROR <= 2.0 * proven,
            "the proof gives {proven:e} ({:.3}u); the published constant is {POSITIVE_PART_RATIO_RELATIVE_ERROR:e}",
            proven / UNIT_ROUNDOFF
        );
    }

    #[test]
    fn normal_cdf_constant_covers_its_proof_and_is_within_twice_of_it() {
        let scaled_tail = table_proof(Target::ScaledTail, 1);
        let proven = relative_to_computed(normal_cdf_proof(scaled_tail));
        assert!(
            proven <= NORMAL_CDF_RELATIVE_ERROR && NORMAL_CDF_RELATIVE_ERROR <= 2.0 * proven,
            "the proof gives {proven:e} ({:.3}u), from the scaled tail's {scaled_tail:e} and the central {:e}; the published constant is {NORMAL_CDF_RELATIVE_ERROR:e}",
            proven / UNIT_ROUNDOFF,
            central_proof()
        );
    }

    #[test]
    fn a_perturbed_piece_fails_the_published_constant() {
        // Positive control for the proof: one slope of piece 7 moved by 1e-13 shifts the polynomial by up to
        // 1e-13·(¼)⁴ ≈ 3.9e-16 against Q ≈ 0.09 there, so a sound proof must reject it.
        let mut slopes = PIECE_SLOPES[7];
        slopes[3] += 1.0e-13;
        let perturbed = piece_proof(Target::ScaledTail, 7, PIECE_CONSTANTS[7], &slopes, PIECE_LOWS[7]);
        let stored = piece_proof(Target::ScaledTail, 7, PIECE_CONSTANTS[7], &PIECE_SLOPES[7], PIECE_LOWS[7]);
        assert!(
            relative_to_computed(perturbed.relative()) > NORMAL_SCALED_TAIL_RELATIVE_ERROR
                && relative_to_computed(stored.relative()) <= NORMAL_SCALED_TAIL_RELATIVE_ERROR,
            "perturbed {:e}, stored {:e}, constant {NORMAL_SCALED_TAIL_RELATIVE_ERROR:e}",
            perturbed.relative(),
            stored.relative()
        );
    }

    #[test]
    fn estrin_rounding_bound_covers_the_computed_polynomials() {
        // The a priori bound walks production's tree; here it must cover production's actual rounding against the
        // same polynomial in bounded double-double, at points across every piece.
        let reach = 0.5 * PIECE_WIDTH;
        let mut largest_ratio = 0.0_f64;
        for (index, slopes) in PIECE_SLOPES.iter().chain(RATIO_PIECE_SLOPES.iter()).enumerate() {
            let (magnitude, bound) = estrin_rounding(slopes, reach);
            for step in 0..=64 {
                let offset = -reach + reach * f64::from(step) / 32.0;
                let computed = estrin(slopes, offset);
                let mut reference = exact(0.0);
                for &coefficient in slopes.iter().rev() {
                    reference = reference.mul_f64(offset).add(exact(coefficient));
                }
                let error = ((computed - reference.value.high) - reference.value.low).abs();
                assert!(
                    error <= bound + reference.rounding && computed.abs() <= magnitude + bound,
                    "piece {index} at v = {offset}: error {error:e} beyond the a priori {bound:e}"
                );
                largest_ratio = largest_ratio.max(error / bound);
            }
        }
        assert!(largest_ratio > 0.0, "no evaluation rounded, so the bound was not exercised");
    }

    #[test]
    fn density_is_bitwise_normal_pdf_and_symmetric() {
        let mut checked = 0_usize;
        for step in 0..=20_000_u32 {
            let x = -40.0 + 80.0 * f64::from(step) / 20_000.0;
            let (cdf, density) = normal_cdf_and_pdf(x);
            let (mirrored_cdf, mirrored_density) = normal_cdf_and_pdf(-x);
            assert_eq!(density.to_bits(), normal_pdf(x).to_bits(), "φ({x})");
            assert_eq!(density.to_bits(), mirrored_density.to_bits(), "φ(±{x})");
            assert!((0.0..=1.0).contains(&cdf) && (0.0..=1.0).contains(&mirrored_cdf), "Φ(±{x}) out of [0, 1]");
            checked += 1;
        }
        assert_eq!(checked, 20_001);
    }

    #[test]
    fn special_values_follow_the_limits() {
        assert_eq!(normal_cdf_and_pdf(f64::NEG_INFINITY), (0.0, 0.0));
        assert_eq!(normal_cdf_and_pdf(f64::INFINITY), (1.0, 0.0));
        let (cdf, density) = normal_cdf_and_pdf(f64::NAN);
        assert!(cdf.is_nan() && density.is_nan());
        assert_eq!(normal_cdf_and_pdf(0.0).0, 0.5);
        assert_eq!(normal_scaled_tail(f64::INFINITY), 0.0);
        assert!(normal_scaled_tail(f64::NAN).is_nan() && normal_scaled_tail(-1.0).is_nan());
        // Positive control for the limits: a finite argument is neither.
        assert!(normal_scaled_tail(1.0) > 0.0 && normal_cdf_and_pdf(1.0).0 < 1.0);
    }

    #[test]
    fn tables_agree_with_the_certified_moments() {
        // The oracle is the proof's own certified moments, R = M_0 and N = M_1: the positive series below t = 4 and the
        // enclosed continued fraction beyond, independent of the tables. Every table value must lie within its
        // published constant of that enclosure, on 7,601 points of t ∈ [0, 38]; Φ is checked on both signs of x.
        let scale = inverse_root_two_pi();
        let mut largest = [0.0_f64; 3];
        for step in 0..=7_600_u32 {
            let magnitude = 38.0 * f64::from(step) / 7_600.0;
            let certified = moments(magnitude, 1);
            let tail_reference = certified[0].mul(scale);
            let ratio_reference = certified[1];
            for (index, (value, reference, constant)) in [
                (scaled_tail(magnitude), tail_reference, NORMAL_SCALED_TAIL_RELATIVE_ERROR),
                (positive_part_ratio(magnitude), ratio_reference, POSITIVE_PART_RATIO_RELATIVE_ERROR),
            ]
            .into_iter()
            .enumerate()
            {
                let error = ((reference.value.high - value) + reference.value.low).abs();
                assert!(
                    error <= constant * value + reference.rounding,
                    "table {index} at t = {magnitude}: {value} against the certified {} ± {:e}",
                    reference.value.high,
                    reference.rounding
                );
                largest[index] = largest[index].max((error - reference.rounding).max(0.0) / value);
            }
            let (density, density_rounding) = normal_pdf_bounded(magnitude);
            let lower = density * certified[0].value.high;
            let lower_rounding = inflated(
                density_rounding * above(certified[0])
                    + density * (certified[0].value.low.abs() + certified[0].rounding)
                    + UNIT_ROUNDOFF * lower,
                4,
            ) + ETA;
            for x in [-magnitude, magnitude] {
                let (reference, reference_rounding) =
                    if x <= 0.0 { (lower, lower_rounding) } else { (1.0 - lower, lower_rounding + UNIT_ROUNDOFF) };
                let (cdf, table_density) = normal_cdf_and_pdf(x);
                assert_eq!(table_density.to_bits(), normal_pdf(x).to_bits(), "φ({x})");
                let error = (cdf - reference).abs();
                assert!(
                    error <= NORMAL_CDF_RELATIVE_ERROR * cdf + NORMAL_CDF_UNDERFLOW_FLOOR + reference_rounding,
                    "Φ({x}) = {cdf} against the certified {reference} ± {reference_rounding:e}"
                );
                if cdf > f64::MIN_POSITIVE {
                    largest[2] = largest[2].max((error - reference_rounding).max(0.0) / cdf);
                }
            }
        }
        assert!(
            largest[0] <= NORMAL_SCALED_TAIL_RELATIVE_ERROR
                && largest[1] <= POSITIVE_PART_RATIO_RELATIVE_ERROR
                && largest[2] <= NORMAL_CDF_RELATIVE_ERROR,
            "largest resolved relative gaps {largest:?}"
        );
    }

    #[test]
    fn normal_left_tail_ratios_read_the_tables_within_their_constants() {
        // The owner's entry reads Q and N from the tables. Its bounds must cover the certified moments, and stay within
        // a few rounding units of the values, so the owner certifies what the tables prove.
        for step in 0..=4_000_u32 {
            let magnitude = 40.0 * f64::from(step) / 4_000.0;
            let ratios = normal_left_tail_ratios(-magnitude, 0.0).expect("the left tail");
            let certified = moments(magnitude, 1);
            for (name, value, rounding, reference) in [
                ("R", ratios.cdf_over_density, ratios.cdf_over_density_rounding, certified[0]),
                ("N", ratios.positive_part_over_density, ratios.positive_part_over_density_rounding, certified[1]),
            ] {
                let error = ((reference.value.high - value) + reference.value.low).abs();
                assert!(
                    error <= rounding + reference.rounding && rounding <= 16.0 * UNIT_ROUNDOFF * value + 2.0 * ETA,
                    "{name}(t = {magnitude}) = {value} ± {rounding:e} against the certified {} ± {:e}",
                    reference.value.high,
                    reference.rounding
                );
            }
        }
    }
}
