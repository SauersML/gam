"""Generate the high-precision reference tables for the tail-probability tests.

Run from the repository root with ``python crates/gam-math/tests/tail_references/generate.py``.
It needs only ``mpmath``, is deterministic, and rewrites the ``*.tsv`` files next
to it. The Rust test ``crates/gam-math/tests/tail_references.rs`` reads them.

Every reference is evaluated at the DOUBLE the Rust side receives: a target
probability picks a statistic, the statistic is rounded to ``f64``, and the tail
is then recomputed at that rounded statistic with 50 significant digits. The
stored value is therefore the exact tail at the tested input, not at a nearby
one.

The weighted chi-square references come from the Bromwich inversion of the
moment generating function,

    P(Q > x) = (1/2 pi i) int_{c - i inf}^{c + i inf} M(s) e^{-s x} ds / s,   0 < c < s_+,

on two different hyperbolic contours through the saddle point, integrated with
mpmath's tanh-sinh rule. The integral does not depend on the contour, so the
two evaluations agreeing to 30 digits is the check that neither has a
quadrature problem; the formula itself is checked against the closed forms it
must reproduce (equal weights give a scaled chi-square, a two-term signed
combination at zero gives an F tail). The script asserts all three.
"""

from __future__ import annotations

import multiprocessing
import pathlib

import mpmath as mp

mp.mp.dps = 50
HERE = pathlib.Path(__file__).resolve().parent
AGREEMENT = mp.mpf(10) ** -30


def fmt(value: mp.mpf) -> str:
    return mp.nstr(value, 25, min_fixed=1, max_fixed=0)


def f64(value: mp.mpf) -> float:
    return float(value)


def solve_increasing_log(fn, target_log, lo, hi):
    """Root of fn(t) = target_log for fn increasing in t on [lo, hi], by bisection."""
    # The target only chooses where to test: the statistic is rounded to f64
    # and the tail recomputed there, so the root is only needed roughly.
    lo, hi = mp.mpf(lo), mp.mpf(hi)
    while hi - lo > mp.mpf(10) ** -6:
        mid = (lo + hi) / 2
        if fn(mid) < target_log:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def statistic_for_tail(sf, p, lo_log, hi_log):
    """The statistic whose survival probability is p, searched on log(statistic)."""
    target = mp.log(p)
    log_x = solve_increasing_log(lambda t: -mp.log(sf(mp.exp(t))), -target, lo_log, hi_log)
    return mp.exp(log_x)


TARGETS = [
    1 - mp.mpf(10) ** -12,
    mp.mpf("0.9"),
    mp.mpf("0.5"),
    mp.mpf("0.1"),
    mp.mpf(10) ** -3,
    mp.mpf(10) ** -6,
    mp.mpf(10) ** -10,
    mp.mpf(10) ** -16,
    mp.mpf(10) ** -32,
    mp.mpf(10) ** -50,
    mp.mpf(10) ** -100,
    mp.mpf(10) ** -200,
    mp.mpf(10) ** -300,
]
DFS = ["0.001", "0.01", "0.1", "0.5", "1", "2", "3", "5", "10", "30", "100", "1000", "10000"]


def representable(x: mp.mpf) -> bool:
    return mp.mpf(2) ** -1022 < x < mp.mpf(2) ** 1023


def chi_square_sf(x, df):
    return mp.gammainc(mp.mpf(df) / 2, x / 2, mp.inf, regularized=True)


def beta_tail(a, b, ratio):
    """I_x(a, b) at x = 1/(1 + ratio), for the beta arguments of the t and F tails.

    Where ratio is below the working precision, x is within one working ulp of 1
    and 1 - x = ratio/(1 + ratio) would keep none of its digits: F(0.001, 1) at
    f = 1.3e-48 has 1 - x = 1.3e-51, and at 50 digits its tail came out 0.0572
    against 0.0575. The digits 1 - x sits below 1 are added to the precision, so
    x carries 1 - x to the full working precision.
    """
    extra = max(0, int(mp.ceil(-mp.log10(ratio)))) if ratio < 1 else 0
    with mp.extradps(extra):
        return +mp.betainc(a, b, 0, 1 / (1 + ratio), regularized=True)


def t_two_sided(t, df):
    df = mp.mpf(df)
    return beta_tail(df / 2, mp.mpf("0.5"), t * t / df)


def f_sf(f, d1, d2):
    d1, d2 = mp.mpf(d1), mp.mpf(d2)
    return beta_tail(d2 / 2, d1 / 2, d1 * f / d2)


def distinct(rows):
    """Drops a row that repeats an earlier one: where two targets lie closer than one
    representable step, or the search clamps at the edge of the range, both land on the
    same statistic."""
    seen = set()
    kept = []
    for row in rows:
        if row not in seen:
            seen.add(row)
            kept.append(row)
    return kept


def chi_square_rows():
    rows = []
    for df in DFS:
        for p in TARGETS:
            sf = lambda x: chi_square_sf(x, df)
            x = statistic_for_tail(sf, p, -2000, 12)
            if not representable(x):
                continue
            x = mp.mpf(f64(x))
            rows.append((df, repr(f64(x)), fmt(sf(x))))
    return distinct(rows)


def t_rows():
    rows = []
    for df in DFS:
        for p in TARGETS:
            sf = lambda t: t_two_sided(t, df)
            t = statistic_for_tail(sf, p, -1000, 1400)
            if not representable(t):
                continue
            t = mp.mpf(f64(t))
            rows.append((df, repr(f64(t)), fmt(sf(t))))
    return distinct(rows)


F_PAIRS = [
    ("0.001", "1"),
    ("1", "0.001"),
    ("0.5", "3"),
    ("1", "1"),
    ("1", "5"),
    ("2", "17"),
    ("3", "26"),
    ("0.7", "24"),
    ("5.4", "191"),
    ("11", "4"),
    ("30", "10000"),
    ("10000", "30"),
    ("1000", "1000"),
]


def f_rows():
    rows = []
    for d1, d2 in F_PAIRS:
        for p in TARGETS:
            sf = lambda f: f_sf(f, d1, d2)
            f = statistic_for_tail(sf, p, -3000, 2000)
            if not representable(f):
                continue
            f = mp.mpf(f64(f))
            rows.append((d1, d2, repr(f64(f)), fmt(sf(f))))
    return distinct(rows)


# ---------------------------------------------------------------------------
# Weighted chi-square: sum_j lambda_j chi^2_{h_j}
# ---------------------------------------------------------------------------


def cgf_derivatives(terms, c):
    k1 = k2 = k3 = mp.mpf(0)
    for lam, h in terms:
        m = 1 - 2 * lam * c
        k1 += h * lam / m
        k2 += 2 * h * lam**2 / m**2
        k3 += 8 * h * lam**3 / m**3
    return k1, k2, k3


def saddle(terms, x):
    """c in (0, s_+) with K'(c) = x + 1/c."""
    positive = [lam for lam, _ in terms if lam > 0]
    s_plus = 1 / (2 * max(positive)) if positive else mp.inf

    def dpsi(c):
        return cgf_derivatives(terms, c)[0] - x - 1 / c

    lo = mp.mpf(0)
    if s_plus == mp.inf:
        hi = mp.mpf(1)
        while dpsi(hi) < 0:
            hi *= 2
    else:
        hi = s_plus
    # Bisect on the increasing dpsi; the interval is open at both ends.
    a, b = lo, hi
    # Any c in (0, s_+) gives the same integral; the saddle only makes the
    # integrand non-oscillatory, so it is located to far less than working precision.
    while b - a > b * max(mp.mpf(10) ** -20, 1000 * mp.eps):
        mid = (a + b) / 2
        if dpsi(mid) < 0:
            a = mid
        else:
            b = mid
    return (a + b) / 2, s_plus


def psi(terms, x, s):
    acc = -s * x - mp.log(s)
    for lam, h in terms:
        acc -= h / 2 * mp.log(1 - 2 * lam * s)
    return acc


def bromwich(terms, x, curvature_fraction):
    c, s_plus = saddle(terms, x)
    _, k2, _ = cgf_derivatives(terms, c)
    width = 1 / mp.sqrt(k2 + 1 / c**2)
    # The contour opens toward the nearest singularity: s_+ for x >= 0, the
    # pole at 0 for x < 0. It is the hyperbola with vertex c, vertex curvature
    # 2a and 45-degree asymptotes, so it stays outside the wedge |Im| < |Re - c|
    # that holds every singularity on its open side: none is ever approached
    # closer than 1/sqrt(2) of its distance from c.
    if x >= 0:
        side, distance = 1, s_plus - c
    else:
        side, distance = -1, c
    a = curvature_fraction / distance
    psi_c = psi(terms, x, c)

    def integrand(u):
        root = mp.sqrt(1 + 4 * a * a * u * u)
        s = c + side * 2 * a * u * u / (root + 1) + 1j * u
        ds = side * 2 * a * u / root + 1j
        return mp.im(mp.exp(psi(terms, x, s) - psi_c) * ds)

    # Geometric breaks out past where the smallest weight's factor turns on
    # (|lambda| u ~ 1); before that the decay rate changes each time another
    # weight becomes active.
    smallest = min(abs(lam) for lam, _ in terms)
    breaks = [mp.mpf(0)]
    while breaks[-1] < max(width * 4**11, 64 / smallest):
        breaks.append(width * 4 ** (len(breaks) - 1))
    breaks.append(mp.inf)
    integral = mp.quad(integrand, breaks)
    return mp.exp(psi_c) * integral / mp.pi


def weighted_sf(terms, x, check=True):
    terms = [(mp.mpf(lam), mp.mpf(h)) for lam, h in terms]
    x = mp.mpf(x)
    # Q has no mass on the far side of zero from all its weights.
    if x <= 0 and all(lam > 0 for lam, _ in terms):
        return mp.mpf(1)
    if x >= 0 and all(lam < 0 for lam, _ in terms):
        return mp.mpf(0)
    first = bromwich(terms, x, mp.mpf("0.5"))
    if not check:
        return first
    second = bromwich(terms, x, mp.mpf("0.2"))
    assert abs(first - second) <= AGREEMENT * abs(first), (terms[:4], x, first, second)
    return first


def check_formula():
    # Equal weights: lambda * chi^2_{sum h}.
    for x in ["0.3", "4", "60"]:
        got = weighted_sf([("2", "1"), ("2", "2.5")], mp.mpf(x) + 0)
        want = chi_square_sf(mp.mpf(x) / 2, "3.5")
        assert abs(got - want) <= AGREEMENT * want, (x, got, want)
    # F as the two-term signed combination at zero.
    for a, b, f in [("1", "5", "0.05"), ("3", "26", "9"), ("0.7", "24", "40")]:
        a_, b_, f_ = mp.mpf(a), mp.mpf(b), mp.mpf(f)
        got = weighted_sf([(1, a_), (-f_ * a_ / b_, b_)], 0)
        want = f_sf(f_, a, b)
        assert abs(got - want) <= AGREEMENT * want, (a, b, f, got, want)


def log_spaced(lo_exp, hi_exp, count):
    return [mp.mpf(10) ** (lo_exp + (hi_exp - lo_exp) * k / (count - 1)) for k in range(count)]


def weighted_cases():
    cases = {}
    cases["three_positive"] = [(1, 1), ("0.5", 1), ("0.25", 1)]
    cases["dynamic_range"] = [(w, 1) for w in log_spaced(-12, 3, 16)]
    cases["one_dominant"] = [(1, 1)] + [(mp.mpf(10) ** -k, 1) for k in range(3, 13, 3)]
    cases["fractional_df"] = [(1, "0.5"), ("0.3", "2.7"), ("0.01", "0.001")]
    cases["many_components"] = [(mp.mpf(1) / j**2, 1) for j in range(1, 1001)]
    cases["mixed_signs"] = [(2, 1), (1, 1), ("-0.5", 1), ("-1.5", 1)]
    cases["mostly_negative"] = [("0.1", 1), (-1, 3), ("-2", "2")]
    cases["profiled_scale_ratio"] = [
        (1, 1),
        ("0.3", 1),
        ("0.05", 1),
        ("-0.04", 1),
        ("-0.009", 1),
        ("-0.02", 50),
    ]
    cases["equal_negative"] = [(-2, 1), (-2, 1)]
    return cases


def weighted_targets(terms, positive, negative):
    """Statistics for the tested tail levels, on whichever side of zero has them.

    A positive statistic reaches the levels below P(Q > 0), a negative one the
    levels above it, since P(Q > -y) increases with y.
    """
    def search(x):
        with mp.workdps(20):
            return weighted_sf(terms, x, check=False)

    at_zero = search(mp.mpf(0)) if positive else mp.mpf(0)
    xs = []
    if positive:
        for p in TARGETS:
            if p < at_zero:
                xs.append(statistic_for_tail(search, p, -60, 20))
        if negative:
            xs.append(mp.mpf(0))
    if negative:
        for p in TARGETS:
            if p > at_zero:
                log_y = solve_increasing_log(
                    lambda t: mp.log(search(-mp.exp(t))), mp.log(p), -800, 20
                )
                xs.append(-mp.exp(log_y))
    return xs


def weighted_case_rows(item):
    name, terms = item
    rows = []
    # The reference is computed at the doubles the Rust side receives.
    terms = [(mp.mpf(f64(mp.mpf(lam))), mp.mpf(f64(mp.mpf(h)))) for lam, h in terms]
    positive = any(lam > 0 for lam, _ in terms)
    negative = any(lam < 0 for lam, _ in terms)
    if name == "many_components":
        mean = sum(lam * h for lam, h in terms)
        xs = [mean * k for k in [mp.mpf("0.1"), 1, 5, 20, 100]]
    else:
        xs = weighted_targets(terms, positive, negative)
    for x in xs:
        x = mp.mpf(f64(x))
        value = weighted_sf(terms, x)
        if value == 0 or value == 1:
            continue
        rows.append((name, repr(f64(x)), fmt(value)))
    print(name, len(rows), flush=True)
    return rows


def weighted_rows():
    cases = weighted_cases()
    term_rows = [
        (name, repr(f64(mp.mpf(lam))), repr(f64(mp.mpf(h))))
        for name, terms in cases.items()
        for lam, h in terms
    ]
    with multiprocessing.Pool() as pool:
        per_case = pool.map(weighted_case_rows, list(cases.items()), chunksize=1)
    return term_rows, [row for rows in per_case for row in rows]


def write(name, header, rows):
    lines = ["\t".join(header)] + ["\t".join(str(v) for v in row) for row in rows]
    (HERE / name).write_text("\n".join(lines) + "\n")


def main():
    check_formula()
    write("chi_square.tsv", ["df", "statistic", "sf"], chi_square_rows())
    write("student_t.tsv", ["df", "statistic", "two_sided"], t_rows())
    write("fisher_snedecor.tsv", ["df1", "df2", "statistic", "sf"], f_rows())
    term_rows, rows = weighted_rows()
    write("weighted_terms.tsv", ["case", "weight", "df"], term_rows)
    write("weighted_chi_square.tsv", ["case", "statistic", "sf"], rows)


if __name__ == "__main__":
    main()
