"""
Lounge eligibility lookup model (BA Heathrow Terminal 3, Forage Data Science Task 1).

What this script does
1. Builds a lookup table of lounge eligibility rates (Tier 1, 2, 3) by flight grouping.
   Rate = eligible passengers for that tier / total seats, across all flights in the group.
2. Checks stability: the same rates calculated month by month.
3. Holdout test: build the table on an earlier period and apply it to an unseen month,
   then compare estimated and actual eligible passengers per tier.
4. Worked example: apply the final table to one morning window, flight by flight.
5. Time-of-day check: is any time-of-day pattern real, within each group? Uses a quasi-binomial
   test (allows for flight-to-flight variation) and a permutation test, with Holm adjustment.
   A plain binomial test is NOT used: eligible counts vary 2 to 3.5 times more between flights
   than independent seats would give, which makes binomial p-values far too small.
6. apply_lookup(): the reusable bit. Give it any future schedule (with a grouping column
   and seat columns) and it returns expected eligible passengers per tier.

Usage
    python lounge_eligibility_model.py path/to/schedule.xlsx --out-dir results

Notes and assumptions
- The dataset has seats, not passenger counts, so rates are shares of seats (assumes
  flights are full, or that load factor is similar across groups).
- Each tier is reported as its own rate. The data does not show whether the tiers overlap
  (nested or separate), so the rates are never summed and no "not eligible" share is derived.
- Tier 1 (Concorde Room) is hypothetical: Terminal 3 has no such lounge today.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

TIERS = ["TIER1_ELIGIBLE_PAX", "TIER2_ELIGIBLE_PAX", "TIER3_ELIGIBLE_PAX"]
SEAT_COLS = ["FIRST_CLASS_SEATS", "BUSINESS_CLASS_SEATS", "ECONOMY_SEATS"]


def load_schedule(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df["SEATS"] = df[SEAT_COLS].sum(axis=1)
    df["MONTH"] = df["FLIGHT_DATE"].dt.to_period("M")
    df["HOUR"] = pd.to_datetime(df["FLIGHT_TIME"].astype(str), format="%H:%M:%S").dt.hour
    return df


def build_lookup(df: pd.DataFrame, group_col: str = "HAUL") -> pd.DataFrame:
    """Eligibility rate per tier for each group, as a fraction of seats."""
    totals = df.groupby(group_col)[TIERS + ["SEATS"]].sum()
    rates = totals[TIERS].div(totals["SEATS"], axis=0)
    rates.columns = ["TIER1_RATE", "TIER2_RATE", "TIER3_RATE"]
    rates["FLIGHTS"] = df.groupby(group_col).size()
    return rates


def monthly_stability(df: pd.DataFrame, group_col: str = "HAUL") -> pd.DataFrame:
    """Tier rates per group per month, plus the max-min range per group."""
    rows = []
    for (grp, month), d in df.groupby([group_col, "MONTH"]):
        r = d[TIERS].sum() / d["SEATS"].sum()
        rows.append({group_col: grp, "MONTH": str(month), "FLIGHTS": len(d),
                     "TIER1_RATE": r.iloc[0], "TIER2_RATE": r.iloc[1], "TIER3_RATE": r.iloc[2]})
    out = pd.DataFrame(rows)
    return out


def apply_lookup(schedule: pd.DataFrame, lookup: pd.DataFrame, group_col: str = "HAUL") -> pd.DataFrame:
    """Expected eligible passengers per tier for any schedule with group_col and SEATS."""
    out = schedule.copy()
    for tier, rate in zip(["T1", "T2", "T3"], ["TIER1_RATE", "TIER2_RATE", "TIER3_RATE"]):
        out[f"EST_{tier}"] = out["SEATS"] * out[group_col].map(lookup[rate])
    return out


def holdout_test(df, group_col, train_months, test_month):
    train = df[df["MONTH"].astype(str).isin(train_months)]
    test = df[df["MONTH"].astype(str) == test_month]
    lookup = build_lookup(train, group_col)
    est = apply_lookup(test, lookup, group_col)
    rows = []
    for grp, d in est.groupby(group_col):
        for tier, col in zip(["T1", "T2", "T3"], TIERS):
            rows.append({group_col: grp, "TIER": tier, "ESTIMATED": d[f"EST_{tier}"].sum(),
                         "ACTUAL": d[col].sum()})
    res = pd.DataFrame(rows)
    total = res.groupby("TIER")[["ESTIMATED", "ACTUAL"]].sum().reset_index()
    total[group_col] = "ALL"
    res = pd.concat([res, total], ignore_index=True)
    res["ERROR_PCT"] = (res["ESTIMATED"] - res["ACTUAL"]) / res["ACTUAL"] * 100
    return res, len(train), len(test)


def worked_example(df, lookup, group_col, date, start_hour, end_hour):
    d = df[(df["FLIGHT_DATE"] == date) & df["HOUR"].between(start_hour, end_hour)]
    d = d.sort_values("FLIGHT_TIME")
    est = apply_lookup(d, lookup, group_col)
    cols = ["FLIGHT_NO", "FLIGHT_TIME", "ARRIVAL_STATION_CD", group_col, "SEATS",
            "EST_T1", "EST_T2", "EST_T3"] + TIERS
    return est[cols]


def _holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj, running, m = np.empty_like(p), 0.0, len(p)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def time_of_day_tests(df, group_col="HAUL", time_col="TIME_OF_DAY", n_perm=2000, seed=42):
    """Does the eligibility rate differ by time of day within each group, for each tier?

    Returns ANOVA p, quasi-binomial F p, dispersion, permutation p, and Holm-adjusted p
    (Holm applied across all group x tier tests, which are treated as one family).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for grp, g in df.groupby(group_col):
        n = g["SEATS"].values.astype(float)
        codes0, _ = pd.factorize(g[time_col])
        k = codes0.max() + 1
        for tier in TIERS:
            y = g[tier].values.astype(float)
            p0 = y.sum() / n.sum()
            # ANOVA on flight-level rates (unweighted), for comparison
            rate = y / n
            _, p_anova = stats.f_oneway(*[rate[codes0 == j] for j in range(k)])
            # quasi-binomial: binomial likelihood-ratio statistic divided by the dispersion
            ys = np.bincount(codes0, weights=y, minlength=k)
            ns = np.bincount(codes0, weights=n, minlength=k)
            pc = np.clip(ys / ns, 1e-12, 1 - 1e-12)
            pm = pc[codes0]
            ll = lambda p: (y * np.log(p) + (n - y) * np.log(1 - p)).sum()
            lr = 2 * (ll(pm) - ll(np.full(len(y), p0)))
            resid_df = len(y) - k
            phi = (((y - n * pm) ** 2) / (n * pm * (1 - pm))).sum() / resid_df
            p_qb = stats.f.sf(lr / (k - 1) / phi, k - 1, resid_df)
            # permutation test on the seat-weighted chi-square statistic
            def stat(c):
                a = np.bincount(c, weights=y, minlength=k)
                b = np.bincount(c, weights=n, minlength=k)
                return (((a - b * p0) ** 2) / (b * p0 * (1 - p0))).sum()
            obs, c, cnt = stat(codes0), codes0.copy(), 0
            for _ in range(n_perm):
                rng.shuffle(c)
                cnt += stat(c) >= obs
            rows.append({group_col: grp, "TIER": tier.replace("_ELIGIBLE_PAX", ""),
                         "ANOVA_P": p_anova, "DISPERSION": phi, "QUASI_BINOMIAL_P": p_qb,
                         "PERMUTATION_P": (cnt + 1) / (n_perm + 1)})
    out = pd.DataFrame(rows)
    out["QUASI_BINOMIAL_HOLM_P"] = _holm(out["QUASI_BINOMIAL_P"].values)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("schedule")
    p.add_argument("--group-by", default="HAUL")
    p.add_argument("--train-months", nargs="+", default=["2025-04", "2025-05", "2025-06"])
    p.add_argument("--test-month", default="2025-07")
    p.add_argument("--example-date", default="2025-07-15")
    p.add_argument("--example-hours", nargs=2, type=int, default=[6, 11])
    p.add_argument("--out-dir", default="results")
    a = p.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = load_schedule(a.schedule)

    lookup = build_lookup(df, a.group_by)
    lookup.to_csv(out / "lookup_table.csv")
    print("\nLookup table (share of seats):\n", (lookup[["TIER1_RATE", "TIER2_RATE", "TIER3_RATE"]] * 100).round(2))

    stab = monthly_stability(df, a.group_by)
    stab.to_csv(out / "monthly_stability.csv", index=False)
    rng = stab.groupby(a.group_by)[["TIER1_RATE", "TIER2_RATE", "TIER3_RATE"]].agg(lambda s: (s.max() - s.min()) * 100)
    print("\nMonthly range of rates (percentage points):\n", rng.round(2))

    res, n_train, n_test = holdout_test(df, a.group_by, a.train_months, a.test_month)
    res.to_csv(out / "holdout_test.csv", index=False)
    print(f"\nHoldout: train {n_train} flights ({', '.join(a.train_months)}), test {n_test} flights ({a.test_month})")
    print(res.round(1).to_string(index=False))

    tod = time_of_day_tests(df, a.group_by)
    tod.to_csv(out / "time_of_day_tests.csv", index=False)
    print("\nTime-of-day tests within each group (Holm across all tests; p above 0.05 means no reliable effect):")
    print(tod.round(4).to_string(index=False))

    ex = worked_example(df, lookup, a.group_by, pd.Timestamp(a.example_date), *a.example_hours)
    ex.to_csv(out / "worked_example.csv", index=False)
    print(f"\nWorked example: {a.example_date}, hours {a.example_hours[0]}-{a.example_hours[1]}, {len(ex)} flights")
    print(ex.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
