"""
Time-of-day audit for BA lounge eligibility (Forage Data Science Task 1).

Question: within each haul type, does the lounge eligibility rate (eligible pax / seats)
differ by TIME_OF_DAY, for each of Tiers 1, 2 and 3?

Only needs numpy, pandas, scipy, matplotlib (no statsmodels).
The binomial likelihood-ratio test for ONE categorical predictor has a closed form, so it is
computed directly. It matches statsmodels.GLM(Binomial) with seats as weights exactly.

Usage:  python lounge_time_of_day_audit.py schedule.xlsx --out-dir audit_results
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TIERS = {"Tier 1": "TIER1_ELIGIBLE_PAX", "Tier 2": "TIER2_ELIGIBLE_PAX", "Tier 3": "TIER3_ELIGIBLE_PAX"}
SEATS = ["FIRST_CLASS_SEATS", "BUSINESS_CLASS_SEATS", "ECONOMY_SEATS"]
ORDER = ["Morning", "Lunchtime", "Afternoon", "Evening"]
SEED = 42


# ------------------------------------------------------------------ 1. validation
def load_and_validate(path):
    df = pd.read_excel(path)
    df["N"] = df[SEATS].sum(axis=1)
    checks = {
        "rows": len(df),
        "missing values": int(df.isna().sum().sum()),
        "duplicate rows": int(df.duplicated().sum()),
        "zero-seat flights": int((df["N"] == 0).sum()),
        "negative counts": int((df[list(TIERS.values())] < 0).any(axis=1).sum()),
        "tier count > seats (any tier)": int((df[list(TIERS.values())].gt(df["N"], axis=0)).any(axis=1).sum()),
        "sum of tiers > seats": int((df[list(TIERS.values())].sum(axis=1) > df["N"]).sum()),
        "Tier1 > Tier2": int((df["TIER1_ELIGIBLE_PAX"] > df["TIER2_ELIGIBLE_PAX"]).sum()),
        "Tier1 > Tier3": int((df["TIER1_ELIGIBLE_PAX"] > df["TIER3_ELIGIBLE_PAX"]).sum()),
        "Tier2 > Tier3": int((df["TIER2_ELIGIBLE_PAX"] > df["TIER3_ELIGIBLE_PAX"]).sum()),
    }
    return df, checks


# ------------------------------------------------------------------ 2. tests
def anova_p(g, col):
    rate = g[col] / g["N"]
    groups = [x.values for _, x in rate.groupby(g["TIME_OF_DAY"])]
    return stats.f_oneway(*groups)


def binomial_lrt(g, col):
    """LRT of null (one rate) vs time-of-day rates, binomial likelihood with n = seats."""
    y, n = g[col].values.astype(float), g["N"].values.astype(float)
    p0 = y.sum() / n.sum()
    cell = g.groupby("TIME_OF_DAY").agg(y=(col, "sum"), n=("N", "sum"))
    p_cell = (cell["y"] / cell["n"]).clip(1e-12, 1 - 1e-12)
    pm = g["TIME_OF_DAY"].map(p_cell).values

    def ll(p):
        return (y * np.log(p) + (n - y) * np.log(1 - p)).sum()

    lr = 2 * (ll(pm) - ll(np.full(len(y), p0)))
    df_ = len(cell) - 1
    # Pearson dispersion of the time-of-day model: should be about 1 if binomial is right
    phi = (((y - n * pm) ** 2) / (n * pm * (1 - pm))).sum() / (len(y) - len(cell))
    resid_df = len(y) - len(cell)
    return lr, df_, stats.chi2.sf(lr, df_), phi, resid_df


def quasi_binomial_F(lr, df_, phi, resid_df):
    F = lr / df_ / phi
    return F, stats.f.sf(F, df_, resid_df)


def permutation_p(g, col, B=5000, seed=SEED):
    """Shuffle time-of-day labels; statistic = Pearson chi-square of cell rates vs haul rate."""
    rng = np.random.default_rng(seed)
    y, n = g[col].values.astype(float), g["N"].values.astype(float)
    codes0, _ = pd.factorize(g["TIME_OF_DAY"])
    k = codes0.max() + 1
    p0 = y.sum() / n.sum()

    def stat(codes):
        ys = np.bincount(codes, weights=y, minlength=k)
        ns = np.bincount(codes, weights=n, minlength=k)
        return (((ys - ns * p0) ** 2) / (ns * p0 * (1 - p0))).sum()

    obs, codes, cnt = stat(codes0), codes0.copy(), 0
    for _ in range(B):
        rng.shuffle(codes)
        cnt += stat(codes) >= obs
    return (cnt + 1) / (B + 1)


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


# ------------------------------------------------------------------ 3. effect sizes
def effect_table(df, B=2000, seed=SEED):
    """Time-specific rates and differences from the haul rate, with date-cluster bootstrap CIs.
    Resampling whole dates keeps flights from the same day together (allows for day-level effects)."""
    rng = np.random.default_rng(seed)
    rows = []
    for haul, g in df.groupby("HAUL"):
        dates = np.sort(g["FLIGHT_DATE"].unique())
        d_idx = {d: i for i, d in enumerate(dates)}
        di = g["FLIGHT_DATE"].map(d_idx).values
        ti = pd.Categorical(g["TIME_OF_DAY"], categories=ORDER).codes
        mean_seats = g["N"].mean()
        n_mat = np.zeros((len(dates), 4))
        np.add.at(n_mat, (di, ti), g["N"].values)
        for tier, col in TIERS.items():
            y_mat = np.zeros((len(dates), 4))
            np.add.at(y_mat, (di, ti), g[col].values)
            point_cell = y_mat.sum(0) / n_mat.sum(0)
            point_all = y_mat.sum() / n_mat.sum()
            boots = rng.integers(0, len(dates), size=(B, len(dates)))
            yb, nb = y_mat[boots].sum(1), n_mat[boots].sum(1)          # B x 4
            rate_b = yb / nb
            all_b = yb.sum(1) / nb.sum(1)
            for j, tod in enumerate(ORDER):
                lo, hi = np.percentile(rate_b[:, j], [2.5, 97.5])
                dlo, dhi = np.percentile(rate_b[:, j] - all_b, [2.5, 97.5])
                rows.append({
                    "Haul": haul, "Tier": tier, "Time of day": tod,
                    "Flights": int((ti == j).sum()),
                    "Rate %": 100 * point_cell[j], "CI low %": 100 * lo, "CI high %": 100 * hi,
                    "Diff vs haul rate (pp)": 100 * (point_cell[j] - point_all),
                    "Diff CI low (pp)": 100 * dlo, "Diff CI high (pp)": 100 * dhi,
                    "Diff in pax per average flight": (point_cell[j] - point_all) * mean_seats,
                })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 4. main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("schedule")
    ap.add_argument("--out-dir", default="audit_results")
    ap.add_argument("--perm", type=int, default=5000)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df, checks = load_and_validate(a.schedule)
    print("DATA VALIDATION")
    for k, v in checks.items():
        print(f"  {k}: {v}")
    print("\nFlights per haul x time of day:\n", pd.crosstab(df["HAUL"], df["TIME_OF_DAY"])[ORDER])

    rows = []
    for haul, g in df.groupby("HAUL"):
        for tier, col in TIERS.items():
            F, p_anova = anova_p(g, col)
            lr, df_, p_lrt, phi, rdf = binomial_lrt(g, col)
            Fq, p_q = quasi_binomial_F(lr, df_, phi, rdf)
            p_perm = permutation_p(g, col, a.perm)
            rows.append({"Haul": haul, "Tier": tier, "ANOVA p": p_anova, "Binomial LRT p": p_lrt,
                         "Dispersion phi": phi, "Quasi-binomial F p": p_q, "Permutation p": p_perm})
    tests = pd.DataFrame(rows)
    for c in ["ANOVA p", "Binomial LRT p", "Quasi-binomial F p", "Permutation p"]:
        tests[c.replace(" p", " Holm p")] = holm(tests[c].values)   # family = these 6 tests
    tests.to_csv(out / "test_results.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nTEST RESULTS (Holm applied across the 6 tests within each method)")
    print(tests.round(4).to_string(index=False))

    eff = effect_table(df, a.boot)
    eff.to_csv(out / "effect_sizes.csv", index=False)
    print("\nEFFECT SIZES (date-cluster bootstrap 95% CI)")
    print(eff.round(3).to_string(index=False))

    # forest-style plot: difference from the haul rate, in percentage points
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharex=False)
    for r, haul in enumerate(["LONG", "SHORT"]):
        for c, tier in enumerate(TIERS):
            ax = axes[r, c]
            s = eff[(eff["Haul"] == haul) & (eff["Tier"] == tier)].set_index("Time of day").loc[ORDER]
            y = np.arange(len(ORDER))
            ax.errorbar(s["Diff vs haul rate (pp)"], y,
                        xerr=[s["Diff vs haul rate (pp)"] - s["Diff CI low (pp)"],
                              s["Diff CI high (pp)"] - s["Diff vs haul rate (pp)"]],
                        fmt="o", color="#021B41", ecolor="#CE210F", capsize=3)
            ax.axvline(0, color="grey", lw=1, ls="--")
            ax.set_yticks(y)
            ax.set_yticklabels(ORDER if c == 0 else [])
            ax.invert_yaxis()
            ax.set_title(f"{haul.title()}-haul, {tier}")
            ax.set_xlabel("Difference from haul rate (percentage points)")
    fig.suptitle("Time-of-day differences in lounge eligibility rate (95% date-cluster bootstrap CI)")
    fig.tight_layout()
    fig.savefig(out / "time_of_day_effects.png", dpi=150)
    print(f"\nSaved results to {out.resolve()}")


if __name__ == "__main__":
    main()
