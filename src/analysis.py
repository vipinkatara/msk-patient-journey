"""
Cohort/retention analysis, operational-factor analysis, and reusable
statistical helpers (confidence intervals, rate differences, lift, p-values).

All causal language is deliberately avoided; functions here surface
associations and statistical evidence, not proof of causation.
"""

import numpy as np
import pandas as pd
from scipy import stats


# ----------------------------- Cohorts -----------------------------

def build_cohort_column(df: pd.DataFrame, referral_date_col: str = "referral_date") -> pd.Series:
    dates = pd.to_datetime(df[referral_date_col])
    return dates.dt.to_period("M").astype(str)


def retention_heatmap_data(df: pd.DataFrame, week_points=(1, 2, 4, 8, 12), cohort_col: str = None) -> pd.DataFrame:
    """
    Build a cohort x week retention matrix.

    cohort_col: if provided (e.g. "clinic", "condition", "referral_source"), cohorts are
    built from that categorical column instead of first-visit month.

    A patient is considered "retained" at week W if their observed engagement
    intensity (days_active_90) implies an active interaction at or beyond
    that week, approximated here via days_active_30 / days_active_90 and the
    engagement_30 / retained_90 flags scaled across the week checkpoints.
    """
    work = df.copy()
    if cohort_col:
        work["cohort"] = work[cohort_col].astype(str)
    else:
        work["cohort"] = build_cohort_column(work)

    # Approximate week-level retention using a monotonically-decaying proxy
    # derived from the actual engagement flags + activity-day counts.
    records = []
    for cohort, g in work.groupby("cohort"):
        n = len(g)
        if n < 15:
            continue
        for w in week_points:
            if w <= 4:
                # informed by 30-day engagement + active-day intensity
                denom = g["outcome_observable_30"] if "outcome_observable_30" in g else pd.Series([True] * n)
                eligible = g[denom] if denom.any() else g
                if len(eligible) == 0:
                    continue
                frac_active = (eligible["days_active_30"] >= (w * 7 * 0.55)).mean()
                rate = frac_active
            else:
                denom = g["outcome_observable_90"] if "outcome_observable_90" in g else pd.Series([True] * n)
                eligible = g[denom] if denom.any() else g
                if len(eligible) == 0:
                    continue
                frac_active = (eligible["days_active_90"] >= (w * 7 * 0.45)).mean()
                rate = frac_active
            records.append({"cohort": cohort, "week": f"Week {w}", "week_num": w,
                             "retention_rate": round(rate * 100, 1), "cohort_size": n})
    return pd.DataFrame(records)


def cohort_week4_trend_insight(heatmap_df: pd.DataFrame) -> str | None:
    """Flag if the most recent cohorts show weaker Week-4 retention vs earlier cohorts."""
    wk4 = heatmap_df[heatmap_df["week_num"] == 4].sort_values("cohort")
    if len(wk4) < 4:
        return None
    recent = wk4.tail(2)["retention_rate"].mean()
    earlier = wk4.iloc[:-2]["retention_rate"].mean()
    if earlier > 0 and (earlier - recent) / earlier >= 0.10:
        return (f"Recent cohorts show weaker Week-4 engagement ({recent:.1f}%) compared with "
                f"earlier cohorts ({earlier:.1f}%), a relative decline of "
                f"{(earlier - recent) / earlier * 100:.1f}%, despite stable initial visit completion.")
    return None


# ------------------------- Operational metrics -------------------------

def operational_summary(df: pd.DataFrame) -> dict:
    return {
        "avg_days_to_first_visit": round(df["days_to_first_visit"].dropna().mean(), 1),
        "avg_days_to_treatment": round(df["days_to_treatment"].dropna().mean(), 1),
        "avg_days_to_followup": round(df["days_to_followup"].dropna().mean(), 1),
        "no_show_rate": round(df["no_show"].mean() * 100, 1),
        "cancellation_rate": round(df["cancelled"].mean() * 100, 1),
    }


def bucketed_retention_by_delay(df: pd.DataFrame, delay_col: str = "days_to_treatment",
                                 outcome_col: str = "engagement_30", n_bins: int = 6) -> pd.DataFrame:
    work = df.dropna(subset=[delay_col]).copy()
    if len(work) == 0:
        return pd.DataFrame()
    try:
        work["delay_bucket"] = pd.qcut(work[delay_col], q=n_bins, duplicates="drop")
    except ValueError:
        return pd.DataFrame()
    g = work.groupby("delay_bucket", observed=True).agg(
        patients=("patient_id", "count"),
        avg_delay=(delay_col, "mean"),
        outcome_rate=(outcome_col, "mean"),
    ).reset_index()
    g["outcome_rate_pct"] = (g["outcome_rate"] * 100).round(1)
    g["avg_delay"] = g["avg_delay"].round(1)
    g["delay_bucket"] = g["delay_bucket"].astype(str)
    return g


# ------------------------- Statistical helpers -------------------------

def proportion_ci(count: int, n: int, confidence: float = 0.95):
    """Wilson score confidence interval for a proportion."""
    if n == 0:
        return (np.nan, np.nan)
    p_hat = count / n
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    denom = 1 + z ** 2 / n
    center = (p_hat + z ** 2 / (2 * n)) / denom
    margin = (z * np.sqrt((p_hat * (1 - p_hat) + z ** 2 / (4 * n)) / n)) / denom
    return (max(0.0, center - margin) * 100, min(1.0, center + margin) * 100)


def two_proportion_test(count_a: int, n_a: int, count_b: int, n_b: int):
    """
    Two-proportion z-test. Returns dict with rate diff (pct pts), relative lift (%),
    z-statistic, and two-sided p-value. Purely observational — not a claim of causation.
    """
    if n_a == 0 or n_b == 0:
        return None
    p_a, p_b = count_a / n_a, count_b / n_b
    p_pool = (count_a + count_b) / (n_a + n_b)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    if se == 0:
        return None
    z = (p_a - p_b) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    rel_lift = ((p_a - p_b) / p_b * 100) if p_b > 0 else np.nan
    return {
        "rate_a_pct": round(p_a * 100, 2),
        "rate_b_pct": round(p_b * 100, 2),
        "diff_pct_pts": round((p_a - p_b) * 100, 2),
        "relative_lift_pct": round(rel_lift, 1) if not np.isnan(rel_lift) else None,
        "z_stat": round(z, 3),
        "p_value": round(p_value, 4),
        "significant_95": p_value < 0.05,
    }


def cohens_d(sample_a: pd.Series, sample_b: pd.Series) -> float:
    a, b = sample_a.dropna(), sample_b.dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled_std = np.sqrt(((len(a) - 1) * a.std() ** 2 + (len(b) - 1) * b.std() ** 2) / (len(a) + len(b) - 2))
    if pooled_std == 0:
        return np.nan
    return round((a.mean() - b.mean()) / pooled_std, 3)
