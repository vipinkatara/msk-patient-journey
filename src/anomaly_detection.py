"""
Statistical process-control style anomaly detection over weekly operational
metrics, plus a root-cause "most associated factors" ranking.

Methods used:
- Rolling mean / rolling standard deviation baselines
- Z-score relative to rolling baseline
- Control limits (mean +/- k * std, Shewhart-style)

All findings are described as statistical associations, never as proven
causes, per the app's methodology.
"""

import numpy as np
import pandas as pd
from scipy import stats

METRIC_DEFINITIONS = {
    "treatment_start_rate": ("treatment_started", "referral"),
    "followup_rate": ("followup_completed", "treatment_started"),
    "no_show_rate": ("no_show", "appointment_scheduled"),
    "cancellation_rate": ("cancelled", "appointment_scheduled"),
    "engagement_30_rate": ("engagement_30", "followup_completed"),
    "retained_90_rate": ("retained_90", "engagement_30"),
}

METRIC_LABELS = {
    "treatment_start_rate": "Treatment-Start Rate",
    "followup_rate": "Follow-up Rate",
    "no_show_rate": "No-show Rate",
    "cancellation_rate": "Cancellation Rate",
    "engagement_30_rate": "30-Day Engagement Rate",
    "retained_90_rate": "90-Day Retention Rate",
    "avg_wait_days": "Appointment Wait Time (days)",
}


def weekly_metric_series(df: pd.DataFrame, metric: str, clinic: str = None,
                          date_col: str = "referral_date") -> pd.DataFrame:
    """Build a weekly time series for a given metric, optionally scoped to one clinic."""
    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col])
    if clinic and clinic != "All Clinics":
        work = work[work["clinic"] == clinic]
    work["week"] = work[date_col].dt.to_period("W").apply(lambda p: p.start_time)

    if metric == "avg_wait_days":
        g = work.groupby("week")["days_to_first_visit"].mean().reset_index()
        g.columns = ["week", "value"]
        g["n"] = work.groupby("week").size().values
        return g.dropna()

    numer_col, denom_col = METRIC_DEFINITIONS[metric]
    rows = []
    for week, g in work.groupby("week"):
        denom = g[denom_col].sum() if denom_col != "referral" else len(g)
        numer = g[numer_col].sum()
        if denom_col == "referral":
            denom = len(g)
        if denom and denom >= 5:
            rows.append({"week": week, "value": numer / denom * 100, "n": int(denom)})
    return pd.DataFrame(rows)


def detect_anomalies(series_df: pd.DataFrame, window: int = 6, z_threshold: float = 2.0) -> pd.DataFrame:
    """
    Flag weeks where the value falls outside rolling control limits.
    Returns the input df augmented with rolling_mean, rolling_std, z_score, is_anomaly.
    """
    if series_df.empty or len(series_df) < window + 2:
        out = series_df.copy()
        out["rolling_mean"] = np.nan
        out["rolling_std"] = np.nan
        out["z_score"] = np.nan
        out["is_anomaly"] = False
        return out

    df = series_df.sort_values("week").reset_index(drop=True).copy()
    df["rolling_mean"] = df["value"].rolling(window=window, min_periods=window).mean().shift(1)
    df["rolling_std"] = df["value"].rolling(window=window, min_periods=window).std().shift(1)
    df["z_score"] = (df["value"] - df["rolling_mean"]) / df["rolling_std"].replace(0, np.nan)
    df["upper_limit"] = df["rolling_mean"] + z_threshold * df["rolling_std"]
    df["lower_limit"] = df["rolling_mean"] - z_threshold * df["rolling_std"]
    df["is_anomaly"] = df["z_score"].abs() >= z_threshold
    return df


def scan_all_metrics(df: pd.DataFrame, clinics: list, window: int = 6, z_threshold: float = 2.0) -> pd.DataFrame:
    """Scan every metric x clinic combination for the latest flagged anomaly."""
    found = []
    for metric in METRIC_DEFINITIONS.keys() | {"avg_wait_days"}:
        for clinic in ["All Clinics"] + clinics:
            series = weekly_metric_series(df, metric, clinic=clinic)
            flagged = detect_anomalies(series, window=window, z_threshold=z_threshold)
            recent_anomalies = flagged[flagged["is_anomaly"]].tail(1)
            if not recent_anomalies.empty:
                row = recent_anomalies.iloc[0]
                direction = "above" if row["z_score"] > 0 else "below"
                pct_vs_baseline = ((row["value"] - row["rolling_mean"]) / row["rolling_mean"] * 100
                                    if row["rolling_mean"] not in (0, np.nan) else np.nan)
                found.append({
                    "metric": metric,
                    "metric_label": METRIC_LABELS.get(metric, metric),
                    "clinic": clinic,
                    "week": row["week"],
                    "value": round(row["value"], 2),
                    "baseline": round(row["rolling_mean"], 2),
                    "z_score": round(row["z_score"], 2),
                    "direction": direction,
                    "pct_vs_baseline": round(pct_vs_baseline, 1) if pd.notna(pct_vs_baseline) else None,
                    "n": int(row["n"]),
                })
    result = pd.DataFrame(found)
    if not result.empty:
        result = result.reindex(result["z_score"].abs().sort_values(ascending=False).index)
    return result


def root_cause_breakdown(df: pd.DataFrame, clinic: str, week, metric: str,
                          window_days: int = 21, date_col: str = "referral_date") -> dict:
    """
    Compare a baseline period vs the anomaly period across candidate explanatory factors,
    for a specific clinic/week/metric anomaly. Returns baseline-vs-anomaly comparisons and
    a ranked list of most-associated factors (by effect size / statistical significance).
    """
    import analysis as _analysis  # flat import: src/ is added to sys.path by app.py

    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col])
    if clinic and clinic != "All Clinics":
        work = work[work["clinic"] == clinic]

    week_ts = pd.Timestamp(week)
    anomaly_period = work[(work[date_col] >= week_ts) & (work[date_col] < week_ts + pd.Timedelta(days=7))]
    baseline_period = work[(work[date_col] >= week_ts - pd.Timedelta(days=window_days)) &
                            (work[date_col] < week_ts)]

    if len(anomaly_period) < 5 or len(baseline_period) < 5:
        return {"baseline": baseline_period, "anomaly": anomaly_period, "comparisons": [], "factors": []}

    numer_col, denom_col = METRIC_DEFINITIONS.get(metric, ("treatment_started", "referral"))

    def _rate(frame, col, denom_col):
        if denom_col == "referral":
            denom = len(frame)
        else:
            denom = frame[denom_col].sum()
        if denom == 0:
            return None, 0
        return frame[col].sum() / denom, int(denom)

    b_rate, b_n = _rate(baseline_period, numer_col, denom_col)
    a_rate, a_n = _rate(anomaly_period, numer_col, denom_col)

    comparisons = []
    comparisons.append({
        "factor": "Patient Volume",
        "baseline": len(baseline_period),
        "anomaly_period": len(anomaly_period),
    })
    for label, col in [("Days to Treatment", "days_to_treatment"), ("Days to First Visit", "days_to_first_visit")]:
        comparisons.append({
            "factor": label,
            "baseline": round(baseline_period[col].dropna().mean(), 1) if baseline_period[col].notna().any() else None,
            "anomaly_period": round(anomaly_period[col].dropna().mean(), 1) if anomaly_period[col].notna().any() else None,
        })
    for label, col in [("No-show Rate (%)", "no_show"), ("Cancellation Rate (%)", "cancelled")]:
        comparisons.append({
            "factor": label,
            "baseline": round(baseline_period[col].mean() * 100, 1),
            "anomaly_period": round(anomaly_period[col].mean() * 100, 1),
        })
    comparisons.append({
        "factor": METRIC_LABELS.get(metric, metric) + " (%)",
        "baseline": round(b_rate * 100, 1) if b_rate is not None else None,
        "anomaly_period": round(a_rate * 100, 1) if a_rate is not None else None,
    })

    # ---- rank candidate factors by association strength ----
    factors = []
    for label, col in [("Time to Treatment", "days_to_treatment"), ("Time to First Visit", "days_to_first_visit")]:
        b_vals, a_vals = baseline_period[col].dropna(), anomaly_period[col].dropna()
        if len(b_vals) >= 5 and len(a_vals) >= 5:
            t_stat, p_val = stats.ttest_ind(a_vals, b_vals, equal_var=False)
            d = _analysis.cohens_d(a_vals, b_vals)
            factors.append({"factor": label, "p_value": round(p_val, 4), "effect_size": abs(d) if pd.notna(d) else 0,
                             "direction": "increased" if a_vals.mean() > b_vals.mean() else "decreased"})

    for label, col in [("Cancellation Rate", "cancelled"), ("No-show Rate", "no_show")]:
        b_count, b_total = baseline_period[col].sum(), len(baseline_period)
        a_count, a_total = anomaly_period[col].sum(), len(anomaly_period)
        test = _analysis.two_proportion_test(int(a_count), a_total, int(b_count), b_total)
        if test:
            factors.append({"factor": label, "p_value": test["p_value"], "effect_size": abs(test["diff_pct_pts"]),
                             "direction": "increased" if test["diff_pct_pts"] > 0 else "decreased"})

    # referral mix shift (chi-square)
    try:
        b_mix = baseline_period["referral_source"].value_counts()
        a_mix = anomaly_period["referral_source"].value_counts()
        all_sources = sorted(set(b_mix.index) | set(a_mix.index))
        contingency = np.array([
            [b_mix.get(s, 0) for s in all_sources],
            [a_mix.get(s, 0) for s in all_sources],
        ])
        if contingency.sum() > 0 and contingency.shape[1] > 1:
            chi2, p_val, _, _ = stats.chi2_contingency(contingency)
            factors.append({"factor": "Referral Source Mix", "p_value": round(p_val, 4),
                             "effect_size": round(chi2, 2), "direction": "shifted"})
    except Exception:
        pass

    # provider concentration shift
    try:
        b_provider = baseline_period["provider"].value_counts(normalize=True)
        a_provider = anomaly_period["provider"].value_counts(normalize=True)
        common = set(b_provider.index) | set(a_provider.index)
        shift = sum(abs(a_provider.get(p, 0) - b_provider.get(p, 0)) for p in common) / 2
        factors.append({"factor": "Provider Distribution", "p_value": None,
                         "effect_size": round(shift * 100, 2), "direction": "shifted"})
    except Exception:
        pass

    factors_sorted = sorted(
        factors,
        key=lambda f: (f["p_value"] if f["p_value"] is not None else 1.0, -f["effect_size"])
    )

    return {
        "baseline": baseline_period, "anomaly": anomaly_period,
        "comparisons": comparisons, "factors": factors_sorted,
        "baseline_n": b_n, "anomaly_n": a_n,
    }
