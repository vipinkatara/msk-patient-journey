"""
metrics.py
----------
Core funnel and KPI calculations shared across the app's pages.
"""

import numpy as np
import pandas as pd

FUNNEL_STAGES = [
    ("referral", "Referral", None),
    ("appointment_scheduled", "Appointment Scheduled", "appointment_scheduled"),
    ("first_visit_completed", "First Visit Completed", "first_visit_completed"),
    ("treatment_started", "Treatment Started", "treatment_started"),
    ("followup_completed", "Follow-up Completed", "followup_completed"),
    ("engagement_30", "30-Day Engagement", "engagement_30"),
    ("retained_90", "90-Day Retention", "retained_90"),
]

STAGE_DEFINITIONS = {
    "Referral": "A new patient enters the MSK care pathway via any referral source.",
    "Appointment Scheduled": "An initial evaluation appointment was booked following referral.",
    "First Visit Completed": "The patient attended the scheduled initial evaluation (excludes no-shows/cancellations).",
    "Treatment Started": "The patient began a recommended treatment plan (e.g., PT, intervention) after evaluation.",
    "Follow-up Completed": "The patient completed at least one follow-up visit after starting treatment.",
    "30-Day Engagement": "The patient had a qualifying care interaction within 30 days of follow-up completion.",
    "90-Day Retention": "The patient remained actively engaged in care through the 90-day mark.",
}


def funnel_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Return a stage-by-stage funnel table with counts, conversion, and drop-off."""
    rows = []
    n_referrals = len(df)
    prev_count = n_referrals
    for key, label, col in FUNNEL_STAGES:
        if col is None:
            count = n_referrals
        else:
            count = int(df[col].sum())
        conv_from_prev = (count / prev_count * 100) if prev_count > 0 else 0.0
        conv_from_start = (count / n_referrals * 100) if n_referrals > 0 else 0.0
        drop_off = 100 - conv_from_prev
        rows.append({
            "stage": label,
            "count": count,
            "conversion_from_prev_pct": round(conv_from_prev, 1),
            "conversion_from_start_pct": round(conv_from_start, 1),
            "drop_off_pct": round(drop_off, 1),
        })
        prev_count = count
    return pd.DataFrame(rows)


def largest_drop(funnel_df: pd.DataFrame) -> dict:
    """Identify the stage transition with the largest percentage drop-off (excluding stage 0)."""
    sub = funnel_df.iloc[1:].copy()
    idx = sub["drop_off_pct"].idxmax()
    row = sub.loc[idx]
    prev_label = funnel_df.iloc[funnel_df.index.get_loc(idx) - 1]["stage"]
    return {
        "from_stage": prev_label,
        "to_stage": row["stage"],
        "drop_off_pct": row["drop_off_pct"],
    }


def kpi_summary(df: pd.DataFrame, prev_df: pd.DataFrame = None) -> list:
    """
    Build the top-line KPI cards: value, previous-period comparison, % change, definition.
    prev_df should be the same filtered dataframe applied to the prior comparable period.
    """
    kpis = [
        ("New Referrals", len(df), len(prev_df) if prev_df is not None else None,
         "Count of new patient referrals entering the MSK pathway in the selected period.", "count"),
        ("Appointment Conversion", _rate(df, "appointment_scheduled"),
         _rate(prev_df, "appointment_scheduled") if prev_df is not None else None,
         "Share of referrals that resulted in a scheduled initial appointment.", "pct"),
        ("First Visit Completion", _rate(df, "first_visit_completed"),
         _rate(prev_df, "first_visit_completed") if prev_df is not None else None,
         "Share of referrals whose initial visit was completed (excludes no-shows/cancellations).", "pct"),
        ("Treatment Start Rate", _rate(df, "treatment_started"),
         _rate(prev_df, "treatment_started") if prev_df is not None else None,
         "Share of referrals who began a recommended treatment plan.", "pct"),
        ("Follow-up Completion", _rate(df, "followup_completed"),
         _rate(prev_df, "followup_completed") if prev_df is not None else None,
         "Share of referrals who completed at least one follow-up visit.", "pct"),
        ("30-Day Engagement", _rate(df, "engagement_30"),
         _rate(prev_df, "engagement_30") if prev_df is not None else None,
         "Share of referrals actively engaged in care at the 30-day mark.", "pct"),
        ("90-Day Retention", _rate(df, "retained_90"),
         _rate(prev_df, "retained_90") if prev_df is not None else None,
         "Share of referrals still engaged in care at the 90-day mark.", "pct"),
    ]
    out = []
    for name, value, prev_value, definition, fmt in kpis:
        pct_change = None
        if prev_value not in (None, 0) and value is not None:
            pct_change = (value - prev_value) / prev_value * 100
        out.append({
            "name": name, "value": value, "prev_value": prev_value,
            "pct_change": pct_change, "definition": definition, "format": fmt,
        })
    return out


def _rate(df, col):
    if df is None or len(df) == 0:
        return None
    return round(df[col].mean() * 100, 1)


def segment_funnel(df: pd.DataFrame, segment_col: str, stage_col: str = "treatment_started") -> pd.DataFrame:
    """Compute a single-stage conversion rate broken out by a segment column, vs network median."""
    g = df.groupby(segment_col).agg(
        patients=("patient_id", "count"),
        conversion_rate=(stage_col, "mean"),
    ).reset_index()
    g["conversion_rate_pct"] = (g["conversion_rate"] * 100).round(1)
    median_rate = g["conversion_rate_pct"].median()
    g["vs_median_pct_pts"] = (g["conversion_rate_pct"] - median_rate).round(1)
    g = g.sort_values("conversion_rate_pct")
    g.attrs["median_rate"] = median_rate
    return g


def generate_segment_insight(seg_df: pd.DataFrame, segment_col: str, stage_label: str,
                              threshold_pts: float = 8.0) -> str | None:
    """Generate a plain-language insight only if a segment materially lags the median."""
    if seg_df.empty:
        return None
    worst = seg_df.iloc[0]
    if worst["vs_median_pct_pts"] <= -threshold_pts and worst["patients"] >= 30:
        return (f"**{worst[segment_col]}** has a materially lower {stage_label.lower()} "
                f"({worst['conversion_rate_pct']}%) than the network median "
                f"({seg_df.attrs.get('median_rate', 0):.1f}%), a gap of "
                f"{abs(worst['vs_median_pct_pts']):.1f} percentage points.")
    return None
