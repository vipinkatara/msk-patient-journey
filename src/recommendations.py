"""
Translates funnel / operational / anomaly findings into structured,
non-causal recommendation cards and a hypothetical experiment design.
"""

import pandas as pd
import analysis as _analysis


def _format_p(p_value) -> str:
    if p_value is None:
        return "p unavailable"
    return "p < 0.001" if p_value < 0.001 else f"p = {p_value}"


def build_recommendations(df: pd.DataFrame) -> list:
    recs = []

    # --- Recommendation 1: treatment-start delay vs engagement ---
    delay_df = df.dropna(subset=["days_to_treatment"]).copy()
    if len(delay_df) > 100:
        median_delay = delay_df["days_to_treatment"].median()
        fast = delay_df[delay_df["days_to_treatment"] <= 7]
        slow = delay_df[delay_df["days_to_treatment"] > 7]
        if len(fast) >= 30 and len(slow) >= 30:
            test = _analysis.two_proportion_test(
                int(fast["engagement_30"].sum()), len(fast),
                int(slow["engagement_30"].sum()), len(slow)
            )
            if test:
                recs.append({
                    "title": "Reduce Treatment-Start Delay",
                    "finding": (f"Patients who start treatment within 7 days show a 30-day engagement rate of "
                                f"{test['rate_a_pct']}%, compared with {test['rate_b_pct']}% for patients who "
                                f"wait longer than 7 days."),
                    "evidence": (f"Two-proportion comparison across {len(fast) + len(slow)} patients "
                                 f"(rate difference {test['diff_pct_pts']} pct pts, "
                                 f"relative lift {test['relative_lift_pct']}%, {_format_p(test['p_value'])}). "
                                 f"Association does not establish causation."),
                    "implication": "Scheduling delays may represent an important point of patient drop-off in the care journey.",
                    "next_step": "Investigate scheduling capacity and evaluate whether eligible patients can be offered earlier treatment slots.",
                    "measurement": "Track treatment-start time, 30-day engagement, and 90-day retention across upcoming cohorts.",
                    "significant": test["significant_95"],
                })

    # --- Recommendation 2: appointment wait time vs. no-show/cancellation likelihood ---
    wait_df = df.dropna(subset=["days_to_first_visit"]).copy()
    if len(wait_df) > 100:
        fast_wait = wait_df[wait_df["days_to_first_visit"] <= 7]
        slow_wait = wait_df[wait_df["days_to_first_visit"] > 7]
        if len(fast_wait) >= 30 and len(slow_wait) >= 30:
            missed_fast = int((fast_wait["no_show"] | fast_wait["cancelled"]).sum())
            missed_slow = int((slow_wait["no_show"] | slow_wait["cancelled"]).sum())
            test2 = _analysis.two_proportion_test(missed_slow, len(slow_wait), missed_fast, len(fast_wait))
            if test2 and test2["diff_pct_pts"] > 0:
                recs.append({
                    "title": "Shorten Time to First Appointment",
                    "finding": (f"Patients waiting longer than 7 days for their first appointment have a "
                                f"combined no-show/cancellation rate of {test2['rate_a_pct']}%, versus "
                                f"{test2['rate_b_pct']}% for patients seen within 7 days."),
                    "evidence": (f"Two-proportion comparison across {len(fast_wait) + len(slow_wait)} scheduled "
                                 f"patients (rate difference {test2['diff_pct_pts']} pct pts, "
                                 f"relative lift {test2['relative_lift_pct']}%, {_format_p(test2['p_value'])}). "
                                 f"Association does not establish causation."),
                    "implication": "Longer scheduling delays may give patients more opportunity to disengage before their first visit even occurs.",
                    "next_step": "Evaluate whether appointment capacity can be added or reallocated to shorten the referral-to-first-visit window.",
                    "measurement": "Track days-to-first-visit alongside no-show/cancellation rate for upcoming referral cohorts.",
                    "significant": test2["significant_95"],
                })

    # --- Recommendation 3: clinic-level treatment-start gap ---
    from metrics import segment_funnel, generate_segment_insight
    seg = segment_funnel(df, "clinic", "treatment_started")
    insight = generate_segment_insight(seg, "clinic", "Treatment Start Rate", threshold_pts=6.0)
    if insight:
        worst = seg.iloc[0]
        recs.append({
            "title": f"Investigate Treatment-Start Gap at {worst['clinic']}",
            "finding": insight.replace("**", ""),
            "evidence": (f"Segment-level conversion analysis across {int(worst['patients'])} referrals at "
                         f"{worst['clinic']}, compared with the network median."),
            "implication": "A persistent site-level gap suggests an operational factor specific to this location rather than patient mix alone.",
            "next_step": f"Review scheduling capacity, provider availability, and referral handling at {worst['clinic']}.",
            "measurement": "Track weekly treatment-start rate for this clinic against the network baseline using control limits.",
            "significant": True,
        })

    # --- Recommendation 4: referral source quality ---
    ref_seg = df.groupby("referral_source").agg(
        patients=("patient_id", "count"),
        treatment_rate=("treatment_started", "mean"),
    ).reset_index()
    ref_seg["treatment_rate_pct"] = (ref_seg["treatment_rate"] * 100).round(1)
    ref_seg = ref_seg.sort_values("treatment_rate_pct")
    if len(ref_seg) > 1:
        weakest = ref_seg.iloc[0]
        strongest = ref_seg.iloc[-1]
        if strongest["treatment_rate_pct"] - weakest["treatment_rate_pct"] >= 8 and weakest["patients"] >= 30:
            recs.append({
                "title": "Strengthen Lower-Performing Referral Channels",
                "finding": (f"'{weakest['referral_source']}' referrals convert to treatment start at "
                            f"{weakest['treatment_rate_pct']}%, versus {strongest['treatment_rate_pct']}% for "
                            f"'{strongest['referral_source']}' referrals."),
                "evidence": f"Comparison across {int(weakest['patients'])} vs {int(strongest['patients'])} referrals by source.",
                "implication": "Referral source appears associated with downstream conversion, possibly reflecting differences in patient readiness or intake handoff quality.",
                "next_step": f"Review the intake and scheduling handoff process for '{weakest['referral_source']}' referrals.",
                "measurement": "Track treatment-start rate by referral source monthly; compare intake time-to-contact by source.",
                "significant": True,
            })

    return recs


def experiment_design(recs: list) -> dict:
    """Generate a hypothetical experiment design based on the top recommendation."""
    if not recs:
        return None
    top = recs[0]
    return {
        "label": "Hypothetical experiment design — not a clinical recommendation.",
        "hypothesis": ("Reducing the time between initial evaluation and treatment start will improve "
                       "downstream patient engagement."),
        "intervention": "Offer eligible patients an earlier treatment slot via priority scheduling.",
        "control": "Current scheduling workflow (standard queue).",
        "treatment": "Priority scheduling intervention for eligible patients.",
        "primary_metric": "Treatment-start rate within 7 days of evaluation.",
        "secondary_metrics": ["30-day engagement", "90-day retention", "Cancellation rate"],
        "guardrails": ["Provider utilization", "Appointment availability for other patients", "Patient complaint rate"],
        "based_on": top["title"],
    }
