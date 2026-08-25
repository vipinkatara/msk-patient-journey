"""
Generates a fully synthetic MSK (musculoskeletal) patient journey dataset.

No real patient, provider, clinic, or organizational data is used anywhere
in this module. All names, distributions, and relationships are invented
for demonstration purposes only.

The generator builds in realistic (but noisy, non-deterministic) relationships
between operational factors and downstream engagement/retention, so that the
rest of the application has something meaningful to discover.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RANDOM_SEED = 42

CLINICS = [
    {"name": "Clinic A - Northgate", "base_wait": 4.0, "capacity": 1.10, "quality": 0.05},
    {"name": "Clinic B - Riverside", "base_wait": 9.5, "capacity": 0.80, "quality": -0.12},
    {"name": "Clinic C - Downtown", "base_wait": 5.5, "capacity": 1.00, "quality": 0.00},
    {"name": "Clinic D - Eastside", "base_wait": 6.5, "capacity": 0.95, "quality": -0.03},
    {"name": "Clinic E - Lakeview", "base_wait": 4.5, "capacity": 1.05, "quality": 0.04},
]

PROVIDERS_PER_CLINIC = 4

REFERRAL_SOURCES = [
    {"name": "Primary Care Physician", "weight": 0.42, "quality": 0.06},
    {"name": "Self-Referral", "weight": 0.18, "quality": -0.05},
    {"name": "Emergency Department", "weight": 0.10, "quality": -0.10},
    {"name": "Orthopedic Specialist", "weight": 0.15, "quality": 0.08},
    {"name": "Employer Wellness Program", "weight": 0.08, "quality": 0.02},
    {"name": "Online / Digital Intake", "weight": 0.07, "quality": -0.02},
]

CONDITIONS = [
    {"name": "Low Back Pain", "weight": 0.30, "engagement": 0.00},
    {"name": "Knee Osteoarthritis", "weight": 0.20, "engagement": 0.04},
    {"name": "Shoulder Impingement", "weight": 0.14, "engagement": 0.02},
    {"name": "Post-Surgical Rehab", "weight": 0.12, "engagement": 0.10},
    {"name": "Neck / Cervical Pain", "weight": 0.14, "engagement": -0.02},
    {"name": "Sports Injury", "weight": 0.10, "engagement": 0.06},
]

INSURANCE_TYPES = [
    {"name": "Commercial PPO", "weight": 0.38, "quality": 0.05},
    {"name": "Commercial HMO", "weight": 0.20, "quality": -0.02},
    {"name": "Medicare", "weight": 0.22, "quality": 0.00},
    {"name": "Medicaid", "weight": 0.13, "quality": -0.08},
    {"name": "Self-Pay", "weight": 0.07, "quality": -0.10},
]

AGE_GROUPS = [
    {"name": "18-34", "weight": 0.18, "engagement": -0.03},
    {"name": "35-49", "weight": 0.24, "engagement": 0.02},
    {"name": "50-64", "weight": 0.30, "engagement": 0.05},
    {"name": "65+", "weight": 0.28, "engagement": 0.03},
]


def _weighted_choice(rng, options, key="weight"):
    names = [o["name"] for o in options]
    if key in options[0]:
        weights = np.array([o[key] for o in options], dtype=float)
        weights = weights / weights.sum()
    else:
        weights = np.full(len(options), 1.0 / len(options))
    return rng.choice(names, p=weights)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate_patients(n_patients: int = 22000, seed: int = RANDOM_SEED,
                       start_date: str = "2024-06-01", end_date: str = "2026-06-01") -> pd.DataFrame:
    """
    Generate a synthetic patient-journey dataset with n_patients rows.

    The generation intentionally injects an operational "shock" for
    Clinic B in a recent window, and a mild network-wide Week-4 engagement
    softening in the most recent cohorts, so that anomaly detection and
    cohort analysis have real (synthetic) signal to surface.
    """
    rng = np.random.default_rng(seed)

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    total_days = (end - start).days

    clinic_lookup = {c["name"]: c for c in CLINICS}
    referral_lookup = {r["name"]: r for r in REFERRAL_SOURCES}
    condition_lookup = {c["name"]: c for c in CONDITIONS}
    insurance_lookup = {i["name"]: i for i in INSURANCE_TYPES}
    age_lookup = {a["name"]: a for a in AGE_GROUPS}

    # referral dates weighted slightly toward more recent months (growth)
    day_offsets = rng.triangular(0, total_days, total_days, n_patients).astype(int)
    referral_dates = start + pd.to_timedelta(day_offsets, unit="D")

    clinics = np.array([_weighted_choice(rng, CLINICS) for _ in range(n_patients)])
    providers = np.array([
        f"{c.split(' - ')[0]} Provider {rng.integers(1, PROVIDERS_PER_CLINIC + 1)}"
        for c in clinics
    ])
    referral_sources = np.array([_weighted_choice(rng, REFERRAL_SOURCES) for _ in range(n_patients)])
    conditions = np.array([_weighted_choice(rng, CONDITIONS) for _ in range(n_patients)])
    insurance = np.array([_weighted_choice(rng, INSURANCE_TYPES) for _ in range(n_patients)])
    age_groups = np.array([_weighted_choice(rng, AGE_GROUPS) for _ in range(n_patients)])

    df = pd.DataFrame({
        "patient_id": [f"P{100000 + i}" for i in range(n_patients)],
        "referral_date": referral_dates,
        "clinic": clinics,
        "provider": providers,
        "referral_source": referral_sources,
        "condition": conditions,
        "insurance_type": insurance,
        "age_group": age_groups,
    })

    # --- Anomaly injection: Clinic B degrades in a specific recent window ---
    anomaly_start = end - timedelta(days=95)
    anomaly_end = end - timedelta(days=35)
    df["clinic_b_shock"] = (
        (df["clinic"] == "Clinic B - Riverside") &
        (df["referral_date"] >= anomaly_start) &
        (df["referral_date"] <= anomaly_end)
    )

    # --- Composite quality score per patient (drives every downstream prob) ---
    clinic_quality = df["clinic"].map(lambda c: clinic_lookup[c]["quality"]).astype(float)
    referral_quality = df["referral_source"].map(lambda r: referral_lookup[r]["quality"]).astype(float)
    insurance_quality = df["insurance_type"].map(lambda i: insurance_lookup[i]["quality"]).astype(float)
    condition_engagement = df["condition"].map(lambda c: condition_lookup[c]["engagement"]).astype(float)
    age_engagement = df["age_group"].map(lambda a: age_lookup[a]["engagement"]).astype(float)

    noise = rng.normal(0, 0.35, n_patients)
    quality_score = (
        clinic_quality + referral_quality + insurance_quality
        + condition_engagement + age_engagement + noise
    )
    # Clinic B shock further suppresses quality during the anomaly window
    quality_score = quality_score - np.where(df["clinic_b_shock"], 0.55, 0.0)

    # --- Recent-cohort Week-4 softening (network-wide, subtle) ---
    days_since_referral = (end - df["referral_date"]).dt.days
    recent_cohort = days_since_referral < 60
    week4_soft_penalty = np.where(recent_cohort, rng.normal(0.18, 0.05, n_patients), 0.0)

    # ---------------- STAGE 1: Appointment Scheduled ----------------
    base_wait = df["clinic"].map(lambda c: clinic_lookup[c]["base_wait"]).astype(float)
    capacity = df["clinic"].map(lambda c: clinic_lookup[c]["capacity"]).astype(float)

    p_scheduled = _sigmoid(1.9 + quality_score * 1.1)
    scheduled = rng.random(n_patients) < p_scheduled
    df["appointment_scheduled"] = scheduled

    days_to_first_visit = rng.gamma(
        shape=2.2, scale=(base_wait / capacity) / 2.2
    ) + np.where(df["clinic_b_shock"], rng.gamma(2.0, 2.5, n_patients), 0.0)
    days_to_first_visit = np.clip(days_to_first_visit, 0.5, 90)

    # ---------------- STAGE 2: First Visit Completed ----------------
    no_show_prob = _sigmoid(-1.6 - quality_score * 0.9 + (days_to_first_visit - 7) * 0.02)
    no_show = (rng.random(n_patients) < no_show_prob) & scheduled
    cancelled_prob = _sigmoid(-2.0 - quality_score * 0.7 + (days_to_first_visit - 7) * 0.015)
    cancelled = (rng.random(n_patients) < cancelled_prob) & scheduled & (~no_show)

    first_visit_completed = scheduled & (~no_show) & (~cancelled)
    df["days_to_first_visit"] = np.where(scheduled, np.round(days_to_first_visit, 1), np.nan)
    df["no_show"] = no_show
    df["cancelled"] = cancelled
    df["first_visit_completed"] = first_visit_completed

    # ---------------- STAGE 3: Treatment Recommended / Started ----------------
    treatment_recommended = first_visit_completed & (rng.random(n_patients) < _sigmoid(2.2 + quality_score * 0.5))
    days_to_treatment = rng.gamma(shape=2.0, scale=3.2, size=n_patients) + np.where(df["clinic_b_shock"], rng.gamma(1.5, 2.0, n_patients), 0.0)
    days_to_treatment = np.clip(days_to_treatment, 0.5, 60)

    p_treatment_started = _sigmoid(1.7 + quality_score * 1.0 - (days_to_treatment - 7) * 0.05)
    treatment_started = treatment_recommended & (rng.random(n_patients) < p_treatment_started)

    df["treatment_recommended"] = treatment_recommended
    df["treatment_started"] = treatment_started
    df["days_to_treatment"] = np.where(treatment_recommended, np.round(days_to_treatment, 1), np.nan)

    # ---------------- STAGE 4: Follow-up Completed ----------------
    days_to_followup = rng.gamma(shape=2.3, scale=6.5, size=n_patients)
    days_to_followup = np.clip(days_to_followup, 1, 90)
    p_followup = _sigmoid(1.3 + quality_score * 0.9 - (days_to_treatment - 7) * 0.04)
    followup_completed = treatment_started & (rng.random(n_patients) < p_followup)

    df["days_to_followup"] = np.where(treatment_started, np.round(days_to_followup, 1), np.nan)
    df["followup_completed"] = followup_completed

    # ---------------- STAGE 5 & 6: 30-day engagement / 90-day retention ----------------
    p_engagement_30 = _sigmoid(
        0.9 + quality_score * 1.2 - (days_to_treatment.clip(0, 40) - 7) * 0.045 - week4_soft_penalty
    )
    engagement_30 = followup_completed & (rng.random(n_patients) < p_engagement_30)

    p_retained_90 = _sigmoid(0.5 + quality_score * 1.35 - (days_to_treatment.clip(0, 40) - 7) * 0.04)
    retained_90 = engagement_30 & (rng.random(n_patients) < p_retained_90)

    # days_active_30 / days_active_90: rough engagement-intensity proxies
    days_active_30 = np.where(
        engagement_30,
        np.clip(rng.normal(12 + quality_score * 4, 4, n_patients), 1, 30),
        np.where(followup_completed, np.clip(rng.normal(4, 2, n_patients), 0, 12), 0)
    )
    days_active_90 = np.where(
        retained_90,
        np.clip(rng.normal(28 + quality_score * 8, 10, n_patients), 1, 90),
        np.where(engagement_30, np.clip(rng.normal(10, 5, n_patients), 0, 30), 0)
    )

    df["engagement_30"] = engagement_30
    df["retained_90"] = retained_90
    df["days_active_30"] = np.round(days_active_30, 1)
    df["days_active_90"] = np.round(days_active_90, 1)

    # Only keep referrals old enough that 90-day outcomes are observable
    df["outcome_observable_90"] = (end - df["referral_date"]).dt.days >= 90
    df["outcome_observable_30"] = (end - df["referral_date"]).dt.days >= 30

    # clean types
    bool_cols = [
        "appointment_scheduled", "first_visit_completed", "treatment_recommended",
        "treatment_started", "followup_completed", "engagement_30", "retained_90",
        "no_show", "cancelled", "clinic_b_shock", "outcome_observable_90", "outcome_observable_30"
    ]
    for c in bool_cols:
        df[c] = df[c].astype(bool)

    df["referral_date"] = df["referral_date"].dt.date

    return df.drop(columns=["clinic_b_shock"])


def load_or_generate(path: str = "data/synthetic_patients.csv", n_patients: int = 22000) -> pd.DataFrame:
    """Load the cached CSV if present, otherwise generate and save it."""
    import os
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["referral_date"])
        df["referral_date"] = df["referral_date"].dt.date
        return df
    df = generate_patients(n_patients=n_patients)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    data = generate_patients(22000)
    data.to_csv("data/synthetic_patients.csv", index=False)
    print(f"Generated {len(data)} synthetic patient journeys.")
    print(data.head())
