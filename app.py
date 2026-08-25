
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_generation import load_or_generate
import metrics as M
import analysis as A
import anomaly_detection as AD
import recommendations as R


st.set_page_config(
    page_title="MSK Patient Journey Intelligence",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main .block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1280px; }
    h1, h2, h3 { font-family: 'Helvetica Neue', Arial, sans-serif; letter-spacing: -0.01em; }
    h1 { font-weight: 700; color: #1a2b4c; }
    h2 { font-weight: 600; color: #1a2b4c; margin-top: 0.4rem; }
    .demo-badge {
        display: inline-block; background: #fff3cd; color: #7a5b00; border: 1px solid #f0d38a;
        padding: 3px 12px; border-radius: 14px; font-size: 0.75rem; font-weight: 600;
        letter-spacing: 0.03em; margin-bottom: 0.6rem;
    }
    .subtitle-text { color: #5b6b85; font-size: 1.02rem; margin-bottom: 1.1rem; }
    /* KPI metric cards */
div[data-testid="stMetric"] {
    background: #f7f9fc;
    border: 1px solid #e6eaf1;
    border-radius: 10px;
    padding: 0.9rem 1rem 0.6rem 1rem;
}

/* KPI label */
div[data-testid="stMetricLabel"] {
    font-size: 0.82rem;
    color: #5b6b85 !important;
    font-weight: 600;
}

/* KPI NUMBER — this fixes the white text */
div[data-testid="stMetricValue"] {
    color: #1a2b4c !important;
}

/* Sometimes Streamlit puts the number inside another div */
div[data-testid="stMetricValue"] > div {
    color: #1a2b4c !important;
}

/* KPI delta */
div[data-testid="stMetricDelta"] {
    color: #5b6b85 !important;
}
    .insight-box {
    background: #eef4ff;
    color: #1a2b4c !important;
    border-left: 4px solid #3a6cf0;
    border-radius: 6px;
    padding: 0.85rem 1.1rem;
    margin: 0.6rem 0 1rem 0;
    font-size: 0.95rem;
}

    .insight-box * {
        color: #1a2b4c !important;
}
    .anomaly-box {
    background: #fff5f2;
    color: #1a2b4c !important;
    border-left: 4px solid #e0562f;
    border-radius: 6px;
    padding: 0.85rem 1.1rem;
    margin: 0.6rem 0 1rem 0;
    font-size: 0.95rem;
}

.anomaly-box * {
    color: #1a2b4c !important;
}
    .rec-card {
    background: #ffffff;
    color: #1a2b4c !important;
    border: 1px solid #e6eaf1;
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 1.1rem;
}

.rec-card * {
    color: #1a2b4c !important;
}

.rec-card .rec-label {
    color: #3a6cf0 !important;
}

.rec-card p {
    color: #1a2b4c !important;
}

.rec-card h3 {
    color: #1a2b4c !important;
}
    .rec-label { font-size: 0.72rem; font-weight: 700; color: #3a6cf0; letter-spacing: 0.04em; text-transform: uppercase; }
    .footer-disclaimer {
        margin-top: 2.5rem; padding: 1rem 1.2rem; background: #f7f9fc; border-radius: 8px;
        font-size: 0.82rem; color: #5b6b85; border: 1px solid #e6eaf1;
    }
    section[data-testid="stSidebar"] { background: #101a30; }
    section[data-testid="stSidebar"] * { color: #e8edf7 !important; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

CHART_TEMPLATE = "plotly_white"
COLOR_SEQUENCE = ["#3a6cf0", "#e0562f", "#1a9e7a", "#8a5cf0", "#e0b02f", "#5b6b85"]



# Data loading

@st.cache_data(show_spinner="Loading synthetic patient journeys...")
def get_data():
    return load_or_generate(path=os.path.join(os.path.dirname(__file__), "data", "synthetic_patients.csv"))


df_full = get_data()
df_full["referral_date"] = pd.to_datetime(df_full["referral_date"])

# Sidebar: navigation + global filters
st.sidebar.markdown("## 🦴 MSK Patient Journey\n### Intelligence")
st.sidebar.markdown(
    '<span style="background:#2a3a5c;color:#cfe0ff;padding:2px 10px;border-radius:10px;'
    'font-size:0.72rem;font-weight:600;">DEMO / SYNTHETIC DATA</span>',
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

PAGES = [
    "Executive Overview",
    "Patient Funnel",
    "Cohorts & Retention",
    "Operational Analysis",
    "Anomalies & Root Cause",
    "Recommendations",
    "Methodology",
]
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown("#### Filters")

min_date, max_date = df_full["referral_date"].min().date(), df_full["referral_date"].max().date()
date_range = st.sidebar.date_input("Referral Date Range", value=(min_date, max_date),
                                    min_value=min_date, max_value=max_date)

clinic_opts = ["All Clinics"] + sorted(df_full["clinic"].unique().tolist())
clinic_sel = st.sidebar.selectbox("Clinic", clinic_opts)

provider_opts = ["All Providers"] + sorted(df_full["provider"].unique().tolist())
provider_sel = st.sidebar.selectbox("Provider", provider_opts)

referral_opts = ["All Sources"] + sorted(df_full["referral_source"].unique().tolist())
referral_sel = st.sidebar.selectbox("Referral Source", referral_opts)

condition_opts = ["All Conditions"] + sorted(df_full["condition"].unique().tolist())
condition_sel = st.sidebar.selectbox("MSK Condition", condition_opts)

insurance_opts = ["All Insurance Types"] + sorted(df_full["insurance_type"].unique().tolist())
insurance_sel = st.sidebar.selectbox("Insurance Type", insurance_opts)

age_opts = ["All Ages"] + sorted(df_full["age_group"].unique().tolist())
age_sel = st.sidebar.selectbox("Patient Age Group", age_opts)

if st.sidebar.button("↺ Reset Filters"):
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption(
    "Portfolio prototype. All data is synthetically generated. No real patient, "
    "provider or clinic data is used or represented."
)


def apply_filters(data: pd.DataFrame) -> pd.DataFrame:
    d = data.copy()
    if isinstance(date_range, tuple) and len(date_range) == 2:
        d = d[(d["referral_date"].dt.date >= date_range[0]) & (d["referral_date"].dt.date <= date_range[1])]
    if clinic_sel != "All Clinics":
        d = d[d["clinic"] == clinic_sel]
    if provider_sel != "All Providers":
        d = d[d["provider"] == provider_sel]
    if referral_sel != "All Sources":
        d = d[d["referral_source"] == referral_sel]
    if condition_sel != "All Conditions":
        d = d[d["condition"] == condition_sel]
    if insurance_sel != "All Insurance Types":
        d = d[d["insurance_type"] == insurance_sel]
    if age_sel != "All Ages":
        d = d[d["age_group"] == age_sel]
    return d


df = apply_filters(df_full)

# previous-period comparison window (same length, immediately prior)
if isinstance(date_range, tuple) and len(date_range) == 2:
    period_len = (date_range[1] - date_range[0]).days
    prev_start = date_range[0] - pd.Timedelta(days=period_len + 1)
    prev_end = date_range[0] - pd.Timedelta(days=1)
    prev_df = df_full.copy()
    prev_df = prev_df[(prev_df["referral_date"].dt.date >= prev_start) & (prev_df["referral_date"].dt.date <= prev_end)]
    if clinic_sel != "All Clinics":
        prev_df = prev_df[prev_df["clinic"] == clinic_sel]
    if provider_sel != "All Providers":
        prev_df = prev_df[prev_df["provider"] == provider_sel]
    if referral_sel != "All Sources":
        prev_df = prev_df[prev_df["referral_source"] == referral_sel]
    if condition_sel != "All Conditions":
        prev_df = prev_df[prev_df["condition"] == condition_sel]
    if insurance_sel != "All Insurance Types":
        prev_df = prev_df[prev_df["insurance_type"] == insurance_sel]
    if age_sel != "All Ages":
        prev_df = prev_df[prev_df["age_group"] == age_sel]
else:
    prev_df = None

if len(df) == 0:
    st.warning("No patients match the current filter selection. Try widening your filters.")
    st.stop()


def footer():
    st.markdown(
        '<div class="footer-disclaimer"><b>Portfolio Prototype</b><br>'
        "This application uses entirely synthetic data and is intended solely to demonstrate "
        "product and operational analytics. It is not a clinical decision-support system and "
        "does not use or represent real patient, provider, clinic, or Flagler Health data.</div>",
        unsafe_allow_html=True,
    )


def format_kpi_value(v, fmt):
    if v is None:
        return "—"
    return f"{v:,.1f}%" if fmt == "pct" else f"{v:,.0f}"


def format_delta(pct_change):
    if pct_change is None:
        return None
    return f"{pct_change:+.1f}% vs prior period"


def add_trend_line(fig, x, y):
    """Add a simple linear best-fit line to a Plotly figure without requiring statsmodels."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < 2:
        return
    slope, intercept = np.polyfit(x[mask], y[mask], 1)
    x_line = np.linspace(x[mask].min(), x[mask].max(), 50)
    y_line = slope * x_line + intercept
    fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", name="Trend",
                              line=dict(color="#5b6b85", dash="dot")))



# PAGE: Executive Overview
if page == "Executive Overview":
    st.markdown('<span class="demo-badge">DEMO / SYNTHETIC DATA</span>', unsafe_allow_html=True)
    st.title("Executive Overview")
    st.markdown(
        '<div class="subtitle-text">Understand how patients move through the care journey and '
        "identify opportunities to improve engagement and retention.</div>",
        unsafe_allow_html=True,
    )

    kpis = M.kpi_summary(df, prev_df)
    cols = st.columns(4)
    print(kpis)
    for i, kpi in enumerate(kpis):
        with cols[i % 4]:
            st.metric(
                kpi["name"],
                format_kpi_value(kpi["value"], kpi["format"]),
                delta=format_delta(kpi["pct_change"]),
                help=kpi["definition"],
                
            )

    st.markdown("### Patient Lifecycle Funnel")
    funnel_df = M.funnel_counts(df)

    fig = go.Figure(go.Funnel(
        y=funnel_df["stage"],
        x=funnel_df["count"],
        textinfo="value+percent initial",
        marker={"color": COLOR_SEQUENCE[0]},
    ))
    fig.update_layout(template=CHART_TEMPLATE, height=430, margin=dict(t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)

    drop = M.largest_drop(funnel_df)
    st.markdown(
        f'<div class="insight-box"><b>Largest Funnel Drop</b><br>'
        f"{drop['from_stage']} → {drop['to_stage']}<br>"
        f"{drop['drop_off_pct']:.1f}% of eligible patients are lost between these stages.</div>",
        unsafe_allow_html=True,
    )

    with st.expander("View full funnel table"):
        st.dataframe(funnel_df, use_container_width=True, hide_index=True)

    footer()


# PAGE: Patient Funnel
elif page == "Patient Funnel":
    st.title("Patient Funnel")
    st.markdown('<div class="subtitle-text">Where are patients falling out of the care journey?</div>',
                unsafe_allow_html=True)

    funnel_df = M.funnel_counts(df)
    prev_funnel_df = M.funnel_counts(prev_df) if prev_df is not None and len(prev_df) > 0 else None

    display_df = funnel_df.copy()
    if prev_funnel_df is not None:
        display_df["prior_period_count"] = prev_funnel_df["count"]
        display_df["change_vs_prior"] = display_df["count"] - display_df["prior_period_count"]

    st.markdown("### Stage-by-Stage Detail")
    st.dataframe(
        display_df.rename(columns={
            "stage": "Stage", "count": "Patients", "conversion_from_prev_pct": "Conv. from Prev (%)",
            "conversion_from_start_pct": "Conv. from Referral (%)", "drop_off_pct": "Drop-off (%)",
            "prior_period_count": "Prior Period Patients", "change_vs_prior": "Change vs Prior",
        }),
        use_container_width=True, hide_index=True,
    )

    st.markdown("### Segment the Funnel")
    seg_choice = st.selectbox(
        "Segment by", ["Clinic", "Provider", "Referral Source", "Condition", "Insurance Type"], index=0
    )
    seg_col_map = {
        "Clinic": "clinic", "Provider": "provider", "Referral Source": "referral_source",
        "Condition": "condition", "Insurance Type": "insurance_type",
    }
    STAGE_LABELS = {
        "appointment_scheduled": "Appointment Scheduled", "first_visit_completed": "First Visit Completed",
        "treatment_started": "Treatment Started", "followup_completed": "Follow-up Completed",
        "engagement_30": "30-Day Engagement", "retained_90": "90-Day Retention",
    }
    stage_choice = st.selectbox(
        "Funnel stage", ["appointment_scheduled", "first_visit_completed", "treatment_started",
                          "followup_completed", "engagement_30", "retained_90"],
        format_func=lambda c: STAGE_LABELS[c],
        index=2,
    )
    stage_label = {
        "appointment_scheduled": "Appointment Scheduled", "first_visit_completed": "First Visit Completed",
        "treatment_started": "Treatment Start Rate", "followup_completed": "Follow-up Completion",
        "engagement_30": "30-Day Engagement", "retained_90": "90-Day Retention",
    }[stage_choice]

    seg_col = seg_col_map[seg_choice]
    seg_df = M.segment_funnel(df, seg_col, stage_choice)

    fig2 = px.bar(
        seg_df, x="conversion_rate_pct", y=seg_col, orientation="h",
        title=f"{stage_label} by {seg_choice}", labels={"conversion_rate_pct": f"{stage_label} (%)", seg_col: seg_choice},
        color_discrete_sequence=[COLOR_SEQUENCE[0]], text="conversion_rate_pct",
    )
    median_rate = seg_df.attrs.get("median_rate", seg_df["conversion_rate_pct"].median())
    fig2.add_vline(x=median_rate, line_dash="dash", line_color="#5b6b85",
                    annotation_text=f"Network median: {median_rate:.1f}%")
    fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig2.update_layout(template=CHART_TEMPLATE, height=max(320, 42 * len(seg_df)), margin=dict(t=50, b=10))
    st.plotly_chart(fig2, use_container_width=True)

    insight = M.generate_segment_insight(seg_df, seg_col, stage_label)
    if insight:
        st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
    else:
        st.caption("No segment currently shows a material gap versus the network median for this stage.")

    with st.expander("View segment table"):
        display_seg = seg_df.rename(columns={
            seg_col: seg_choice, "patients": "Patients", "conversion_rate_pct": f"{stage_label} (%)",
            "vs_median_pct_pts": "vs Median (pct pts)",
        })[[seg_choice, "Patients", f"{stage_label} (%)", "vs Median (pct pts)"]]
        st.dataframe(display_seg, use_container_width=True, hide_index=True)

    footer()

# PAGE: Cohorts & Retention
elif page == "Cohorts & Retention":
    st.title("Cohorts & Retention")
    st.markdown('<div class="subtitle-text">Analyze patient behavior over time, grouped by cohort.</div>',
                unsafe_allow_html=True)

    cohort_dim = st.selectbox("Cohort grouping", ["First Visit Month", "Clinic", "Condition", "Referral Source"])
    cohort_dim_col_map = {"Clinic": "clinic", "Condition": "condition", "Referral Source": "referral_source"}
    cohort_col = cohort_dim_col_map.get(cohort_dim)  # None => group by first-visit month

    heatmap_df = A.retention_heatmap_data(df, cohort_col=cohort_col)

    if heatmap_df.empty:
        st.info("Not enough patients in the current filter selection to build a reliable cohort heatmap.")
    else:
        pivot = heatmap_df.pivot(index="cohort", columns="week", values="retention_rate")
        week_order = [f"Week {w}" for w in (1, 2, 4, 8, 12)]
        pivot = pivot.reindex(columns=[w for w in week_order if w in pivot.columns])
        y_axis_label = "First-Visit Cohort" if cohort_col is None else cohort_dim

        fig3 = px.imshow(
            pivot, text_auto=".0f", color_continuous_scale="Blues", aspect="auto",
            labels=dict(x="Weeks Since First Visit", y=y_axis_label, color="Retention (%)"),
        )
        fig3.update_layout(template=CHART_TEMPLATE, height=max(360, 32 * len(pivot)), margin=dict(t=30, b=10))
        st.plotly_chart(fig3, use_container_width=True)

        with st.expander("What does 'retained' mean here?"):
            st.write(
                "A patient is considered retained at a given week if they show a qualifying care "
                "interaction (an active-engagement day) at or beyond that point in their journey, "
                "based on their observed 30-day and 90-day activity intensity. Cohorts with fewer "
                "than 15 patients in the current filter are excluded to avoid unstable estimates."
            )

        if cohort_col is None:
            trend_insight = A.cohort_week4_trend_insight(heatmap_df)
            if trend_insight:
                st.markdown(f'<div class="insight-box">{trend_insight}</div>', unsafe_allow_html=True)

    st.markdown("### Cohort Sizes")
    cohort_sizes = df.copy()
    cohort_sizes["cohort"] = A.build_cohort_column(cohort_sizes)
    size_tbl = cohort_sizes.groupby("cohort").size().reset_index(name="patients")
    fig4 = px.bar(size_tbl, x="cohort", y="patients", color_discrete_sequence=[COLOR_SEQUENCE[2]],
                  labels={"cohort": "First-Visit Cohort (Month)", "patients": "Patients"})
    fig4.update_layout(template=CHART_TEMPLATE, height=300, margin=dict(t=20, b=10))
    st.plotly_chart(fig4, use_container_width=True)

    footer()

# PAGE: Operational Analysis
elif page == "Operational Analysis":
    st.title("Operational Analysis")
    st.markdown('<div class="subtitle-text">Investigating operational factors associated with patient drop-off.</div>',
                unsafe_allow_html=True)

    op = A.operational_summary(df)
    cols = st.columns(5)
    labels_vals = [
        ("Avg. Days to First Visit", op["avg_days_to_first_visit"]),
        ("Avg. Days to Treatment", op["avg_days_to_treatment"]),
        ("Avg. Days to Follow-up", op["avg_days_to_followup"]),
        ("No-show Rate", f'{op["no_show_rate"]}%'),
        ("Cancellation Rate", f'{op["cancellation_rate"]}%'),
    ]
    for i, (label, val) in enumerate(labels_vals):
        with cols[i]:
            st.metric(label, val)

    st.markdown("### Days to Treatment Start vs. 30-Day Engagement")
    bucket_df = A.bucketed_retention_by_delay(df, "days_to_treatment", "engagement_30")
    if not bucket_df.empty:
        fig5 = px.scatter(
            bucket_df, x="avg_delay", y="outcome_rate_pct", size="patients",
            labels={"avg_delay": "Avg. Days to Treatment Start", "outcome_rate_pct": "30-Day Engagement (%)"},
            color_discrete_sequence=[COLOR_SEQUENCE[0]],
        )
        fig5.update_traces(marker=dict(size=14))
        add_trend_line(fig5, bucket_df["avg_delay"], bucket_df["outcome_rate_pct"])
        fig5.update_layout(template=CHART_TEMPLATE, height=420, margin=dict(t=20, b=10))
        st.plotly_chart(fig5, use_container_width=True)
        st.caption(
            "Patients experiencing longer treatment-start delays show lower engagement in this synthetic "
            "dataset. This is an observed association, not a claim of causation."
        )
    else:
        st.info("Not enough data in the current filter selection to build this chart.")

    st.markdown("### Days to Treatment Start vs. 90-Day Retention")
    bucket_df2 = A.bucketed_retention_by_delay(df, "days_to_treatment", "retained_90")
    if not bucket_df2.empty:
        fig6 = px.scatter(
            bucket_df2, x="avg_delay", y="outcome_rate_pct", size="patients",
            labels={"avg_delay": "Avg. Days to Treatment Start", "outcome_rate_pct": "90-Day Retention (%)"},
            color_discrete_sequence=[COLOR_SEQUENCE[1]],
        )
        fig6.update_traces(marker=dict(size=14))
        add_trend_line(fig6, bucket_df2["avg_delay"], bucket_df2["outcome_rate_pct"])
        fig6.update_layout(template=CHART_TEMPLATE, height=420, margin=dict(t=20, b=10))
        st.plotly_chart(fig6, use_container_width=True)
        st.caption(
            "Patients experiencing longer treatment-start delays show lower retention in this synthetic "
            "dataset. This is an observed association, not a claim of causation."
        )

    st.markdown("### Time-to-First-Appointment Distribution")
    fig7 = px.histogram(df.dropna(subset=["days_to_first_visit"]), x="days_to_first_visit", nbins=40,
                         color_discrete_sequence=[COLOR_SEQUENCE[3]],
                         labels={"days_to_first_visit": "Days from Referral to First Appointment"})
    fig7.update_layout(template=CHART_TEMPLATE, height=340, margin=dict(t=20, b=10))
    st.plotly_chart(fig7, use_container_width=True)

    footer()

# PAGE: Anomalies & Root Cause
elif page == "Anomalies & Root Cause":
    st.title("Anomalies & Root Cause")
    st.markdown('<div class="subtitle-text">Automated statistical monitoring of operational and funnel metrics.</div>',
                unsafe_allow_html=True)

    with st.expander("Detection method"):
        st.write(
            "Each metric is aggregated weekly. A rolling mean and rolling standard deviation "
            "(6-week window) form a dynamic baseline. A week is flagged as anomalous when its "
            "z-score relative to that baseline exceeds a control-limit threshold (default ±2.0 "
            "standard deviations)."
        )

    z_threshold = st.slider("Sensitivity (z-score threshold)", 1.5, 3.0, 2.0, 0.1)

    clinics_list = sorted(df_full["clinic"].unique().tolist())
    with st.spinner("Scanning metrics for anomalies..."):
        anomalies = AD.scan_all_metrics(df_full, clinics_list, window=6, z_threshold=z_threshold)

    if anomalies.empty:
        st.success("No anomalies detected at the current sensitivity across monitored metrics.")
    else:
        st.markdown(f"### {len(anomalies)} Anomal{'y' if len(anomalies)==1 else 'ies'} Detected")
        options = []
        for _, row in anomalies.iterrows():
            options.append(
                f"{row['metric_label']} — {row['clinic']} — week of {pd.Timestamp(row['week']).strftime('%Y-%m-%d')} "
                f"(z={row['z_score']})"
            )
        sel_idx = st.selectbox("Select an anomaly to investigate", range(len(options)), format_func=lambda i: options[i])
        chosen = anomalies.iloc[sel_idx]

        pct_txt = f"{abs(chosen['pct_vs_baseline']):.1f}%" if chosen["pct_vs_baseline"] is not None else "a notable margin"
        st.markdown(
            f'<div class="anomaly-box">⚠️ <b>{chosen["metric_label"]} anomaly</b><br>'
            f"{chosen['clinic']} {chosen['metric_label'].lower()} was {pct_txt} {chosen['direction']} its recent "
            f"baseline in the week of {pd.Timestamp(chosen['week']).strftime('%B %d, %Y')} "
            f"(value {chosen['value']}, baseline {chosen['baseline']}, z-score {chosen['z_score']}).</div>",
            unsafe_allow_html=True,
        )

        st.markdown("### Trend")
        series = AD.weekly_metric_series(df_full, chosen["metric"], clinic=chosen["clinic"])
        flagged = AD.detect_anomalies(series, window=6, z_threshold=z_threshold)
        fig8 = go.Figure()
        fig8.add_trace(go.Scatter(x=flagged["week"], y=flagged["value"], mode="lines+markers",
                                   name="Observed", line=dict(color=COLOR_SEQUENCE[0])))
        fig8.add_trace(go.Scatter(x=flagged["week"], y=flagged["rolling_mean"], mode="lines",
                                   name="Rolling Baseline", line=dict(color="#5b6b85", dash="dash")))
        anomaly_pts = flagged[flagged["is_anomaly"]]
        fig8.add_trace(go.Scatter(x=anomaly_pts["week"], y=anomaly_pts["value"], mode="markers",
                                   name="Anomaly", marker=dict(color="#e0562f", size=11, symbol="diamond")))
        fig8.update_layout(template=CHART_TEMPLATE, height=380, margin=dict(t=20, b=10),
                            yaxis_title=chosen["metric_label"])
        st.plotly_chart(fig8, use_container_width=True)

        st.markdown("## Root Cause Investigation")
        rc = AD.root_cause_breakdown(df_full, chosen["clinic"], chosen["week"], chosen["metric"])

        if not rc["comparisons"]:
            st.info("Not enough patients in the baseline and anomaly windows to run a root-cause breakdown.")
        else:
            st.markdown("### What Changed?")
            comp_df = pd.DataFrame(rc["comparisons"]).rename(
                columns={"factor": "Factor", "baseline": "Baseline Period", "anomaly_period": "Anomaly Period"}
            )
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            st.markdown("### Most Associated Factors")
            st.caption("Ranked by statistical evidence. These are associations, not proven causes.")
            if rc["factors"]:
                for i, f in enumerate(rc["factors"][:5], start=1):
                    if f["p_value"] is None:
                        p_txt = "descriptive shift"
                    elif f["p_value"] < 0.001:
                        p_txt = "p < 0.001"
                    else:
                        p_txt = f"p = {f['p_value']}"
                    st.markdown(f"**{i}. {f['factor']}** — {f['direction']} in the anomaly period ({p_txt})")
            else:
                st.write("No factors showed a measurable shift between the baseline and anomaly periods.")

        st.markdown("### Investigate By")
        breakdown_dim = st.selectbox(
            "Breakdown dimension", ["Provider", "Condition", "Referral Source", "Insurance Type"], key="rc_breakdown"
        )
        dim_col_map = {"Provider": "provider", "Condition": "condition", "Referral Source": "referral_source",
                       "Insurance Type": "insurance_type"}
        anomaly_period_df = rc["anomaly"]
        if len(anomaly_period_df) > 0:
            dcol = dim_col_map[breakdown_dim]
            numer_col, denom_col = AD.METRIC_DEFINITIONS.get(chosen["metric"], ("treatment_started", "referral"))
            g = anomaly_period_df.groupby(dcol).agg(
                patients=("patient_id", "count"), rate=(numer_col, "mean")
            ).reset_index()
            g["rate_pct"] = (g["rate"] * 100).round(1)
            fig9 = px.bar(g.sort_values("rate_pct"), x="rate_pct", y=dcol, orientation="h",
                          color_discrete_sequence=[COLOR_SEQUENCE[1]],
                          labels={"rate_pct": f"{chosen['metric_label']} (%)", dcol: breakdown_dim})
            fig9.update_layout(template=CHART_TEMPLATE, height=max(300, 40 * len(g)), margin=dict(t=20, b=10))
            st.plotly_chart(fig9, use_container_width=True)

    footer()

# PAGE: Recommendations
elif page == "Recommendations":
    st.title("Recommendations")
    st.markdown('<div class="subtitle-text">Turning analysis into action.</div>', unsafe_allow_html=True)

    recs = R.build_recommendations(df)

    if not recs:
        st.info("Not enough signal in the current filter selection to generate recommendations. Try widening filters.")
    else:
        for rec in recs:
            sig_badge = "✓ Statistically significant (p < 0.05)" if rec.get("significant") else "Directional signal"
            st.markdown(
                f'<div class="rec-card">'
                f'<div class="rec-label">Recommendation</div>'
                f'<h3 style="margin-top:0.2rem;">{rec["title"]}</h3>'
                f'<p class="rec-sig">{sig_badge}</p>'
                f'<b>Finding</b><p>{rec["finding"]}</p>'
                f'<b>Evidence</b><p>{rec["evidence"]}</p>'
                f'<b>Operational Implication</b><p>{rec["implication"]}</p>'
                f'<b>Recommended Next Step</b><p>{rec["next_step"]}</p>'
                f'<b>Measurement Plan</b><p>{rec["measurement"]}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("## What Should We Test?")
        exp = R.experiment_design(recs)
        if exp:
            st.caption(exp["label"])
            st.markdown(f"**Hypothesis**  \n{exp['hypothesis']}")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Control**  \n{exp['control']}")
            with c2:
                st.markdown(f"**Treatment**  \n{exp['treatment']}")
            st.markdown(f"**Intervention**  \n{exp['intervention']}")
            st.markdown(f"**Primary Metric**  \n{exp['primary_metric']}")
            st.markdown("**Secondary Metrics**")
            for m in exp["secondary_metrics"]:
                st.markdown(f"- {m}")
            st.markdown("**Guardrails**")
            for g in exp["guardrails"]:
                st.markdown(f"- {g}")

    footer()

# PAGE: Methodology
elif page == "Methodology":
    st.title("Methodology")

    st.markdown("### Data")
    st.write(
        "All data in this application is synthetically generated (see `src/data_generation.py`). "
        "The generator produces over 20,000 simulated patient journeys with realistic, noisy "
        "relationships between operational factors (clinic, provider, referral source, scheduling "
        "delays) and downstream engagement outcomes. No real patient, provider, clinic, or "
        "organizational data is used anywhere in this project."
    )

    st.markdown("### Funnel Definitions")
    for stage, definition in M.STAGE_DEFINITIONS.items():
        st.markdown(f"**{stage}** — {definition}")

    st.markdown("### Retention")
    st.write(
        "A patient is considered retained at a given checkpoint (Week 1 through Week 12) if their "
        "observed engagement activity implies a qualifying care interaction at or beyond that point "
        "in their journey. Retention is estimated from 30-day and 90-day activity-intensity fields, "
        "scaled to each weekly checkpoint."
    )

    st.markdown("### Cohorts")
    st.write(
        "Cohorts are constructed by grouping patients according to their first-visit month. "
        "Cohorts with fewer than 15 patients under the active filter selection are excluded from "
        "heatmap visualizations to avoid unstable, low-sample estimates."
    )

    st.markdown("### Anomaly Detection")
    st.write(
        "Weekly time series are built for each monitored metric (treatment-start rate, follow-up "
        "rate, no-show rate, cancellation rate, appointment wait time, 30-day engagement, and "
        "90-day retention). A rolling mean and rolling standard deviation over a 6-week trailing "
        "window form a dynamic baseline; a week is flagged when its z-score relative to that "
        "baseline exceeds an adjustable control-limit threshold."
    )

    st.markdown("### Root Cause Analysis")
    st.write(
        "When an anomaly is selected, the application compares a baseline period against the "
        "anomaly period across candidate explanatory factors (scheduling delay, no-show/cancellation "
        "rate, referral mix, provider distribution) using two-proportion z-tests, Welch's t-tests, "
        "and chi-square tests of independence where appropriate. Factors are ranked by statistical "
        "evidence. **This analysis identifies associations and potential drivers, not proven "
        "causation.**"
    )

    st.markdown("### Experimentation")
    st.write(
        "The Recommendations page includes a hypothetical A/B experiment design derived from the "
        "top finding. A future controlled experiment — randomizing eligible patients between a "
        "control and treatment scheduling workflow — would be required to establish causal impact. "
        "No such experiment has been run; this is a design proposal only."
    )

    st.markdown("### Limitations")
    st.write(
        "This is a portfolio prototype built entirely on synthetic data. The relationships embedded "
        "in the generator are illustrative and do not represent, and cannot be used to infer, actual "
        "patient behavior at Flagler Health or any real organization. Findings should be read as a "
        "demonstration of analytical approach, not as operational fact."
    )

    footer()
