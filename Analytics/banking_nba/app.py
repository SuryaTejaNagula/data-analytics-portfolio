# app.py — COMPLETE STREAMLIT DASHBOARD
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Banking Lead Intelligence",
    page_icon="🏦",
    layout="wide"
)

# ── LOAD DATA ──────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    # return pd.read_csv("scored_leads.csv")
    import os
    base = os.path.dirname(os.path.abspath(__file__))
    return pd.read_csv(os.path.join(base, "scored_leads.csv"))

df = load_data()

# ── HEADER ────────────────────────────────────────────────────────────
st.title("🏦 Retail Banking — Lead Intelligence Dashboard")
st.markdown("*Next-Best-Action · Propensity Scoring · Weekly Calling Priority*")
st.divider()

# ── KPI SUMMARY CARDS ─────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
tier_a = df[df['lead_tier'] == 'A - High']
baseline = df['target'].mean()

c1.metric("Total Customers Scored",  f"{len(df):,}")
c2.metric("Tier A Priority Leads",   f"{len(tier_a):,}",
          delta=f"{len(tier_a)/len(df):.1%} of book")
c3.metric("Tier A Conversion Rate",  f"{tier_a['target'].mean():.1%}",
          delta=f"+{tier_a['target'].mean()-baseline:.1%} vs baseline")
c4.metric("Baseline Conversion Rate", f"{baseline:.1%}")

st.divider()

# ── SIDEBAR FILTERS ────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    tier_filter = st.multiselect(
        "Lead Tier",
        ['A - High', 'B - Medium', 'C - Low'],
        default=['A - High']
    )
    min_score = st.slider("Min Propensity Score", 0.0, 1.0, 0.5, 0.05)
    prev_contact_only = st.checkbox("Previously contacted only", False)
    prev_success_only  = st.checkbox("Previous success only",    False)
    top_n = st.slider("Show top N leads", 10, 500, 100, 10)

# Apply filters
filtered = df[df['lead_tier'].isin(tier_filter)]
filtered = filtered[filtered['propensity_score'] >= min_score]
if prev_contact_only:
    filtered = filtered[filtered['had_previous_contact'] == 1]
if prev_success_only:
    filtered = filtered[filtered['previous_success'] == 1]
filtered = filtered.sort_values('propensity_score', ascending=False).head(top_n)

# ── TABS ───────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Lead List", "Score Distribution", "Segment Insights"])

with tab1:
    st.subheader(f"Priority Lead List — {len(filtered)} customers shown")
    show_cols = ['propensity_score', 'lead_tier', 'age',
                 'is_cellular', 'had_previous_contact', 'previous_success']
    show_cols = [c for c in show_cols if c in filtered.columns]

    st.dataframe(
        filtered[show_cols].style.background_gradient(
            subset=['propensity_score'], cmap='Greens'),
        use_container_width=True,
        height=420
    )
    csv_data = filtered[show_cols].to_csv(index=False)
    st.download_button(
        label="Download Lead List as CSV",
        data=csv_data,
        file_name="weekly_lead_list.csv",
        mime="text/csv"
    )

with tab2:
    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.histogram(
            df, x='propensity_score', color='lead_tier',
            title='Propensity Score Distribution by Tier',
            color_discrete_map={
                'A - High':'#1a5c38','B - Medium':'#EF9F27','C - Low':'#aaaaaa'},
            labels={'propensity_score':'Score','count':'Customers'}
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        tier_conv = (df.groupby('lead_tier')['target']
                     .agg(['mean','count']).reset_index())
        tier_conv.columns = ['Tier','Conv Rate','Count']
        tier_conv['Conv Rate %'] = (tier_conv['Conv Rate'] * 100).round(1)

        fig2 = px.bar(
            tier_conv, x='Tier', y='Conv Rate %',
            title='Conversion Rate by Lead Tier',
            color='Tier',
            color_discrete_map={
                'A - High':'#1a5c38','B - Medium':'#EF9F27','C - Low':'#aaaaaa'},
            text='Conv Rate %'
        )
        fig2.update_traces(texttemplate='%{text}%', textposition='outside')
        st.plotly_chart(fig2, use_container_width=True)

with tab3:
    col1, col2 = st.columns(2)

    with col1:
        if 'job_category' in df.columns:
            jc = (df.groupby('job_category')['target']
                  .mean().sort_values(ascending=False) * 100)
            fig3 = px.bar(
                x=jc.values.round(1), y=jc.index,
                orientation='h',
                title='Conversion Rate by Job Category',
                labels={'x':'Conv Rate %','y':''},
                color=jc.values,
                color_continuous_scale='Greens'
            )
            st.plotly_chart(fig3, use_container_width=True)

    with col2:
        prev = df.groupby('previous_success')['target'].agg(['mean','count']).reset_index()
        prev['Group'] = prev['previous_success'].map({0:'No prev success',1:'Prev success'})
        prev['Conv Rate %'] = (prev['mean'] * 100).round(1)
        fig4 = px.bar(
            prev, x='Group', y='Conv Rate %',
            title='Previous Campaign Success as Signal',
            color='Conv Rate %', color_continuous_scale='Greens',
            text='Conv Rate %'
        )
        fig4.update_traces(texttemplate='%{text}%', textposition='outside')
        st.plotly_chart(fig4, use_container_width=True)

st.divider()
st.caption("Model: XGBoost + Isotonic Calibration  ·  Data: UCI Bank Marketing  ·  Project by Surya Nagula")