import os
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# Load API key from .env
load_dotenv()

# Page title
st.set_page_config(page_title="KPI Tracker", layout="wide")
st.title("KPI Tracker")

# Load data
df = pd.read_csv("sample_kpi_data.csv")

# Convert date column
df["date"] = pd.to_datetime(df["date"])

# Show data
st.subheader("Raw KPI Data")
st.write(df)

# Show basic metrics
st.subheader("Quick Metrics")

total_visits = df["visits"].sum()
total_signups = df["signups"].sum()
total_revenue = df["revenue"].sum()

col1, col2, col3 = st.columns(3)
col1.metric("Total Visits", f"{total_visits:,}")
col2.metric("Total Signups", f"{total_signups:,}")
col3.metric("Total Revenue", f"${total_revenue:,}")

# Revenue chart
st.subheader("Revenue Over Time")
fig = px.line(df, x="date", y="revenue", color="channel", markers=True)
st.plotly_chart(fig, use_container_width=True)

# Create a simple text summary for AI
latest_revenue = df["revenue"].iloc[-1]
first_revenue = df["revenue"].iloc[0]
revenue_change = latest_revenue - first_revenue

summary_text = f"""
Here is the KPI summary:
- Total visits: {total_visits}
- Total signups: {total_signups}
- Total revenue: {total_revenue}
- Revenue on first day: {first_revenue}
- Revenue on last day: {latest_revenue}
- Revenue change: {revenue_change}

Channel level revenue:
{df.groupby('channel')['revenue'].sum().to_string()}
"""

st.subheader("Summary Sent to AI")
st.text(summary_text)

# AI summary button
st.subheader("AI Business Summary")

if st.button("Generate AI Summary"):
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        st.error("OPENAI_API_KEY not found in .env file")
    else:
        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a product analytics assistant. Summarize KPI performance in simple business language. Do not invent facts."
                },
                {
                    "role": "user",
                    "content": summary_text
                }
            ],
            temperature=0.3,
        )

        answer = response.choices[0].message.content
        st.write(answer)