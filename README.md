# Data Analytics Portfolio
### Surya Teja Nagula

End-to-end data and AI projects spanning product analytics, machine learning, and ai automation.

---

## Table of Contents

- [About This Repository](#about-this-repository)
- [Project Index](#project-index)
  - [AI Projects](#ai-projects)
  - [Analytics](#analytics)
- [Repository Structure](#repository-structure)
- [Tech Stack](#tech-stack)
- [About Me](#about-me)

---

## About This Repository

This repository is a living portfolio of applied data science and AI projects built to demonstrate end-to-end delivery across the full analytics stack — from raw data ingestion and machine learning to interactive dashboards and automated workflows.

Projects span two domains:

- **AI Projects** — LLM-powered tools, intelligent assistants, and workflow automation
- **Analytics** — Decision analytics, propensity modeling, lead scoring, and business intelligence

New projects are added continuously. Each project folder contains its own `README.md` with full documentation, setup instructions, and results.

---

## Project Index

### AI Projects

| # | Project | Description | Key Tools | Status |
|---|---------|-------------|-----------|--------|
| 1 | [AI KPI Assistant](#1-ai-kpi-assistant) | Conversational BI tool — ask questions about business metrics in plain English and get instant chart outputs | Python · Streamlit · OpenAI API | Active |

---

### Analytics

| # | Project | Description | Key Tools | Status |
|---|---------|-------------|-----------|--------|
| 1 | [Banking Next-Best-Action Lead Scoring](#1-banking-next-best-action-lead-scoring) | End-to-end propensity scoring engine for retail banking — scores 41,000+ customers, generates a weekly Tier A lead list with 6.4x conversion lift | Python · XGBoost · Streamlit · Plotly | Active |

---

## Project Details

### AI Projects

---

#### 1. AI KPI Assistant

**Folder:** `AI_Projects/StreamlitApp/AI_Assistant_KPI_Tracker-Streamlit`

**What it does:**
A conversational business intelligence prototype that lets non-technical users query business metrics using plain English. Type a question, get a chart. No SQL, no dashboards to navigate — just a conversation with your data.

**Key features:**
- Natural language querying of KPI data via OpenAI API
- Auto-generated chart outputs from conversational responses
- Streamlit interface — runs in browser, no installation required for end users
- Designed for sales managers, operations leads, and business stakeholders

**Business value:**
Removes the analyst bottleneck for routine data questions. A sales manager can ask "show me revenue by region this quarter" and get an instant chart without filing a ticket.

**Tech stack:** Python · Streamlit · OpenAI API · Pandas · Plotly

**Docs:** See `AI_Projects/StreamlitApp/AI_Assistant_KPI_Tracker-Streamlit/README.md`

---

### Analytics

---

#### 1. Banking Next-Best-Action Lead Scoring

**Folder:** `Analytics/banking_nba`

**What it does:**
An end-to-end Next-Best-Action (NBA) lead scoring engine for retail banking. Ingests 41,188 historical customer records, trains an XGBoost propensity model, scores every customer, assigns lead tiers (A/B/C), and surfaces a ranked weekly calling list through a live Streamlit dashboard.

**Key results:**

| Metric | Value |
|--------|-------|
| Total customers scored | 41,188 |
| Model AUC score | 0.810 |
| Tier A conversion rate | 72.2% |
| Baseline conversion rate | 11.3% |
| Lift over baseline | **6.4x** |
| Top-200 list precision uplift | **8.0x** |

**Key features:**
- XGBoost classifier with isotonic probability calibration
- Automated lead tiering: Tier A (High), Tier B (Medium), Tier C (Low)
- Interactive Streamlit dashboard with filters, lead list, score distribution, and segment insights
- One-click CSV export of the weekly priority calling list
- Covers all banking product motions: acquisition, cross-sell, activation, balance build, retention

**Business value:**
Transforms a 11% baseline conversion rate into 72% for prioritized leads. A banker using the Tier A list converts 6x more customers per call — reducing cost-per-acquisition and improving campaign ROI without increasing headcount.

**Tech stack:** Python · XGBoost · Scikit-learn · Streamlit · Plotly · Pandas

**Docs:** See `Analytics/banking_nba/README.md`

---

## Repository Structure

```
data-analytics-portfolio/
|
|-- AI_Projects/
|   |-- StreamlitApp/
|       |-- AI_Assistant_KPI_Tracker-Streamlit/
|           |-- app.py
|           |-- README.md
|           |-- requirements.txt
|           |-- (other project files)
|
|-- Analytics/
|   |-- banking_nba/
|       |-- data_prep.py
|       |-- model.py
|       |-- app.py
|       |-- requirements.txt
|       |-- README.md
|       |-- (auto-generated data files)
|
|-- README.md                    ← You are here
```

Each project folder is self-contained with its own dependencies, setup instructions, and README.

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Languages | Python · SQL |
| Machine Learning | XGBoost · Scikit-learn · Pandas · NumPy |
| AI / LLM | OpenAI API · Anthropic Claude · Google Gemini · Prompt Engineering |
| AI Coding Assistants | Claude Code · Cursor · GitHub Copilot |
| Dashboards | Streamlit · Plotly · Tableau · Power BI |
| AI Automation | n8n · Workflow Automation · Agent Pipelines |
| Data Platforms | Snowflake · Google Workspace |
| Dev Tools | VS Code · Git · GitHub |

---

## About Me

**Surya Teja Nagula**
Analytics Lead | Product Ownership | AI Automation 

10+ years of experience across banking and retail domains, translating complex data into insight-driven strategies.

- MBA — Indian Institute of Management (IIM) Tiruchirappalli
- MS Data Analytics — University of South Carolina, Darla Moore School of Business (GPA 4.0)
- CFA Level I · Tableau Certified · Power BI Certified · Snowflake Essentials · PSPO I · SAFe 6

📍 Columbus, OH
📧 suryatejanagula2024@gmail.com

---

*This portfolio is actively maintained. New projects are added as they are completed.*
