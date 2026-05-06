# Retail Banking — Lead Intelligence Dashboard
### Next-Best-Action · Propensity Scoring · Weekly Calling Priority
**Project by Surya Teja Nagula**

---

## Table of Contents

1. [Project Goal](#1-project-goal)
2. [Business Problem Being Solved](#2-business-problem-being-solved)
3. [How It Works — High Level](#3-how-it-works--high-level)
4. [Key Results Achieved](#4-key-results-achieved)
5. [Folder Structure](#5-folder-structure)
6. [File-by-File Breakdown](#6-file-by-file-breakdown)
7. [Data Source](#7-data-source)
8. [Requirements](#8-requirements)
9. [First-Time Setup (Windows and Mac)](#9-first-time-setup-windows-and-mac)
10. [How to Run the App (Every Time)](#10-how-to-run-the-app-every-time)
11. [App Features Walkthrough](#11-app-features-walkthrough)
12. [Troubleshooting](#12-troubleshooting)
13. [Portfolio Context](#13-portfolio-context)

---

## 1. Project Goal

Build a **complete, end-to-end Next-Best-Action (NBA) lead scoring engine** for retail banking — from raw customer data through machine learning to a live, interactive Streamlit dashboard that a non-technical sales manager can use every Monday morning.

The system answers one business question:

> *"Out of 41,000+ customers, which 200 should our bankers call this week — and what should they offer?"*

This project was built as **Portfolio Project 1** for the **Decision Analytics Lead** role application at **Huntington Bank**, demonstrating:

- Quantitative analytics and propensity modeling
- Banking domain knowledge (acquisition, cross-sell, activation, balance build, retention)
- Data storytelling and executive communication
- End-to-end delivery from raw data to deployed dashboard

---

## 2. Business Problem Being Solved

Retail banks run outbound sales campaigns — bankers call customers to offer products like term deposits, savings accounts, loans, and credit cards. Without a scoring system:

- Bankers call customers randomly or by intuition
- Conversion rates hover around **11%** (9 out of 10 calls go nowhere)
- Cost-per-acquisition is high
- High-value customers get missed; low-propensity customers get over-contacted

**This system solves that** by scoring every customer with a machine learning model, ranking them by conversion likelihood, and generating a prioritized weekly lead list — so bankers spend time on the customers most likely to say yes.

---

## 3. How It Works — High Level

```
Raw Customer Data (41,188 records)
        |
        v
  data_prep.py          <-- Clean, engineer features, drop leakage columns
        |
        v
  model.py              <-- Train XGBoost model, calibrate probabilities,
                            score all customers, assign lead tiers (A/B/C),
                            generate weekly lead list
        |
        v
  scored_leads.csv      <-- Every customer with a propensity score + tier
  weekly_lead_list.csv  <-- Top 200 Tier A leads ready for banker use
  model.pkl             <-- Saved trained model
        |
        v
  app.py                <-- Streamlit dashboard — filters, lead list,
                            score distribution charts, segment insights
```

**Model architecture:**
- Algorithm: XGBoost (gradient boosting — ensemble of 200 decision trees)
- Calibration: Isotonic probability calibration (converts raw scores to true probabilities)
- Class imbalance handling: `scale_pos_weight` to compensate for 89% non-subscriber majority
- Train/test split: 80% training, 20% held-out test set

---

## 4. Key Results Achieved

| Metric | Value |
|--------|-------|
| Total customers scored | 41,188 |
| AUC score | 0.810 |
| Tier A leads identified | 1,466 (3.6% of book) |
| Tier A conversion rate | 72.2% |
| Baseline conversion rate | 11.3% |
| Lift over baseline | **6.4x** |
| Top-200 list conversion rate | 90.5% |
| Precision uplift (top 200) | **8.0x** |

**What 6.4x lift means in practice:** A banker making 100 random calls expects ~11 conversions. Using this system's Tier A list, the same 100 calls yield ~72 conversions — without calling any additional customers.

---

## 5. Folder Structure

```
E:\SuryaNagula-GitHub\data-analytics-portfolio\Analytics\banking_nba\          # Windows path
~/SuryaNagula-GitHub/data-analytics-portfolio/Analytics/banking_nba/           # Mac path
|
|-- data_prep.py            # Step 1: Download data, engineer features
|-- model.py                # Step 2: Train model, score customers, generate lead list
|-- app.py                  # Step 3: Streamlit dashboard
|
|-- bank_data.csv           # Raw downloaded dataset (auto-created by data_prep.py)
|-- bank_prepped.csv        # Feature-engineered dataset (auto-created by data_prep.py)
|-- scored_leads.csv        # All 41,188 customers with scores and tiers (auto-created by model.py)
|-- weekly_lead_list.csv    # Top 200 Tier A leads (auto-created by model.py)
|-- model.pkl               # Saved trained model (auto-created by model.py)
|
|-- requirements.txt        # Python package list (for Streamlit Cloud deployment)
|-- README.md               # This file
|
|-- venv\                   # Virtual environment - Windows (do not edit manually)
|-- venv/                   # Virtual environment - Mac (do not edit manually)
```

> Files marked **auto-created** do not need to be committed to GitHub if you want to keep the repo clean. They are regenerated by running the scripts in order.

---

## 6. File-by-File Breakdown

### `data_prep.py`
**Purpose:** Downloads the raw dataset, cleans it, engineers features, and saves `bank_prepped.csv`.

**What it does step by step:**
1. Downloads the UCI Bank Marketing dataset from the UCI repository (auto-skips if already downloaded)
2. Loads 41,188 rows with 21 columns into a pandas DataFrame
3. **Drops `duration`** — the call length column is dropped because it constitutes data leakage: you only know call duration after the call happens, so it cannot be used in a pre-call lead scoring model
4. Creates engineered features:
   - `age_bracket` — groups age into life stages (under_30, 30s, 40s, 50s, 60_plus)
   - `job_category` — simplifies 12 job types into 6 broader categories
   - `had_previous_contact` — binary flag: was this customer contacted in a prior campaign?
   - `previous_success` — binary flag: did the prior campaign succeed?
   - `econ_score` — single composite score combining 3 economic indicators
   - `is_cellular` — binary flag: was contact via mobile phone?
   - `contact_count` — number of contacts this campaign, capped at 5 to reduce noise
5. Encodes binary text columns (marital, education, housing, loan) as 0/1 integers
6. Encodes target column: `y` (yes/no) becomes `target` (1/0)
7. Saves `bank_prepped.csv`

**Run with:**
```
python data_prep.py
```

**Output:** `bank_prepped.csv`

---

### `model.py`
**Purpose:** Trains the XGBoost model, evaluates it, scores all customers, assigns lead tiers, and generates the weekly lead list.

**What it does step by step:**
1. Loads `bank_prepped.csv`
2. Label-encodes categorical columns (`age_bracket`, `job_category`) into numeric form
3. Splits data: 80% training / 20% test, stratified to preserve class balance
4. Calculates class imbalance ratio (~7.9 non-subscribers per subscriber) and passes to XGBoost as `scale_pos_weight`
5. Trains `XGBClassifier` with 200 estimators, depth 4, learning rate 0.05
6. Wraps in `CalibratedClassifierCV` (isotonic, 3-fold) — this step ensures scores are true probabilities, not just relative ranks
7. Evaluates on held-out test set: prints AUC score
8. Scores all 41,188 customers
9. Assigns lead tiers:
   - **Tier A (High):** score > 0.50
   - **Tier B (Medium):** score 0.25 to 0.50
   - **Tier C (Low):** score < 0.25
10. Adds recommended action per customer based on profile flags
11. Saves `scored_leads.csv`, `weekly_lead_list.csv`, and `model.pkl`

**Run with:**
```
python model.py
```

**Expected runtime:** 30 to 60 seconds on a standard Windows laptop

**Output:** `scored_leads.csv`, `weekly_lead_list.csv`, `model.pkl`

---

### `app.py`
**Purpose:** Streamlit web dashboard that reads `scored_leads.csv` and presents an interactive lead intelligence interface.

**What it contains:**
- `st.set_page_config` — sets page title, icon, and wide layout
- `@st.cache_data` decorated `load_data()` — loads `scored_leads.csv` once and caches it for performance
- **KPI cards row** — 4 headline metrics: Total Customers Scored, Tier A Priority Leads, Tier A Conversion Rate, Baseline Conversion Rate
- **Sidebar filters** — Lead Tier multiselect, Min Propensity Score slider, Previously Contacted checkbox, Previous Success checkbox, Show Top N slider
- **Tab 1: Lead List** — filtered, sortable, colour-coded table with a CSV download button
- **Tab 2: Score Distribution** — histogram of propensity scores by tier, bar chart of conversion rate by tier
- **Tab 3: Segment Insights** — conversion rate by job category (horizontal bar chart), previous campaign success as a conversion signal (bar chart)
- Footer — model attribution and project credit

**Run with:**
```
streamlit run app.py
```

**Requires:** `scored_leads.csv` to exist in the same folder (run `model.py` first)

---

### `requirements.txt`
**Purpose:** Lists all Python packages needed. Used by Streamlit Community Cloud during deployment so it knows what to install.

**Contents:**
```
pandas
numpy
scikit-learn
xgboost
plotly
streamlit
```

**Not needed for local runs** (your venv already has everything). Only needed for cloud deployment.

---

### `bank_data.csv`
Raw dataset downloaded from the UCI Machine Learning Repository. 41,188 rows, 21 columns, semicolon-delimited. Auto-downloaded by `data_prep.py` if not present.

### `bank_prepped.csv`
Cleaned and feature-engineered version of the raw data. 41,188 rows with additional engineered columns and `duration` removed. Created by `data_prep.py`.

### `scored_leads.csv`
All 41,188 customers with their propensity scores, lead tiers, and engineered features. This is the file the Streamlit app reads. Created by `model.py`.

### `weekly_lead_list.csv`
Top 200 Tier A customers sorted by propensity score, with recommended actions. This is the file a banker or sales manager would download Monday morning. Created by `model.py`.

### `model.pkl`
The trained and calibrated XGBoost model saved to disk using Python's `pickle` module. Can be loaded later to score new customers without retraining. Created by `model.py`.

---

## 7. Data Source

**Dataset:** UCI Bank Marketing Dataset  
**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing)  
**File used:** `bank-additional-full.csv`  
**Records:** 41,188 customers  
**Features:** 21 columns  
**Target:** `y` — did the customer subscribe to a term deposit? (yes / no)  
**Origin:** A Portuguese bank's direct marketing phone campaigns (2008 to 2013)

**Key columns used in the model:**

| Column | Type | Description |
|--------|------|-------------|
| age | numeric | Customer age |
| job | categorical | Occupation |
| marital | categorical | Marital status |
| education | categorical | Education level |
| housing | categorical | Has housing loan? |
| loan | categorical | Has personal loan? |
| contact | categorical | Contact channel (cellular / telephone) |
| campaign | numeric | Number of contacts this campaign |
| previous | numeric | Contacts in prior campaign |
| poutcome | categorical | Outcome of prior campaign |
| emp.var.rate | numeric | Employment variation rate |
| cons.price.idx | numeric | Consumer price index |
| cons.conf.idx | numeric | Consumer confidence index |
| euribor3m | numeric | 3-month Euribor rate |
| nr.employed | numeric | Number employed nationally |
| y | categorical | **TARGET:** subscribed (yes / no) |

**Column dropped intentionally:**

| Column | Reason |
|--------|--------|
| duration | Data leakage — call duration is only known after the call occurs. Using it would make the model appear highly accurate in testing but completely unusable in production, since you need to score customers before calling them. |

---

## 8. Requirements

### Python version
Python 3.9 or higher (3.11 recommended)

### Python packages

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | any | Data loading and manipulation |
| numpy | any | Numerical operations |
| scikit-learn | any | Train/test split, calibration, AUC scoring |
| xgboost | any | Gradient boosting model |
| streamlit | any | Web dashboard |
| plotly | any | Interactive charts in the dashboard |
| matplotlib | any | Supporting charts |
| seaborn | any | Supporting charts |

### System requirements

**Windows:**
- Windows 10 or 11
- VS Code (any recent version)
- Python installed with "Add to PATH" ticked during installation
- Internet connection (for initial dataset download only)
- ~50MB free disk space

**Mac:**
- macOS 11 (Big Sur) or later
- VS Code (any recent version) or Terminal app
- Python 3.9+ installed via python.org or Homebrew (`brew install python`)
- Internet connection (for initial dataset download only)
- ~50MB free disk space

---

## 9. First-Time Setup (Windows and Mac)

Follow these steps **once only** when setting up the project for the first time.

---

### Windows Setup (VS Code + PowerShell)

#### Step 1 — Open the project in VS Code
Open VS Code, then open the terminal with `Ctrl+`` ` `` (backtick key, top-left under Escape).

Navigate to the project folder:
```powershell
cd E:\SuryaNagula-GitHub\data-analytics-portfolio\Analytics\banking_nba
```

#### Step 2 — Create a virtual environment
```powershell
python -m venv venv
```

#### Step 3 — Allow scripts to run (one-time Windows security setting)
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### Step 4 — Activate the virtual environment
```powershell
.\venv\Scripts\Activate.ps1
```
You will see `(venv)` appear at the start of your terminal line. This confirms the virtual environment is active.

#### Step 5 — Install all required packages
```powershell
pip install pandas numpy scikit-learn xgboost streamlit matplotlib seaborn plotly
```
This takes 2 to 5 minutes depending on internet speed.

#### Step 6 — Run the data preparation script
```powershell
python data_prep.py
```
Wait for: `Saved: bank_prepped.csv`

#### Step 7 — Train the model and generate lead scores
```powershell
python model.py
```
Wait for: `Saved: scored_leads.csv` and `Saved: model.pkl`
This takes 30 to 60 seconds.

#### Step 8 — Launch the dashboard
```powershell
streamlit run app.py
```
Your browser opens automatically to `http://localhost:8501`

---

### Mac Setup (VS Code or Terminal)

#### Step 1 — Open Terminal
Open VS Code and press `` Ctrl+` `` to open the integrated terminal, or open the Mac Terminal app directly from Spotlight (`Cmd+Space`, type Terminal).

Navigate to the project folder:
```bash
cd ~/SuryaNagula-GitHub/data-analytics-portfolio/Analytics/banking_nba
```

> If the folder does not exist yet, create it first:
> ```bash
> mkdir -p ~/SuryaNagula-GitHub/data-analytics-portfolio/Analytics/banking_nba
> cd ~/SuryaNagula-GitHub/data-analytics-portfolio/Analytics/banking_nba
> ```

#### Step 2 — Check Python is installed
```bash
python3 --version
```
You should see `Python 3.9.x` or higher. If not, install Python from [python.org](https://www.python.org/downloads/) or run:
```bash
brew install python
```

#### Step 3 — Create a virtual environment
```bash
python3 -m venv venv
```

#### Step 4 — Activate the virtual environment
```bash
source venv/bin/activate
```
You will see `(venv)` appear at the start of your terminal line. This confirms the virtual environment is active.

> On Mac you use `source venv/bin/activate` — this is different from Windows which uses `.\venv\Scripts\Activate.ps1`

#### Step 5 — Install all required packages
```bash
pip install pandas numpy scikit-learn xgboost streamlit matplotlib seaborn plotly
```
This takes 2 to 5 minutes depending on internet speed.

#### Step 6 — Run the data preparation script
```bash
python data_prep.py
```
Wait for: `Saved: bank_prepped.csv`

#### Step 7 — Train the model and generate lead scores
```bash
python model.py
```
Wait for: `Saved: scored_leads.csv` and `Saved: model.pkl`
This takes 30 to 60 seconds.

#### Step 8 — Launch the dashboard
```bash
streamlit run app.py
```
Your browser opens automatically to `http://localhost:8501`

---

## 10. How to Run the App (Every Time)

Once the first-time setup is complete and `scored_leads.csv` exists, you only need to do the following each time you want to use the dashboard.

---

### Windows — Daily Launch

#### Step 1 — Open VS Code and open the terminal
Press `` Ctrl+` ``

#### Step 2 — Navigate to the project folder
```powershell
cd E:\SuryaNagula-GitHub\data-analytics-portfolio\Analytics\banking_nba
```

#### Step 3 — Activate the virtual environment
```powershell
.\venv\Scripts\Activate.ps1
```
Confirm you see `(venv)` at the start of the line.

#### Step 4 — Launch the app
```powershell
streamlit run app.py
```

Your browser opens to `http://localhost:8501` automatically.
If it does not open, navigate there manually in Chrome or Edge.

#### To stop the app
Click inside the terminal and press `Ctrl+C`.

#### Windows quick reference — 3-command launch
```powershell
cd E:\SuryaNagula-GitHub\data-analytics-portfolio\Analytics\banking_nba
.\venv\Scripts\Activate.ps1
streamlit run app.py
```

---

### Mac — Daily Launch

#### Step 1 — Open VS Code terminal or Mac Terminal
In VS Code press `` Ctrl+` ``, or open Terminal from Spotlight (`Cmd+Space`, type Terminal).

#### Step 2 — Navigate to the project folder
```bash
cd ~/SuryaNagula-GitHub/data-analytics-portfolio/Analytics/banking_nba
```

#### Step 3 — Activate the virtual environment
```bash
source venv/bin/activate
```
Confirm you see `(venv)` at the start of the line.

#### Step 4 — Launch the app
```bash
streamlit run app.py
```

Your browser opens to `http://localhost:8501` automatically.
If it does not open, navigate there manually in Safari, Chrome, or Firefox.

#### To stop the app
Click inside the terminal and press `Ctrl+C`.

#### Mac quick reference — 3-command launch
```bash
cd ~/SuryaNagula-GitHub/data-analytics-portfolio/Analytics/banking_nba
source venv/bin/activate
streamlit run app.py
```

---

### When to re-run data_prep.py and model.py

You only need to re-run these scripts if:
- You deleted `scored_leads.csv` or `bank_prepped.csv`
- You want to retrain the model with different parameters
- You are setting up the project on a new machine

If `scored_leads.csv` already exists, you can go straight to `streamlit run app.py`.

---

## 11. App Features Walkthrough

### Sidebar (left panel)
The sidebar contains all filters. Changes apply instantly across all three tabs.

| Filter | What it does |
|--------|-------------|
| Lead Tier | Show only A - High, B - Medium, or C - Low leads (or any combination) |
| Min Propensity Score | Slider from 0.0 to 1.0 — hides any leads below this score |
| Previously contacted only | Checkbox — show only customers contacted in a prior campaign |
| Previous success only | Checkbox — show only customers whose prior campaign was a success |
| Show top N leads | Slider — limits the table to the top N results (10 to 500) |

### Tab 1 — Lead List
- Displays a filtered, ranked table of customers sorted by propensity score
- The propensity score column is colour-coded green — darker green means higher score
- Columns shown: propensity_score, lead_tier, age, is_cellular, had_previous_contact, previous_success
- **Download Lead List as CSV** button exports the current filtered view

### Tab 2 — Score Distribution
- Left chart: histogram showing how propensity scores are distributed across all customers, coloured by tier
- Right chart: bar chart comparing conversion rates by tier (A vs B vs C)

### Tab 3 — Segment Insights
- Left chart: conversion rate by job category (horizontal bar chart) — shows which customer types convert best
- Right chart: previous campaign success as a conversion signal — shows the dramatic difference between customers with and without prior success

---

## 12. Troubleshooting

### "streamlit is not recognized" or "streamlit: command not found"

Your virtual environment is not active.

**Windows:**
```powershell
.\venv\Scripts\Activate.ps1
```

**Mac:**
```bash
source venv/bin/activate
```
Confirm you see `(venv)` at the start of the line, then try again.

---

### "ModuleNotFoundError: No module named xgboost" (or any other module)

Your virtual environment is not active, or packages were installed outside the venv.

**Windows:**
```powershell
.\venv\Scripts\Activate.ps1
pip install xgboost
```

**Mac:**
```bash
source venv/bin/activate
pip install xgboost
```

---

### App opens but shows a blank page

`scored_leads.csv` does not exist yet. Open a second terminal tab and run:

**Windows** (`Ctrl+Shift+`` ` `` ` for new tab):
```powershell
.\venv\Scripts\Activate.ps1
python data_prep.py
python model.py
```

**Mac** (`Cmd+T` for new tab in Terminal):
```bash
source venv/bin/activate
python data_prep.py
python model.py
```
Then hard-refresh the browser with `Ctrl+Shift+R` (Windows) or `Cmd+Shift+R` (Mac).

---

### "python is not recognized" (Windows)

Python is not on your system PATH. Reinstall Python from python.org and tick **"Add Python to PATH"** during installation. Then restart VS Code.

### "python3: command not found" (Mac)

Install Python via Homebrew:
```bash
brew install python
```
Or download directly from [python.org](https://www.python.org/downloads/macos/).

---

### Scripts disabled error when activating venv (Windows only)

Run this once:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

### App opened in browser but shows an error about scored_leads.csv path

Make sure you launched `streamlit run app.py` from inside the project folder, not from a different directory.

**Windows:** Confirm your terminal shows `E:\SuryaNagula-GitHub\data-analytics-portfolio\Analytics\banking_nba>`
**Mac:** Confirm your terminal shows `~/SuryaNagula-GitHub/data-analytics-portfolio/Analytics/banking_nba`

---

### Port 8501 already in use

Another Streamlit app is already running. Either stop it (`Ctrl+C` in its terminal), or launch on a different port:

**Windows:**
```powershell
streamlit run app.py --server.port 8502
```

**Mac:**
```bash
streamlit run app.py --server.port 8502
```
Then go to `http://localhost:8502` in your browser.

---

### Mac only — "SSL: CERTIFICATE_VERIFY_FAILED" when downloading dataset

Run this once after installing Python on Mac:
```bash
/Applications/Python\ 3.11/Install\ Certificates.command
```
Replace `3.11` with your installed Python version.

---

## 13. Portfolio Context

This project was built as **Portfolio Project 1 of 3** for the **Decision Analytics Lead** role application at **Huntington Bank** (Columbus, OH — Reference R0071952).

**What this project demonstrates:**

| JD Requirement | How This Project Addresses It |
|----------------|-------------------------------|
| "Quantitative go-to within the group" | XGBoost model with probability calibration, AUC evaluation, class imbalance handling |
| "Sales force lead lists and next best action methods" | Ranked Tier A weekly lead list with recommended action per customer |
| "Acquisition, cross-sell, activation, utilization, balance build, retention" | UCI dataset models subscription acquisition; framework extends to all product motions |
| "Bring insights to life — truth and meaning of data" | Streamlit dashboard with KPI cards, score distribution, segment insights |
| "Helps create the visual for commercial success and ROI" | Tier A 72.2% vs 11.3% baseline — the ROI story is explicit in the dashboard |
| "Strong knowledge of deposit, lending, investment products" | Terminology and framework apply directly to banking product motions |
| "Python, SQL or other tools" | Python (pandas, scikit-learn, XGBoost, Streamlit, Plotly) |
| "May mentor less experienced colleagues" | README and step-by-step guide demonstrate communication and documentation skills |

**Model:** XGBoost + Isotonic Calibration
**Data:** UCI Bank Marketing Dataset (proxy for real banking customer file)
**Stack:** Python, pandas, scikit-learn, XGBoost, Streamlit, Plotly
**Deployment:** Streamlit Community Cloud (free tier)

---

*Built by Surya Teja Nagula*
