import pandas as pd
import urllib.request
import os
import zipfile

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
zip_path = "bank_data.zip"
extract_folder = "bank_data"

# Download zip
if not os.path.exists(zip_path):
    print("Downloading dataset...")
    urllib.request.urlretrieve(url, zip_path)
    print("Download complete!")
else:
    print("Zip already exists.")

# Extract
if not os.path.exists(extract_folder):
    print("Extracting files...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)
    print("Extraction complete!")
else:
    print("Files already extracted.")

# Load CSV
csv_path = os.path.join(extract_folder, "bank-additional", "bank-additional-full.csv")
df = pd.read_csv(csv_path, sep=";")


print(f"\nLoaded {len(df):,} customers with {len(df.columns)} columns")
print("\nColumn names:")
print(df.columns.tolist())
print("\nFirst 3 rows:")
print(df.head(3).to_string())
print("\nTarget column (y) — did they subscribe?")
print(df['y'].value_counts())

print("\n=== DATA QUALITY CHECK ===")
print("Any missing values?", df.isnull().sum().sum(), "nulls found")
print("\nColumn types:")
print(df.dtypes)

# Add to data_prep.py
print("""
=== WHAT EACH COLUMN MEANS ===

CUSTOMER PROFILE:
  age        - how old the customer is
  job        - their occupation (admin, blue-collar, technician, etc.)
  marital    - married / single / divorced
  education  - highest level of schooling
  default    - have they ever defaulted on a loan? (yes/no/unknown)
  housing    - do they have a housing loan? (yes/no/unknown)
  loan       - do they have a personal loan? (yes/no/unknown)

CONTACT HISTORY:
  contact    - how we reached them: 'cellular' or 'telephone'
  month      - which month of the year we last called
  day_of_week - which day we last called
  duration   - how long the last call lasted in SECONDS
                *** WARNING: we will DELETE this column - see tip below ***

CAMPAIGN DATA:
  campaign   - how many times we've called them THIS campaign
  pdays      - days since we last called them in a PREVIOUS campaign
  previous   - how many times we called in a PREVIOUS campaign
  poutcome   - what happened in the PREVIOUS campaign (success/failure/nonexistent)

ECONOMIC INDICATORS (external data about the economy):
  emp.var.rate   - employment variation rate
  cons.price.idx - consumer price index
  cons.conf.idx  - consumer confidence index
  euribor3m      - 3-month interest rate
  nr.employed    - number of people employed nationally

TARGET:
  y          - did the customer subscribe to a term deposit? (yes / no)
               THIS is what our model will learn to predict.
""")

# Check conversion rate by contact channel
print("Conversion rate by contact channel:")
print(df.groupby('contact')['y'].value_counts(normalize=True).unstack())

print("\nConversion rate by previous campaign outcome:")
print(df.groupby('poutcome')['y'].value_counts(normalize=True).unstack())

print("\nAverage duration by outcome (THIS is why we must drop it):")
print(df.groupby('y')['duration'].mean().round(0))
print("Subscribers had 3x longer calls - but we only know duration AFTER the call!")

#Feature engineering steps:
# ── STEP 1: DROP duration (data leakage — only known AFTER the call) ──
df = df.drop(columns=['duration'])
print("Dropped 'duration' column (data leakage risk)")

# ── STEP 2: CREATE FEATURE — Age bracket ──────────────────────────────
# Instead of raw age, bucket into life stages
df['age_bracket'] = pd.cut(
    df['age'],
    bins=[0, 30, 40, 50, 60, 100],
    labels=['under_30', '30s', '40s', '50s', '60_plus']
)

# ── STEP 3: CREATE FEATURE — Simplified job category ─────────────────
df['job_category'] = df['job'].map({
    'admin.'        : 'white_collar',
    'management'    : 'white_collar',
    'technician'    : 'skilled',
    'services'      : 'skilled',
    'blue-collar'   : 'blue_collar',
    'self-employed' : 'self_employed',
    'entrepreneur'  : 'self_employed',
    'housemaid'     : 'other',
    'student'       : 'other',
    'retired'       : 'retired',
    'unemployed'    : 'other',
    'unknown'       : 'other'
})

# ── STEP 4: CREATE FEATURE — Previous campaign signals ───────────────
df['had_previous_contact'] = (df['previous'] > 0).astype(int)
df['previous_success']     = (df['poutcome'] == 'success').astype(int)

# ── STEP 5: CREATE FEATURE — Economy score (combine 3 indicators) ────
df['econ_score'] = (
    df['emp.var.rate']  * 0.4 +
    df['cons.conf.idx'] * 0.3 +
    df['euribor3m']     * 0.3
)

# ── STEP 6: CREATE FEATURE — Contact channel ─────────────────────────
df['is_cellular'] = (df['contact'] == 'cellular').astype(int)

# ── STEP 7: CREATE FEATURE — Contact frequency (capped at 5) ─────────
df['contact_count'] = df['campaign'].clip(upper=5)

# ── STEP 8: ENCODE text columns as 0/1 numbers ───────────────────────
df['marital_single']    = (df['marital'] == 'single').astype(int)
df['education_high']    = df['education'].isin(
    ['university.degree', 'professional.course']).astype(int)
df['has_housing_loan']  = (df['housing'] == 'yes').astype(int)
df['has_personal_loan'] = (df['loan'] == 'yes').astype(int)

# ── STEP 9: ENCODE target variable (yes=1, no=0) ─────────────────────
df['target'] = (df['y'] == 'yes').astype(int)

# ── DEFINE FINAL FEATURE LIST ─────────────────────────────────────────
FEATURE_COLS = [
    'age', 'age_bracket', 'job_category',
    'marital_single', 'education_high',
    'has_housing_loan', 'has_personal_loan',
    'is_cellular', 'contact_count',
    'had_previous_contact', 'previous_success',
    'econ_score', 'nr.employed', 'cons.price.idx'
]

print(f"\nFeature engineering complete!")
print(f"Number of features: {len(FEATURE_COLS)}")
print(f"Customers who subscribed: {df['target'].sum():,} ({df['target'].mean():.1%})")

# Preview the new features
print("\nSample of engineered features (first 5 rows):")
preview_cols = ['age', 'age_bracket', 'job_category', 'is_cellular',
                'previous_success', 'econ_score', 'target']
print(df[preview_cols].head().to_string())

# ── SAVE PREPPED DATA ─────────────────────────────────────────────────
df.to_csv("bank_prepped.csv", index=False)
print("\nSaved: bank_prepped.csv")



