import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

# ── LOAD PREPPED DATA ─────────────────────────────────────────────────
df = pd.read_csv("bank_prepped.csv")
print(f"Loaded {len(df):,} customers")

# ── DEFINE FEATURES ───────────────────────────────────────────────────
FEATURES = [
    'age', 'marital_single', 'education_high',
    'has_housing_loan', 'has_personal_loan',
    'is_cellular', 'contact_count',
    'had_previous_contact', 'previous_success',
    'econ_score', 'nr.employed', 'cons.price.idx'
]

# Encode text columns (age_bracket, job_category) into numbers
for col in ['age_bracket', 'job_category']:
    if col in df.columns:
        le = LabelEncoder()
        df[col + '_enc'] = le.fit_transform(df[col].astype(str))
        FEATURES.append(col + '_enc')

X = df[FEATURES].copy()
y = df['target'].copy()

print(f"Features being used: {len(FEATURES)}")
print(f"Feature list: {FEATURES}")

# ── TRAIN / TEST SPLIT ────────────────────────────────────────────────
# 80% of data trains the model, 20% tests it (data it's never seen)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining set: {len(X_train):,} customers")
print(f"Test set:      {len(X_test):,} customers")

# ── HANDLE CLASS IMBALANCE ────────────────────────────────────────────
# Only 11% of customers subscribed. Without this fix, the model
# would just predict "no" for everyone and be 89% accurate (but useless).
ratio = (y_train == 0).sum() / (y_train == 1).sum()
print(f"\nClass balance: {ratio:.1f} non-subscribers per subscriber (handled)")

# ── TRAIN THE MODEL ───────────────────────────────────────────────────
print("\nTraining XGBoost model... (30-60 seconds)")
base_model = XGBClassifier(
    n_estimators=200,       # 200 decision trees voting together
    max_depth=4,            # Each tree looks 4 levels deep
    learning_rate=0.05,     # Small steps = more accurate final model
    scale_pos_weight=ratio, # Fix the class imbalance
    random_state=42,
    eval_metric='logloss',
    verbosity=0
)

# Calibration: makes scores TRUE probabilities (0.8 = 80% likely)
# Without this, XGBoost scores are relative ranks, not real probabilities.
print("Calibrating probability scores...")
calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
calibrated_model.fit(X_train, y_train)
print("Training complete!")

# ── EVALUATE THE MODEL ────────────────────────────────────────────────
y_proba = calibrated_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)
print(f"\n=== MODEL PERFORMANCE ===")
print(f"AUC Score: {auc:.3f}")
print("Interpretation: 0.5=random | 0.7=decent | 0.8+=good | 0.9+=excellent")

# ── SCORE ALL 41,000 CUSTOMERS ────────────────────────────────────────
print("\nScoring all customers...")
df['propensity_score'] = calibrated_model.predict_proba(X[FEATURES])[:, 1]

# Assign lead tiers: A = top priority, B = medium, C = low
df['lead_tier'] = pd.cut(
    df['propensity_score'],
    bins=[0, 0.25, 0.50, 1.0],
    labels=['C - Low', 'B - Medium', 'A - High']
)

print("\n=== LEAD TIER RESULTS ===")
print(df['lead_tier'].value_counts().to_string())
print(f"\nTier A conversion rate: {df[df['lead_tier']=='A - High']['target'].mean():.1%}")
print(f"Tier B conversion rate: {df[df['lead_tier']=='B - Medium']['target'].mean():.1%}")
print(f"Tier C conversion rate: {df[df['lead_tier']=='C - Low']['target'].mean():.1%}")
print(f"Overall baseline rate:  {df['target'].mean():.1%}")
print(f"\nLift (Tier A vs baseline): {df[df['lead_tier']=='A - High']['target'].mean() / df['target'].mean():.1f}x")

# ── SAVE OUTPUTS ──────────────────────────────────────────────────────
df.to_csv("scored_leads.csv", index=False)
pickle.dump({'model': calibrated_model, 'features': FEATURES},
            open("model.pkl", "wb"))
print("\nSaved: scored_leads.csv")
print("Saved: model.pkl")


# Build the Tier A weekly calling list
tier_a_leads = df[df['lead_tier'] == 'A - High'].copy()
tier_a_leads = tier_a_leads.sort_values('propensity_score', ascending=False)

# Assign readable customer IDs
tier_a_leads['customer_id'] = [
    'CUST-' + str(10000 + i) for i in range(len(tier_a_leads))
]

# Recommend a contact action based on each customer's profile
def get_action(row):
    if row['previous_success'] == 1:
        return "Re-engage: past subscriber — strong loyalty signal"
    elif row['had_previous_contact'] == 1:
        return "Follow-up: previously contacted, already warmed up"
    elif row['is_cellular'] == 1:
        return "New outreach: digitally-engaged customer"
    else:
        return "New outreach: prefer branch or landline contact"

tier_a_leads['recommended_action'] = tier_a_leads.apply(get_action, axis=1)

# Select and round final output columns
output = tier_a_leads[[
    'customer_id', 'propensity_score', 'lead_tier',
    'age', 'job_category', 'is_cellular',
    'had_previous_contact', 'previous_success',
    'recommended_action', 'target'
]].head(200).copy()

output['propensity_score'] = output['propensity_score'].round(3)

print("\n=== TOP 10 LEADS THIS WEEK ===")
print(output.head(10)[[
    'customer_id','propensity_score','age','recommended_action'
]].to_string(index=False))

print(f"\nTotal leads in list: {len(output)}")
print(f"Expected conversion rate: {output['target'].mean():.1%}")
print(f"vs random calling baseline: {df['target'].mean():.1%}")
print(f"Precision uplift: {output['target'].mean() / df['target'].mean():.1f}x")

output.to_csv("weekly_lead_list.csv", index=False)
print("\nSaved: weekly_lead_list.csv — ready for the sales team")