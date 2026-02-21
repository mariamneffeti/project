"""
=============================================================
  Data Cleaning & Normalization Pipeline
  Dataset: Insurance Policy Train Data (60,868 rows x 29 cols)
=============================================================
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import warnings


# ─────────────────────────────────────────────
# 0. LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_csv("train.csv")
print(f"✅ Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
print("=" * 60)


# ─────────────────────────────────────────────
# 1. DROP USELESS COLUMNS
# ─────────────────────────────────────────────
# User_ID   → pure identifier, no predictive value
# Employer_ID → 94.3% missing — keep as binary flag then drop original
df["Has_Employer_ID"] = df["Employer_ID"].notna().astype(int)
df.drop(columns=["User_ID", "Employer_ID"], inplace=True)
print("✅ [Step 1] Dropped 'User_ID', converted 'Employer_ID' → 'Has_Employer_ID' flag")


# ─────────────────────────────────────────────
# 2. HANDLE MISSING VALUES
# ─────────────────────────────────────────────

# --- Broker_ID (13.7% missing) ---
# Keep a binary flag (same pattern as Has_Employer_ID), then treat missing as separate category
df["Has_Broker_ID"] = df["Broker_ID"].notna().astype(int)
df["Broker_ID"] = df["Broker_ID"].fillna(-1).astype(int).astype(str)

# --- Acquisition_Channel (~1% missing) ---
df["Acquisition_Channel"] = df["Acquisition_Channel"].fillna(
    df["Acquisition_Channel"].mode()[0]
)

# --- Region_Code (~0.5% missing) ---
df["Region_Code"] = df["Region_Code"].fillna("Unknown")

# --- Deductible_Tier (~0.5% missing) ---
df["Deductible_Tier"] = df["Deductible_Tier"].fillna(
    df["Deductible_Tier"].mode()[0]
)

# --- Child_Dependents (4 missing rows) ---
df["Child_Dependents"] = df["Child_Dependents"].fillna(
    df["Child_Dependents"].median()
).astype(int)

print("✅ [Step 2] Missing values handled for all columns")


# ─────────────────────────────────────────────
# 3. FIX SUSPICIOUS / OUTLIER VALUES
# ─────────────────────────────────────────────

# --- Child_Dependents: cap at 5 (value 10 is likely a data entry error) ---
df["Child_Dependents"] = df["Child_Dependents"].clip(upper=5)

# --- Estimated_Annual_Income: winsorize at 99th percentile, then log-transform ---
income_cap = df["Estimated_Annual_Income"].quantile(0.99)  # ~$94,571
df["Estimated_Annual_Income"] = df["Estimated_Annual_Income"].clip(upper=income_cap)
df["Estimated_Annual_Income_Log"] = np.log1p(df["Estimated_Annual_Income"])
# Keep original as well for reference; drop if desired
df.drop(columns=["Estimated_Annual_Income"], inplace=True)

# --- Days_Since_Quote: winsorize at 99th percentile ---
days_quote_cap = df["Days_Since_Quote"].quantile(0.99)  # ~350 days
df["Days_Since_Quote"] = df["Days_Since_Quote"].clip(upper=days_quote_cap)

# --- Underwriting_Processing_Days: heavily skewed, log-transform ---
df["Underwriting_Processing_Days"] = np.log1p(df["Underwriting_Processing_Days"])

print("✅ [Step 3] Outliers handled (winsorize + log transforms applied)")


# ─────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ─────────────────────────────────────────────

# --- Total dependents ---
df["Total_Dependents"] = (
    df["Adult_Dependents"] + df["Child_Dependents"] + df["Infant_Dependents"]
)

# --- Cyclical encoding for Policy_Start_Month ---
month_order = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}
df["Month_Num"] = df["Policy_Start_Month"].map(month_order)
df["Month_Sin"] = np.sin(2 * np.pi * df["Month_Num"] / 12)
df["Month_Cos"] = np.cos(2 * np.pi * df["Month_Num"] / 12)
df.drop(columns=["Policy_Start_Month", "Month_Num"], inplace=True)

# --- Cyclical encoding for Policy_Start_Day ---
df["Day_Sin"] = np.sin(2 * np.pi * df["Policy_Start_Day"] / 31)
df["Day_Cos"] = np.cos(2 * np.pi * df["Policy_Start_Day"] / 31)
df.drop(columns=["Policy_Start_Day"], inplace=True)

# --- Cyclical encoding for Policy_Start_Week ---
df["Week_Sin"] = np.sin(2 * np.pi * df["Policy_Start_Week"] / 52)
df["Week_Cos"] = np.cos(2 * np.pi * df["Policy_Start_Week"] / 52)
df.drop(columns=["Policy_Start_Week"], inplace=True)

print("✅ [Step 4] Feature engineering done (Total_Dependents, cyclical date features)")


# ─────────────────────────────────────────────
# 5. ENCODE CATEGORICAL VARIABLES
# ─────────────────────────────────────────────

# --- Ordinal Encoding: Deductible_Tier ---
deductible_order = {
    "Tier_1_High_Ded": 1,
    "Tier_2_Mid_Ded":  2,
    "Tier_3_Low_Ded":  3,
    "Tier_4_Zero_Ded": 4
}
df["Deductible_Tier"] = df["Deductible_Tier"].map(deductible_order)

# --- Binary Encoding: Broker_Agency_Type (only 2 values) ---
df["Is_National_Corporate"] = (df["Broker_Agency_Type"] == "National_Corporate").astype(int)
df.drop(columns=["Broker_Agency_Type"], inplace=True)

# --- One-Hot Encoding: low-cardinality categoricals ---
ohe_cols = [
    "Acquisition_Channel",    # 5 unique
    "Payment_Schedule",       # 3 unique
    "Employment_Status",      # 4 unique
    "Purchased_Coverage_Bundle",  # 10 unique
]
df = pd.get_dummies(df, columns=ohe_cols, drop_first=True, dtype=int)

# --- Target Encoding: Region_Code (high cardinality, many country codes) ---
# Using the mean of the target variable per region
target_col = "Policy_Cancelled_Post_Purchase"
region_means = df.groupby("Region_Code")[target_col].mean()
df["Region_Code_TargetEnc"] = df["Region_Code"].map(region_means)
df.drop(columns=["Region_Code"], inplace=True)

# --- Broker_ID: already turned into string; apply target encoding ---
broker_means = df.groupby("Broker_ID")[target_col].mean()
df["Broker_ID_TargetEnc"] = df["Broker_ID"].map(broker_means)
df.drop(columns=["Broker_ID"], inplace=True)

print("✅ [Step 5] Categorical encoding complete")
print("           → Ordinal:       Deductible_Tier")
print("           → Binary:        Broker_Agency_Type")
print("           → One-Hot:       Acquisition_Channel, Payment_Schedule,")
print("                            Employment_Status, Purchased_Coverage_Bundle")
print("           → Target Enc:    Region_Code, Broker_ID")


# ─────────────────────────────────────────────
# 6. SCALE NUMERIC FEATURES
# ─────────────────────────────────────────────
# RobustScaler → columns that still have skew/outlier sensitivity
# StandardScaler → roughly normal columns
# NOTE: Skip scaling if you plan to use tree-based models (XGBoost, LightGBM, RF)

robust_cols = [
    "Estimated_Annual_Income_Log",
    "Underwriting_Processing_Days",
    "Days_Since_Quote",
]

standard_cols = [
    "Previous_Policy_Duration_Months",
    "Grace_Period_Extensions",
    "Adult_Dependents",
    "Child_Dependents",
    "Infant_Dependents",
    "Total_Dependents",
    "Policy_Amendments_Count",
    "Vehicles_on_Policy",
    "Custom_Riders_Requested",
    "Years_Without_Claims",
    "Previous_Claims_Filed",
    "Deductible_Tier",
    "Region_Code_TargetEnc",
    "Broker_ID_TargetEnc",
]

# Only scale features that exist after all transformations
robust_cols   = [c for c in robust_cols   if c in df.columns]
standard_cols = [c for c in standard_cols if c in df.columns]

robust_scaler   = RobustScaler()
standard_scaler = StandardScaler()

df[robust_cols]   = robust_scaler.fit_transform(df[robust_cols])
df[standard_cols] = standard_scaler.fit_transform(df[standard_cols])

print("✅ [Step 6] Scaling applied")
print(f"           → RobustScaler:   {robust_cols}")
print(f"           → StandardScaler: {standard_cols}")


# ─────────────────────────────────────────────
# 7. FINAL VALIDATION
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("📊 FINAL DATASET SUMMARY")
print("=" * 60)
print(f"  Shape          : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"  Missing values : {df.isnull().sum().sum()}")
print(f"  Target balance : {df[target_col].value_counts().to_dict()}")
print(f"  Dtypes         : {df.dtypes.value_counts().to_dict()}")
print("\n  Columns in final dataset:")
for col in df.columns:
    print(f"    - {col}")


# ─────────────────────────────────────────────
# 8. SAVE CLEANED DATASET
# ─────────────────────────────────────────────
output_path = "train_cleaned.csv"
df.to_csv(output_path, index=False)
print(f"\n✅ Cleaned dataset saved to: {output_path}")
print("=" * 60)