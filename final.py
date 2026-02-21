import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder

df = pd.read_csv("train.csv")

X = df.drop(columns=["Purchased_Coverage_Bundle"])
Y = df["Purchased_Coverage_Bundle"]

df["Has_Employer_ID"] = df["Employer_ID"].notna().astype(int)
df["Has_Real_Broker"] = df["Broker_ID"].notna().astype(int)
df["Child_Dependents"] = df["Child_Dependents"].fillna(0)
df["Employer_ID"] = df["Employer_ID"].fillna(-1)
df["Broker_ID"] = df["Broker_ID"].fillna(-1)
df["Region_Code"] = df["Region_Code"].fillna("Unknown")
df["Deductible_Tier"] = df["Deductible_Tier"].fillna("Unknown")
df["Acquisition_Channel"] = df["Acquisition_Channel"].fillna("Unknown")
df["Is_Year_End_Purchase"] = df["Policy_Start_Month"].isin(["November", "December"]).astype(int)
df["Is_Corporate_Acquisition"] = (df["Acquisition_Channel"] == "Corporate_Partner").astype(int)
df["Channel_Agency_Combo"] = df["Acquisition_Channel"] + "_" + df["Broker_Agency_Type"]

df["Total_Dependents"] = df["Adult_Dependents"] + df["Child_Dependents"] + df["Infant_Dependents"]
df["Has_Children"] = ((df["Child_Dependents"] + df["Infant_Dependents"]) > 0).astype(int)
df["Has_Infants"] = (df["Infant_Dependents"] > 0).astype(int)
df["Family_Size_Category"] = pd.cut(df["Total_Dependents"], bins=[-1, 1, 3, 6, 100], labels=[0, 1, 2, 3]).astype(
        int)
df["Is_Single_Adult"] = ((df["Adult_Dependents"] == 1) & (df["Total_Dependents"] == 1)).astype(int)

df["Income_Per_Dependent"] = df["Estimated_Annual_Income"] / (df["Total_Dependents"] + 1)
df["Is_High_Earner"] = (df["Estimated_Annual_Income"] > df["Estimated_Annual_Income"].quantile(0.75)).astype(int)
df["Income_Bracket"] = pd.qcut(df["Estimated_Annual_Income"], q=5, labels=[0, 1, 2, 3, 4],
                                   duplicates='drop').astype(int)

df["Risk_Score"] = (df["Previous_Claims_Filed"] * 2 +
                        df["Policy_Cancelled_Post_Purchase"] * 3 +
                        df["Grace_Period_Extensions"] -
                        df["Years_Without_Claims"])
df["Is_Clean_Customer"] = ((df["Previous_Claims_Filed"] == 0) &
                               (df["Policy_Cancelled_Post_Purchase"] == 0) &
                               (df["Grace_Period_Extensions"] == 0)).astype(int)
df["Claims_Per_Month"] = df["Previous_Claims_Filed"] / (df["Previous_Policy_Duration_Months"] + 1)
df["Is_Loyal_Customer"] = (df["Previous_Policy_Duration_Months"] >
                               df["Previous_Policy_Duration_Months"].median()).astype(int)

df["Is_Fast_Buyer"] = (df["Days_Since_Quote"] <= 7).astype(int)
df["Is_High_Maintenance"] = ((df["Policy_Amendments_Count"] > 2) |
                                 (df["Grace_Period_Extensions"] > 3)).astype(int)
df["Loyal_No_Cancel"] = ((df["Existing_Policyholder"] == 1) &
                             (df["Policy_Cancelled_Post_Purchase"] == 0)).astype(int)

df["Has_Vehicles"] = (df["Vehicles_on_Policy"] > 0).astype(int)
df["Has_Multiple_Vehicles"] = (df["Vehicles_on_Policy"] > 1).astype(int)
df["Is_Customizer"] = (df["Custom_Riders_Requested"] > 0).astype(int)
df["Riders_Per_Vehicle"] = df["Custom_Riders_Requested"] / (df["Vehicles_on_Policy"] + 1)

df["Week_Bin"] = pd.cut(df["Policy_Start_Week"], bins=[0, 13, 26, 39, 53], labels=[0, 1, 2, 3]).astype(int)

cat_cols = ["Region_Code", "Broker_Agency_Type", "Deductible_Tier",
                "Acquisition_Channel", "Payment_Schedule",
                "Employment_Status", "Policy_Start_Month", "Channel_Agency_Combo"]
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))



'''
numerical_col = df.select_dtypes(['int64', 'float64']).columns.tolist()

object_col = df.select_dtypes(['object']).columns.tolist()

numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])
'''