import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, PowerTransformer, StandardScaler


TEST_CSV = "test_trial.csv"

vitamin_cols = [
    "vitamin_a_percent_rda",
    "vitamin_c_percent_rda",
    "vitamin_d_percent_rda",
    "vitamin_e_percent_rda",
    "vitamin_b12_percent_rda",
    "folate_percent_rda",
    "calcium_percent_rda",
    "iron_percent_rda",
]
pt_cols = vitamin_cols + ["symptoms_count"]


def encode_data(df):
    df = df.copy()
    df = pd.get_dummies(df, columns=["gender"])

    df["smoking_status"] = df["smoking_status"].map(
        {"Never": 0, "Former": 1, "Current": 2}
    )
    df["alcohol_consumption"] = df["alcohol_consumption"].replace(
        {"None": np.nan, "none": np.nan, "": np.nan}
    )
    df["alcohol_consumption"] = df["alcohol_consumption"].map(
        {"Moderate": 0, "Heavy": 1}
    )
    df["exercise_level"] = df["exercise_level"].map(
        {"Sedentary": 0, "Light": 1, "Moderate": 2, "Active": 3}
    )
    df = pd.get_dummies(df, columns=["diet_type"])
    df["sun_exposure"] = df["sun_exposure"].map({"Low": 0, "Moderate": 1, "High": 2})
    df["income_level"] = df["income_level"].map({"Low": 0, "Middle": 1, "High": 2})
    df["latitude_region"] = df["latitude_region"].map({"Low": 0, "Mid": 1, "High": 2})

    df.drop(columns=["symptoms_list"], inplace=True, errors="ignore")
    df.drop(
        columns=[
            "gender_Male",
            "gender_Female",
            "age",
            "bmi",
            "smoking_status",
            "exercise_level",
            "latitude_region",
        ],
        inplace=True,
        errors="ignore",
    )
    return df

if not os.path.isfile("preprocessors.pkl"):
    print("Building preprocessors.pkl from train_data.csv ...")
    train_df = pd.read_csv("train_data.csv")

    le = LabelEncoder()
    y = le.fit_transform(train_df["disease_diagnosis"])
    X = encode_data(train_df.drop(columns=["disease_diagnosis"]))

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clip_bounds = {}
    for col in vitamin_cols:
        q1 = x_train[col].quantile(0.25)
        q3 = x_train[col].quantile(0.75)
        iqr = q3 - q1
        clip_bounds[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        x_train[col] = x_train[col].clip(clip_bounds[col][0], clip_bounds[col][1])
        x_test[col] = x_test[col].clip(clip_bounds[col][0], clip_bounds[col][1])

    alcohol_mode = x_train["alcohol_consumption"].mode()[0]
    x_train["alcohol_consumption"] = x_train["alcohol_consumption"].fillna(alcohol_mode)
    x_test["alcohol_consumption"] = x_test["alcohol_consumption"].fillna(alcohol_mode)

    pt = PowerTransformer(standardize=False)
    x_train[pt_cols] = pt.fit_transform(x_train[pt_cols])
    x_test[pt_cols] = pt.transform(x_test[pt_cols])

    scaler = StandardScaler()
    x_train[vitamin_cols] = scaler.fit_transform(x_train[vitamin_cols])
    x_test[vitamin_cols] = scaler.transform(x_test[vitamin_cols])

    feature_columns = list(x_train.columns)

    prep = {
        "label_encoder": le,
        "clip_bounds": clip_bounds,
        "alcohol_mode": alcohol_mode,
        "power_transformer": pt,
        "scaler": scaler,
        "feature_columns": feature_columns,
    }

    with open("preprocessors.pkl", "wb") as f:
        pickle.dump(prep, f)
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    print("Saved preprocessors.pkl")


with open("preprocessors.pkl", "rb") as f:
    prep = pickle.load(f)

with open("random_forest.pkl", "rb") as f:
    random_forest = pickle.load(f)

le = prep["label_encoder"]
clip_bounds = prep["clip_bounds"]
alcohol_mode = prep["alcohol_mode"]
pt = prep["power_transformer"]
scaler = prep["scaler"]
feature_columns = prep["feature_columns"]

df = pd.read_csv(TEST_CSV)

y_true = None
if "disease_diagnosis" in df.columns:
    y_true = le.transform(df["disease_diagnosis"])
    df = df.drop(columns=["disease_diagnosis"])

x = encode_data(df)

for col in vitamin_cols:
    low, high = clip_bounds[col]
    x[col] = x[col].clip(low, high)

x["alcohol_consumption"] = x["alcohol_consumption"].fillna(alcohol_mode)
x[pt_cols] = pt.transform(x[pt_cols])
x[vitamin_cols] = scaler.transform(x[vitamin_cols])

for col in feature_columns:
    if col not in x.columns:
        x[col] = 0

x = x[feature_columns]

predictions = random_forest.predict(x)
labels = le.inverse_transform(predictions)

print("Predictions:", labels)

if y_true is not None:
    correct = (predictions == y_true).sum()
    print("Correct:", correct, "of", len(y_true))
    print("Accuracy %:", round(correct / len(y_true) * 100, 2))
