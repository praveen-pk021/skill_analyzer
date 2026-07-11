import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "Placement_Data_Full_Class.csv"
MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "placement_model.pkl"
STATS_PATH = Path(__file__).resolve().parents[1] / "model" / "placed_cohort_stats.json"


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    df["status"] = df["status"].map({"Placed": 1, "Not Placed": 0})
    df = df[[
        "gender",
        "ssc_p",
        "hsc_p",
        "degree_p",
        "etest_p",
        "mba_p",
        "specialisation",
        "workex",
        "status",
    ]].copy()
    features = df.drop(columns=["status"])
    target = df["status"]
    return features, target


def build_model() -> Pipeline:
    features, target = prepare_features(pd.read_csv(DATASET_PATH))
    numeric_features = ["ssc_p", "hsc_p", "degree_p", "etest_p", "mba_p"]
    categorical_features = ["gender", "specialisation", "workex"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=200, random_state=42)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.3f}")
    print("Classification report:")
    print(classification_report(y_test, predictions))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    model_path = MODEL_PATH
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as handle:
        pickle.dump(model, handle)

    placed_df = pd.read_csv(DATASET_PATH)
    placed_df = placed_df[placed_df["status"] == "Placed"].copy()
    stats = {
        feature: float(placed_df[feature].mean())
        for feature in ["ssc_p", "hsc_p", "degree_p", "etest_p", "mba_p"]
    }
    with STATS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    return model


def load_model() -> Pipeline:
    with MODEL_PATH.open("rb") as handle:
        return pickle.load(handle)


def load_stats() -> Dict[str, float]:
    with STATS_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def predict_probability(model: Pipeline, profile: Dict[str, object]) -> float:
    df = pd.DataFrame([profile])
    df = df[[
        "gender",
        "ssc_p",
        "hsc_p",
        "degree_p",
        "etest_p",
        "mba_p",
        "specialisation",
        "workex",
    ]]
    probability = model.predict_proba(df)[0, 1]
    return float(probability)


def build_gap_report(user_profile: Dict[str, object], placed_stats: Dict[str, float], top_n: int = 3) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for feature, placed_mean in placed_stats.items():
        user_value = float(user_profile.get(feature, 0.0))
        gap = max(0.0, placed_mean - user_value)
        rows.append({
            "feature": feature,
            "placed_mean": placed_mean,
            "user_value": user_value,
            "gap": gap,
        })

    rows.sort(key=lambda item: item["gap"], reverse=True)
    return rows[:top_n]
