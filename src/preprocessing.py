"""
Helper functions for the diabetes project.

These functions load the data, remove duplicates, create new features, split the data
and build the preprocessing pipeline. The notebook imports them, so the same code is
used in every experiment and the results can be repeated.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "Diabetes_binary"
SEED = 42

# Yes/no questions already coded as 0 or 1: they need no scaling.
BINARY_COLUMNS = ["HighBP", "HighChol", "CholCheck", "Smoker", "Stroke", "HeartDiseaseorAttack",
                  "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare",
                  "NoDocbcCost", "DiffWalk", "Sex"]

# Ordered numbers (health score, days, age group, education, income): these are scaled.
# They are already numbers, so no one-hot encoding is needed.
ORDINAL_COLUMNS = ["GenHlth", "MentHlth", "PhysHlth", "Age", "Education", "Income"]

# Columns created by add_features()
ENGINEERED_COLUMNS = ["bmi_category", "lifestyle_risk_score", "condition_count", "poor_health_days"]


def load_data(path):
    """Read the raw CSV file into a DataFrame."""
    return pd.read_csv(path)


def remove_duplicates(df):
    """
    Drop rows that are exact copies of an earlier row.
    Done BEFORE splitting, so the same person-profile cannot be in both train and test.
    Returns the cleaned DataFrame and how many rows were removed.
    """
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    return df, before - len(df)


def add_features(df):
    """
    Create new columns from the existing answers. Each new value only depends on the same
    row, so this cannot leak information from the test set.
    """
    out = df.copy()

    # BMI group using the WHO cut-offs: 0 underweight, 1 normal, 2 overweight, 3 obese
    out["bmi_category"] = pd.cut(out["BMI"], bins=[0, 18.5, 25, 30, 1000],
                                 labels=[0, 1, 2, 3], right=False).astype(int)

    # Lifestyle risk score from 0 (healthy habits) to 4: smoker, no physical activity,
    # no daily fruit, no daily vegetables. (Heavy drinking is left out on purpose:
    # in this data it goes with LOWER diabetes rates, so calling it a "risk" would be wrong.)
    out["lifestyle_risk_score"] = (out["Smoker"] + (1 - out["PhysActivity"])
                                   + (1 - out["Fruits"]) + (1 - out["Veggies"]))

    # How many of these health conditions the person reports (0 to 5)
    out["condition_count"] = out[["HighBP", "HighChol", "Stroke", "HeartDiseaseorAttack", "DiffWalk"]].sum(axis=1)

    # Total days of poor mental + physical health in the last 30 days (0 to 60)
    out["poor_health_days"] = out["MentHlth"] + out["PhysHlth"]
    return out


def split_data(df, seed=SEED):
    """
    Split into train (70%), validation (15%) and test (15%).
    'stratify' keeps the same share of diabetes cases in all three parts.
    """
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.30, stratify=y, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=0.50, stratify=y_rest, random_state=seed)
    return X_train, X_val, X_test, y_train, y_val, y_test


class PercentileCapper(BaseEstimator, TransformerMixin):
    """
    Limit extreme values to chosen percentiles (here the 1st and 99th) instead of deleting rows.
    The limits are learned in fit(), so with the pipeline they come from the TRAINING data only.
    """

    def __init__(self, lower=1, upper=99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.low_ = np.percentile(X, self.lower, axis=0)
        self.high_ = np.percentile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.low_, self.high_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features)


def build_preprocessor(use_engineered=True):
    """
    Build the preprocessing pipeline (not fitted yet):
      - BMI: cap extreme values, then standardise
      - ordered columns (and engineered columns): standardise
      - yes/no columns: keep as they are
    """
    ordinal = ORDINAL_COLUMNS + (ENGINEERED_COLUMNS if use_engineered else [])
    return ColumnTransformer(
        transformers=[
            ("bmi", Pipeline([("cap", PercentileCapper(1, 99)), ("scale", StandardScaler())]), ["BMI"]),
            ("ordinal", StandardScaler(), ordinal),
            ("binary", "passthrough", BINARY_COLUMNS),
        ],
        verbose_feature_names_out=False,
    )


def transform_to_frame(preprocessor, X):
    """Apply a fitted preprocessor and keep the column names (easier to read and debug)."""
    return pd.DataFrame(preprocessor.transform(X), columns=preprocessor.get_feature_names_out(), index=X.index)
