import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier


df = pd.read_csv("data/processed/interactions.csv")

# feature engineering
df["is_free"] = (df["price"] == 0).astype(int)
df["genre_popularity"] = df["genre_match"] * df["popularity"]
df["popularity_norm"] = df["popularity"] / df["popularity"].max()

# features used by model
X = df[
[
"genre_match",
"popularity_norm",
"runtime",
"price",
"is_free",
"genre_popularity",
"time_of_day"
]
]

y = df["clicked"]

categorical = ["time_of_day"]

numeric = [
"genre_match",
"popularity_norm",
"runtime",
"price",
"is_free",
"genre_popularity"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", StandardScaler(), numeric)
    ]
)

model = Pipeline(
    steps=[
        ("prep", preprocessor),
        ("clf", LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=8,
            random_state=42
        ))
    ]
)

# group by session to prevent data leakage
session_ids = df["session_id"].unique()
train_sessions, test_sessions = train_test_split(
    session_ids, test_size=0.2, random_state=42
)

train_df = df[df["session_id"].isin(train_sessions)]
test_df = df[df["session_id"].isin(test_sessions)]

# save test_df for evaluation
test_df.to_csv("data/processed/test_interactions.csv", index=False)

X_train = train_df[numeric + categorical]
y_train = train_df["clicked"]

model.fit(X_train, y_train)

joblib.dump(model, "models/ranking_model.pkl")

print("Model trained and saved.")