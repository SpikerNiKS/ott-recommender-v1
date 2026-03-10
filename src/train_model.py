import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


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
        ("clf", RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            random_state=42
        ))
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

joblib.dump(model, "models/ranking_model.pkl")

print("Model trained and saved.")