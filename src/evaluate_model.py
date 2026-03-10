import pandas as pd
import joblib
import numpy as np

from candidate_generator import generate_candidates

# load dataset
df = pd.read_csv("data/processed/interactions.csv")

# feature engineering (must match training)
df["is_free"] = (df["price"] == 0).astype(int)
df["genre_popularity"] = df["genre_match"] * df["popularity"]
df["popularity_norm"] = df["popularity"] / df["popularity"].max()

# load trained model
model = joblib.load("models/ranking_model.pkl")

# feature set
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

# predict scores
df["ml_score"] = model.predict_proba(X)[:,1]

top_k = 3
precision_list = []

for session in df["session_id"].unique():

    session_df = df[df["session_id"] == session]

    # candidate generation
    candidates = generate_candidates(session_df, top_n=5)

    # ranking
    ranked = candidates.sort_values("ml_score", ascending=False)

    top_items = ranked.head(top_k)

    precision = top_items["clicked"].sum() / top_k

    precision_list.append(precision)

precision_at_3 = np.mean(precision_list)

print("ML Precision@3:", round(precision_at_3,4))

metrics = {
    "precision_at_3": round(precision_at_3, 4)
}

pd.DataFrame([metrics]).to_csv("results/metrics.csv", index=False)
print("Metrics saved to results/metrics.csv")