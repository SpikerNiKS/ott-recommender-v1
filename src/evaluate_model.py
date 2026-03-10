import pandas as pd
import joblib
import numpy as np

from candidate_generator import generate_candidates
from metrics import precision_at_k, recall_at_k, ndcg_at_k

# load dataset
df = pd.read_csv("data/processed/test_interactions.csv")

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
recall_list = []
ndcg_list = []

for session in df["session_id"].unique():

    session_df = df[df["session_id"] == session]

    # candidate generation
    candidates = generate_candidates(session_df, top_n=5)

    # ranking
    ranked = candidates.sort_values("ml_score", ascending=False)

    top_items = ranked.head(top_k)

    # use actual clicked values from session to compare
    clicked_list = candidates["clicked"].tolist()
    # rank them according to our model's sort output
    ranked_clicked_list = ranked["clicked"].tolist()

    precision = precision_at_k(ranked_clicked_list, top_k)
    recall = recall_at_k(ranked_clicked_list, top_k)
    ndcg = ndcg_at_k(ranked_clicked_list, top_k)

    precision_list.append(precision)
    recall_list.append(recall)
    ndcg_list.append(ndcg)

precision_at_3 = np.mean(precision_list)
recall_at_3 = np.mean(recall_list)
ndcg_at_3 = np.mean(ndcg_list)

print("ML Precision@3:", round(precision_at_3,4))
print("ML Recall@3:", round(recall_at_3,4))
print("ML NDCG@3:", round(ndcg_at_3,4))

metrics = {
    "precision_at_3": round(precision_at_3, 4),
    "recall_at_3": round(recall_at_3, 4),
    "ndcg_at_3": round(ndcg_at_3, 4)
}

import os
os.makedirs("results", exist_ok=True)
pd.DataFrame([metrics]).to_csv("results/metrics.csv", index=False)
print("Metrics saved to results/metrics.csv")