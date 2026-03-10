import pandas as pd
import numpy as np

df = pd.read_csv("data/processed/interactions.csv")

# normalize popularity
df["popularity_norm"] = df["popularity"] / df["popularity"].max()

# cost bonus
df["cost_bonus"] = df["price"].apply(lambda x: 1 if x == 0 else 0)

# baseline score
df["baseline_score"] = (
    0.5 * df["genre_match"]
    + 0.3 * df["popularity_norm"]
    + 0.2 * df["cost_bonus"]
)

# rank top 3 per user
top_k = 3

precision_list = []

for user in df["user_id"].unique():

    user_df = df[df["user_id"] == user]

    ranked = user_df.sort_values("baseline_score", ascending=False)

    top_items = ranked.head(top_k)

    precision = top_items["clicked"].sum() / top_k

    precision_list.append(precision)

precision_at_3 = np.mean(precision_list)

print("Baseline Precision@3:", round(precision_at_3, 4))
print(df.head())
print(df["clicked"].mean())