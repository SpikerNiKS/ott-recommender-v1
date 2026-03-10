import pandas as pd

df = pd.read_csv("data/processed/interactions.csv")

# compute user genre preference from clicked items
user_profiles = (
    df[df["clicked"] == 1]
    .groupby("user_id")["genre_match"]
    .mean()
)

user_profiles.to_csv("data/processed/user_profiles.csv")

print("User profiles generated")