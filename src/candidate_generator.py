import pandas as pd

df = pd.read_csv("data/processed/interactions.csv")

def generate_candidates(session_df, top_n=5):

    ranked = session_df.sort_values("genre_match", ascending=False)

    return ranked.head(top_n)