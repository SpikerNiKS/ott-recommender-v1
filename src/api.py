from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from typing import List
from pydantic import BaseModel
from src.candidate_generator import generate_candidates

class RecommendationInfo(BaseModel):
    content_id: int
    title: str
    score: float

app = FastAPI(title="OTT Recommender Inference API")

# Load our newly trained model on startup
try:
    model = joblib.load("models/ranking_model.pkl")
except Exception as e:
    model = None

# To accurately run inference we ideally need the underlying user and content data,
# but for the scope of this project we'll simulate an active session DataFrame for the user 
# using our interactions dataframe as a proxy dataset since this is an offline demo.
try:
    df_all = pd.read_csv("data/processed/interactions.csv")
except Exception as e:
    df_all = pd.DataFrame()


@app.get("/health")
def health_check():
    """
    Standard health check endpoint.
    """
    return {"status": "ok"}

@app.get("/recommend", response_model=List[RecommendationInfo])
def recommend(user_id: int, top_k: int = 3):
    """
    Returns the top K recommended content IDs for a given user.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Ranking model not loaded.")
        
    if df_all.empty:
        raise HTTPException(status_code=500, detail="Data unavailable for candidate generation.")

    # Get all interactions for this particular user to build their pseudo-session context
    user_df = df_all[df_all["user_id"] == user_id]
    
    if user_df.empty:
         raise HTTPException(status_code=404, detail=f"No data found for user_id={user_id}")
         
    # Take a particular session for that user (arbitrary latest session logic)
    session_id = user_df["session_id"].iloc[-1]
    session_df = user_df[user_df["session_id"] == session_id]
    
    # 1. Candidate Generation
    candidates = generate_candidates(session_df, top_n=10)
    
    if candidates.empty:
        return []

    # 2. Feature Engineering Logic (simulated pipeline matching)
    df_feat = candidates.copy()
    df_feat["is_free"] = (df_feat["price"] == 0).astype(int)
    df_feat["genre_popularity"] = df_feat["genre_match"] * df_feat["popularity"]
    df_feat["popularity_norm"] = df_feat["popularity"] / (df_all["popularity"].max() + 1e-9)
    
    features = [
        "genre_match",
        "popularity_norm",
        "runtime",
        "price",
        "is_free",
        "genre_popularity",
        "time_of_day"
    ]
    
    X = df_feat[features]
    
    # 3. Model Scoring
    df_feat["ml_score"] = model.predict_proba(X)[:, 1]
    
    # 4. Return Top-K
    ranked = df_feat.sort_values("ml_score", ascending=False)
    top_items = ranked.head(top_k)
    
    # Format the structured response
    recs = []
    
    # Simple list of mock genres to make pseudo-realistic titles
    mock_genres = ["Fantasy", "Sci-Fi", "Action", "Drama", "Comedy", "Thriller", "Romance", "Adventure", "Documentary"]
    
    for _, row in top_items.iterrows():
        c_id = int(row["content_id"])
        
        # Create a stable mock title based on the content ID
        mock_title = f"{mock_genres[c_id % len(mock_genres)]} Movie {c_id}"
        
        recs.append(RecommendationInfo(
            content_id=c_id,
            title=mock_title,
            score=round(float(row["ml_score"]), 2)
        ))
        
    return recs
