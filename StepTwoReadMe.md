# OTT Recommender System — Improvement Roadmap (Next Development Phase)

## Purpose

This document outlines the **next set of improvements** for the OTT Recommender System project.
The goal is to evolve the current prototype into a **more production-like ML system** by improving evaluation, modeling, data handling, and deployment readiness.

This document can be used as a **task specification for the next development iteration**.

---

# Current System Overview

The current pipeline works as follows:

```
Synthetic Data Generation
        ↓
Session-based Dataset
        ↓
Candidate Generation
        ↓
Feature Engineering
        ↓
RandomForest Ranking Model
        ↓
Precision@3 Evaluation
        ↓
Pipeline Execution (main.py)
```

The system already includes:

* Synthetic dataset generation
* Session-based recommendation candidates
* Candidate generator
* Feature engineering
* Ranking model
* Precision@K evaluation
* Model artifact saving
* Automated pipeline execution

---

# Target System Architecture

The improved system should evolve toward the following architecture:

```
Data Generation
      ↓
Feature Engineering Layer
      ↓
Candidate Generation
      ↓
Ranking Model (LightGBM / Boosted Trees)
      ↓
Evaluation (Precision / Recall / NDCG)
      ↓
Experiment Tracking
      ↓
Inference API
```

---

# Improvement Tasks

## 1. Implement Additional Ranking Metrics

### Current Limitation

Only **Precision@3** is used.

### Why This Matters

Precision alone does not capture the quality of the ranking order.

For example:

```
Candidate list size = 10
Clicked item = rank 4
```

Precision@3 = 0
But the recommendation is still reasonable.

Ranking metrics should include:

* Precision@K
* Recall@K
* NDCG@K
* HitRate@K
* MAP@K

### Implementation

Create:

```
src/metrics.py
```

Example implementation:

```python
import numpy as np

def precision_at_k(clicked_list, k):
    return np.sum(clicked_list[:k]) / k

def recall_at_k(clicked_list, k):
    return np.sum(clicked_list[:k]) / np.sum(clicked_list)

def ndcg_at_k(clicked_list, k):

    dcg = 0
    for i, rel in enumerate(clicked_list[:k]):
        dcg += rel / np.log2(i + 2)

    ideal = sorted(clicked_list, reverse=True)
    idcg = 0
    for i, rel in enumerate(ideal[:k]):
        idcg += rel / np.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0
```

Update `evaluate_model.py` to compute these metrics.

---

# 2. Upgrade Ranking Model

### Current Model

```
RandomForestClassifier
```

### Limitation

RandomForest is often outperformed by **gradient boosting models** in ranking problems.

### Recommended Models

```
LightGBM
XGBoost
CatBoost
```

LightGBM is recommended for this project.

### Installation

```
pip install lightgbm
```

### Example Model

```python
from lightgbm import LGBMClassifier

model = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=8
)
```

---

# 3. Introduce Session-Based Train/Test Split

### Current Problem

Training and evaluation use the **same dataset**, causing data leakage.

### Correct Approach

Split by **session_id**.

Example:

```python
train_sessions = df["session_id"].sample(frac=0.8)

train_df = df[df["session_id"].isin(train_sessions)]
test_df = df[~df["session_id"].isin(train_sessions)]
```

Then:

```
Train model on train_df
Evaluate model on test_df
```

This produces more reliable metrics.

---

# 4. Add Experiment Tracking

### Current Limitation

Metrics are printed to the console.

Example:

```
Precision@3 = 0.286
```

### Recommended Improvement

Store results in a **results directory**.

Example structure:

```
results/
    experiment_001.csv
    experiment_002.csv
```

Example code:

```python
metrics = {
    "precision@3": precision,
    "recall@3": recall,
    "ndcg@3": ndcg
}

pd.DataFrame([metrics]).to_csv("results/metrics.csv", index=False)
```

This allows tracking improvements over time.

---

# 5. Implement a Recommendation API

### Goal

Expose the ranking model via an **inference API**.

Example flow:

```
User Request
     ↓
Candidate Generator
     ↓
Ranking Model
     ↓
Return Top-K Recommendations
```

### Recommended Framework

FastAPI

Installation:

```
pip install fastapi uvicorn
```

Example endpoint:

```
GET /recommend?user_id=42
```

Response:

```
[
  "content_12",
  "content_87",
  "content_53"
]
```

---

# Implementation Priority

Recommended order of implementation:

1️⃣ Add ranking metrics (Precision / Recall / NDCG)
2️⃣ Replace RandomForest with LightGBM
3️⃣ Implement session-based train/test split
4️⃣ Add experiment result tracking
5️⃣ Build inference API

---

# Expected Outcome

After implementing these improvements, the system will include:

* Candidate generation stage
* Feature engineering layer
* Gradient boosting ranking model
* Proper ranking metrics
* Session-safe evaluation
* Experiment tracking
* Inference API

This will transform the project from a **prototype recommender** into a **more production-style ML system**.

---

# Notes for Next Development Iteration

* Maintain reproducible pipeline execution through `main.py`
* Ensure feature engineering is shared between training and inference
* Avoid data leakage between training and evaluation
* Track experiments to measure model improvements

---

End of Document
