# OTT Recommender System (ML Ranking Pipeline)

An end-to-end machine learning pipeline that simulates and evaluates a recommendation system for OTT platforms (movies / series).
The project demonstrates how modern recommender systems use **candidate generation + ML ranking** to recommend content.

This repository contains:

* Synthetic user session generation
* Candidate generation stage
* Feature engineering
* ML ranking model
* Offline evaluation metrics
* Model artifact management

The goal is to replicate the architecture used by modern streaming platforms in a simplified experimental environment.

---

# Architecture Overview

The system follows a **two-stage recommender architecture**.

User Session
→ Candidate Generation
→ Feature Engineering
→ ML Ranking Model
→ Top-K Recommendations
→ Evaluation

This architecture is commonly used in large-scale recommender systems.

---

# Project Structure

```
ott-recommender-v1
│
├── data/
│   └── processed/
│       └── interactions.csv
│
├── models/
│   └── ranking_model.pkl
│
├── src/
│   ├── data_generation.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── candidate_generator.py
│   ├── user_profiles.py
│
├── requirements.txt
└── README.md
```

---

# Dataset

The dataset is synthetically generated to simulate OTT user sessions.

Each session contains multiple candidate items.

Columns:

| Column           | Description                               |
| ---------------- | ----------------------------------------- |
| session_id       | session identifier                        |
| user_id          | simulated user                            |
| content_id       | movie/series ID                           |
| genre_match      | similarity between user taste and content |
| popularity       | content popularity score                  |
| runtime          | duration of content                       |
| price            | cost to watch                             |
| time_of_day      | viewing time context                      |
| clicked          | whether the user clicked the item         |
| return_gap_hours | proxy for engagement                      |

---

# Feature Engineering

Additional features are created before model training:

* **is_free** → whether content is free
* **genre_popularity** → interaction between genre affinity and popularity
* **popularity_norm** → normalized popularity score

These features help the ranking model learn user preferences.

---

# Model

The ranking model is implemented using **RandomForestClassifier** inside a scikit-learn pipeline.

Pipeline components:

* OneHotEncoding for categorical features
* Standard scaling for numerical features
* RandomForest ranking model

The model predicts:

```
P(clicked | features)
```

Items are ranked by predicted probability.

---

# Evaluation Metric

Primary metric:

```
Precision@3
```

This measures how often the clicked item appears in the top-3 recommendations.

Example:

```
Precision@3 = 0.286
```

Meaning the model successfully ranks the clicked item in the top 3 about **28.6% of the time**.

---

# Installation

Clone the repository and install dependencies.

```
git clone <repo-url>
cd ott-recommender-v1

pip install -r requirements.txt
```

---

# Running the Pipeline

Step 1 — Generate dataset

```
python src/data_generation.py
```

Step 2 — Train ranking model

```
python src/train_model.py
```

Step 3 — Evaluate recommender

```
python src/evaluate_model.py
```

---

# Example Output

```
Dataset generated: (15000, 10)
Model trained and saved.
ML Precision@3: 0.286
```

---

# Future Improvements

Possible extensions for this system:

* LightGBM / XGBoost ranking models
* Hyperparameter tuning
* Additional ranking metrics (Recall@K, NDCG)
* Session-aware feature engineering
* REST API inference endpoint
* Online evaluation simulation

---

# Learning Goals

This project demonstrates:

* ranking-based recommendation systems
* feature engineering for recommender models
* candidate generation pipelines
* ML model training and inference separation
* offline recommender evaluation

---

# License

MIT License
