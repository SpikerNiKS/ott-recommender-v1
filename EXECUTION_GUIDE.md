# Execution Guide for New Collaborators

Welcome to the **OTT Recommender System** project! This document outlines how the pipeline works and provides step-by-step instructions on setting up and running the code locally. This is particularly useful if you are new to the codebase.

## Prerequisites

Ensure you have Python installed on your system. We also highly recommend using a virtual environment to manage dependencies locally.

## 1. Environment Setup

First, navigate to the project directory:

```bash
cd ott-recommender-v1
```

Next, create and activate a virtual environment:

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## 2. Running the End-to-End Pipeline

The project is structured into individual modules for data generation, model training, and evaluation. You can run the entire pipeline in one go using the provided main script.

```bash
python main.py
```

### Under the Hood: The `main.py` Workflow

When you run `main.py`, it executes the following modules sequentially:

1. **`src/data_generation.py`**: Constructs a synthetic dataset simulating OTT user interactions (movie/series candidate generation) and outputs the results to `data/processed/interactions.csv`.
2. **`src/train_model.py`**: Reads the processed dataset, engineers features (e.g., `genre_popularity`), trains a Random Forest ranking model, and saves the artifact as `models/ranking_model.pkl`.
3. **`src/user_profiles.py`**: Processes and generates simulated user profile features needed for the recommender logic.
4. **`src/evaluate_model.py`**: Loads the locally saved test sets and model artifacts to calculate offline metrics (primary metric: `Precision@3`).

## 3. Running Scripts Individually

If you are developing, enhancing, or debugging a specific part of the pipeline, you can run the Python modules standalone. Make sure to run them in logical order:

```bash
# 1. Generate the initial user sessions
python src/data_generation.py

# 2. Train the Random Forest ranking model
python src/train_model.py

# 3. Simulate or build user profiles
python src/user_profiles.py

# 4. Evaluate the trained pipeline
python src/evaluate_model.py
```

## 4. Verifying Outputs

After successful execution of the full pipeline, expect the following changes in your workspace:
- A generated dataset: `data/processed/interactions.csv`
- A serialized model file: `models/ranking_model.pkl`
- Terminal outputs notifying you of the completion and displaying the primary evaluation metric (`Precision@3`).

---

**Tip for Collaborators**: If you are adding a new feature (e.g., a LightGBM ranker or a new evaluation metric), check the relevant `src/` modules. Consider making a corresponding update to this document or `README.md` to reflect architectural changes.
