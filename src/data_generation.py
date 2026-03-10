import pandas as pd
import numpy as np

np.random.seed(42)

NUM_USERS = 200
NUM_CONTENT = 500

sessions = 1500
items_per_session = 10

data = []

for session_id in range(sessions):

    user_id = np.random.randint(1, NUM_USERS)

    candidates = []
    scores = []

    for i in range(items_per_session):

        content_id = np.random.randint(1, NUM_CONTENT)

        genre_match = np.random.rand()
        popularity = np.random.uniform(1, 10)
        runtime = np.random.randint(80, 150)

        price = np.random.choice([0, 99, 149, 199], p=[0.6,0.2,0.1,0.1])
        time_of_day = np.random.choice(["morning","afternoon","night"])

        noise = np.random.normal(0, 0.1)

        score = (
            0.4 * genre_match +
            0.2 * (popularity / 10) +
            0.2 * (1 if price == 0 else 0) +
            0.2 * (1 if time_of_day == "night" else 0) +
            noise
        )

        score = np.clip(score, 0, 1)

        scores.append(score)

        candidates.append([
            session_id,
            user_id,
            content_id,
            genre_match,
            popularity,
            runtime,
            price,
            time_of_day
        ])

    # choose item with highest score
    clicked_index = np.argmax(scores)

    for i in range(items_per_session):

        clicked = 1 if i == clicked_index else 0

        if clicked:
            return_gap = np.random.exponential(24)
        else:
            return_gap = np.random.uniform(0, 1)

        row = candidates[i] + [clicked, return_gap]

        data.append(row)


columns = [
    "session_id",
    "user_id",
    "content_id",
    "genre_match",
    "popularity",
    "runtime",
    "price",
    "time_of_day",
    "clicked",
    "return_gap_hours"
]

df = pd.DataFrame(data, columns=columns)

df.to_csv("data/processed/interactions.csv", index=False)

print("Dataset generated:", df.shape)