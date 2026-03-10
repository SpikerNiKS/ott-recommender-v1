import numpy as np

def precision_at_k(clicked_list, k):
    """
    Computes Precision@K:
    What proportion of top-K recommendations are relevant?
    """
    if len(clicked_list) == 0:
        return 0.0
    return np.sum(clicked_list[:k]) / k

def recall_at_k(clicked_list, k):
    """
    Computes Recall@K:
    What proportion of ALL relevant items made it into the top-K recommendations?
    """
    total_relevant = np.sum(clicked_list)
    if total_relevant == 0:
        return 0.0
    return np.sum(clicked_list[:k]) / total_relevant

def ndcg_at_k(clicked_list, k):
    """
    Computes NDCG@K (Normalized Discounted Cumulative Gain):
    Accounts for BOTH relevance and the rank/position.
    """
    if len(clicked_list) == 0:
        return 0.0
        
    # Calculate DCG (Discounted Cumulative Gain)
    dcg = 0
    for i, rel in enumerate(clicked_list[:k]):
        dcg += rel / np.log2(i + 2) # i=0 is rank 1 -> log2(2)=1
        
    # Calculate IDCG (Ideal DCG - What the perfect ranking would give)
    ideal = sorted(clicked_list, reverse=True)
    idcg = 0
    for i, rel in enumerate(ideal[:k]):
        idcg += rel / np.log2(i + 2)
        
    return dcg / idcg if idcg > 0 else 0.0
