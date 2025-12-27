import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

def reduce_dimensions(embeddings, n_components=100, pca_model=None):
    """
    Reduces dimensions with PCA for risk density calculation.
    Returns: (reduced_embeddings, pca_model)
    """
    print(f"Reducing dimensions with PCA ({n_components} components)...")
    if embeddings.shape[1] <= n_components:
        print(f"Warning: Embeddings dimension ({embeddings.shape[1]}) <= n_components ({n_components}). Skipping PCA.")
        return embeddings, None
        
    if pca_model is None:
        pca_model = PCA(n_components=n_components)
        reduced_data = pca_model.fit_transform(embeddings)
    else:
        reduced_data = pca_model.transform(embeddings)
        
    return reduced_data, pca_model

def calculate_risk_density(query_embeddings, reference_embeddings, reference_labels, k=100):
    """
    Calculates the 'Risk Density' for each item in query_embeddings based on its 
    k-nearest neighbors in reference_embeddings.
    
    Args:
        query_embeddings: Vectors to score.
        reference_embeddings: Vectors to search against (the "Teacher").
        reference_labels: Binary labels (1=Risk, 0=Safe) for the reference_embeddings.
        k: Number of neighbors.
    """
    print(f"\n--- Calculating Weighted Risk Density (k={k}) ---")
    
    # 1. Build Index on Reference Set
    nbrs = NearestNeighbors(n_neighbors=k, metric='cosine', n_jobs=-1)
    nbrs.fit(reference_embeddings)
    
    print("Querying Neighbors...")
    # Find k neighbors for each query vector
    distances, indices = nbrs.kneighbors(query_embeddings)
    
    risk_scores = []
    
    # Pre-fetch binary labels
    # indices shape: (n_query, k)
    all_neighbor_labels = reference_labels[indices]
    
    # 2. Compute Weighted Score per Item
    for i in range(len(query_embeddings)):
        dists = distances[i]
        lbls = all_neighbor_labels[i]
        
        # Inverse Distance Weighting
        # Add epsilon to avoid division by zero (if distance is 0)
        weights = 1.0 / (dists + 1e-6)
        weights = weights / np.sum(weights)
        
        risk_score = np.sum(weights * lbls)
        risk_scores.append(risk_score)
        
    return np.array(risk_scores)
