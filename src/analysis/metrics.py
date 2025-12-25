import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

def reduce_dimensions(embeddings, n_components=50):
    """
    Reduces dimensions with PCA for risk density calculation.
    """
    print(f"Reducing dimensions with PCA ({n_components} components)...")
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(embeddings)
    return reduced_data

def calculate_risk_density(embeddings, binary_labels, k=100):
    """
    Calculates the 'Risk Density' for each item based on its k-nearest neighbors.
    """
    print(f"\n--- Calculating Weighted Risk Density (k={k}) ---")
    
    # 1. Build Index
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='cosine', n_jobs=-1)
    nbrs.fit(embeddings)
    
    print("Querying Neighbors...")
    distances, indices = nbrs.kneighbors(embeddings)
    
    risk_scores = []
    
    # Pre-fetch binary labels
    all_neighbor_labels = binary_labels[indices]
    
    # 2. Compute Weighted Score per Item
    for i in range(len(embeddings)):
        dists = distances[i, 1:]
        lbls = all_neighbor_labels[i, 1:]
        
        # Inverse Distance Weighting
        weights = 1.0 / (dists + 1e-6)
        weights = weights / np.sum(weights)
        
        risk_score = np.sum(weights * lbls)
        risk_scores.append(risk_score)
        
    return np.array(risk_scores)
