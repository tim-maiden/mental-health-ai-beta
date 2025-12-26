import os
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def select_active_learning_samples(input_file, output_file, n_samples=500):
    print(f"--- Active Learning Sampler ---")
    
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
        return

    print(f"Loading inference results from {input_file}...")
    df = pd.read_pickle(input_file)
    print(f"Total rows: {len(df)}")
    
    # 1. Filter for Uncertainty
    # We look for risk_scores between 0.2 and 0.8
    # (Items that the model is not confident are Safe (<0.2) or Risk (>0.8))
    mask_uncertain = (df['risk_score'] >= 0.2) & (df['risk_score'] <= 0.8)
    df_uncertain = df[mask_uncertain].copy()
    
    print(f"Uncertain Samples (0.2 <= score <= 0.8): {len(df_uncertain)}")
    
    if len(df_uncertain) == 0:
        print("No uncertain samples found. Exiting.")
        return

    # 2. Diversity Sampling (Clustering)
    # Since we don't have embeddings for these specific inference rows (unless we fetch them),
    # we'll use TF-IDF + KMeans to ensure textual diversity.
    
    target_samples = min(n_samples, len(df_uncertain))
    print(f"Selecting {target_samples} diverse samples...")
    
    # Vectorize
    print("Vectorizing text (TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    # Fill NaNs if any
    texts = df_uncertain['text'].fillna("").astype(str).values
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Cluster
    # We want to pick samples from different clusters.
    # Heuristic: Number of clusters = target_samples // 5 (so we pick ~5 from each)
    n_clusters = max(1, target_samples // 5)
    print(f"Clustering into {n_clusters} clusters...")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(tfidf_matrix)
    
    df_uncertain['cluster'] = clusters
    
    # 3. Stratified Sampling
    # Sample equally from each cluster to ensure diversity
    samples_per_cluster = target_samples // n_clusters
    
    final_samples = df_uncertain.groupby('cluster', group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), samples_per_cluster), random_state=42)
    )
    
    # If we still need more samples (due to rounding or small clusters), fill up from the rest
    if len(final_samples) < target_samples:
        remaining = df_uncertain.drop(final_samples.index)
        n_needed = target_samples - len(final_samples)
        if not remaining.empty:
            extra_samples = remaining.sample(n=min(len(remaining), n_needed), random_state=42)
            final_samples = pd.concat([final_samples, extra_samples])
            
    print(f"Selected {len(final_samples)} samples.")
    
    # 4. Export
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Select useful columns
    out_cols = ['text', 'risk_score', 'confidence', 'label']
    if 'id' in final_samples.columns:
        out_cols.insert(0, 'id')
        
    final_samples[out_cols].to_csv(output_file, index=False)
    print(f"Saved candidates to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select active learning candidates from inference results.")
    parser.add_argument("--input", type=str, required=True, help="Path to inference output (.pkl)")
    parser.add_argument("--output", type=str, default="data/active_learning_candidates.csv", help="Output CSV path")
    parser.add_argument("--n", type=int, default=500, help="Number of samples to select")
    
    args = parser.parse_args()
    
    select_active_learning_samples(args.input, args.output, args.n)

