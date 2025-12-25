import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_audit_results(input_file):
    """Loads the audit results PKL and standardizes risk_density."""
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"{input_file} not found.")
    
    # Check extension to decide, though we prefer pkl now
    if input_file.endswith('.pkl'):
        df = pd.read_pickle(input_file)
    else:
        # Fallback if someone still passes csv
        df = pd.read_csv(input_file)
    
    # Ensure risk_density is float and present
    if 'risk_density' not in df.columns:
        if 'purity_score' in df.columns:
            print("Warning: 'risk_density' not found, using 'purity_score' as proxy.")
            df['risk_density'] = df['purity_score']
    
    df['risk_density'] = pd.to_numeric(df['risk_density'], errors='coerce').fillna(0.0)
    return df

def create_pools(df):
    """Separates the dataframe into Risk and Safe pools based on dataset_type."""
    is_risk_source = df['dataset_type'].astype(str).str.contains('mental_health', na=False)
    is_safe_source = df['dataset_type'].astype(str).str.contains('safe', na=False)

    pool_risk = df[is_risk_source].copy()
    pool_risk['label'] = 1
    
    pool_safe = df[is_safe_source].copy()
    pool_safe['label'] = 0
    
    return pool_risk, pool_safe

def compile_risk_set(train_risk, oversample_factor=1.5):
    """
    Oversamples risk items, giving higher weight to 'hard' positives (those looking like safe items).
    """
    risk_densities = train_risk['risk_density'].values
    # Ambiguity Score: 1.0 means "Looks completely Safe". 0.0 means "Looks completely Risk".
    ambiguity = 1.0 - risk_densities
    
    target_risk_count = int(len(train_risk) * oversample_factor)
    
    # Weights: Base + Ambiguity^2 (Emphasize really hard cases)
    risk_weights = 0.2 + (ambiguity ** 2)
    risk_weights = risk_weights / risk_weights.sum()
    
    train_risk_oversampled = train_risk.sample(n=target_risk_count, replace=True, weights=risk_weights, random_state=42)
    return train_risk_oversampled

def compile_safe_set(train_safe, target_count):
    """
    Samples safe items, giving higher weight to 'hard' negatives (those looking like risk items).
    """
    safe_densities = train_safe['risk_density'].values
    
    # Weights: Base + RiskDensity (Emphasize hard cases)
    safe_weights = 0.1 + safe_densities
    safe_weights = safe_weights / safe_weights.sum()
    
    if len(train_safe) > target_count:
        train_safe_sampled = train_safe.sample(n=target_count, replace=False, weights=safe_weights, random_state=42)
    else:
        print("Warning: Oversampling Safe data to meet ratio.")
        train_safe_sampled = train_safe.sample(n=target_count, replace=True, weights=safe_weights, random_state=42)
        
    return train_safe_sampled

def create_ambiguous_test_set(test_risk, test_safe, size=1000):
    """Creates a test set of items closest to the decision boundary (risk_density ~ 0.5)."""
    test_all = pd.concat([test_risk, test_safe])
    dist_from_boundary = np.abs(test_all['risk_density'] - 0.5)
    test_all['boundary_dist'] = dist_from_boundary
    
    test_ambiguous = test_all.sort_values('boundary_dist', ascending=True).head(size).sample(frac=1, random_state=42)
    return test_ambiguous

