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

def compile_risk_set(train_risk, min_purity=0.1):
    """
    Filters out Risk items that look too much like Safe items (likely label errors).
    
    Args:
        train_risk: DataFrame containing risk items
        min_purity: Minimum risk_density required to keep the item. 
                   0.1 means at least 10% of neighbors must be Risk.
    """
    initial_count = len(train_risk)
    
    # We want Risk items to look like Risk (high risk_density)
    # If risk_density is 0.05, it means 95% of neighbors are Safe. Likely a bad label.
    train_risk_filtered = train_risk[train_risk['risk_density'] >= min_purity].copy()
    
    dropped_count = initial_count - len(train_risk_filtered)
    if dropped_count > 0:
        print(f"Dropped {dropped_count} Risk items with purity < {min_purity}")
        
    return train_risk_filtered

def compile_safe_set(train_safe, max_risk_density=0.9):
    """
    Filters out Safe items that look too much like Risk items (likely label errors).
    
    Args:
        train_safe: DataFrame containing safe items
        max_risk_density: Maximum risk_density allowed.
                         0.9 means if >90% of neighbors are Risk, we drop this 'Safe' item.
    """
    initial_count = len(train_safe)
    
    # We want Safe items to look Safe (low risk_density)
    train_safe_filtered = train_safe[train_safe['risk_density'] <= max_risk_density].copy()
    
    dropped_count = initial_count - len(train_safe_filtered)
    if dropped_count > 0:
        print(f"Dropped {dropped_count} Safe items with risk_density > {max_risk_density}")
        
    return train_safe_filtered

def create_ambiguous_test_set(test_risk, test_safe, size=1000):
    """Creates a test set of items closest to the decision boundary (risk_density ~ 0.5)."""
    test_all = pd.concat([test_risk, test_safe])
    dist_from_boundary = np.abs(test_all['risk_density'] - 0.5)
    test_all['boundary_dist'] = dist_from_boundary
    
    test_ambiguous = test_all.sort_values('boundary_dist', ascending=True).head(size).sample(frac=1, random_state=42)
    return test_ambiguous

