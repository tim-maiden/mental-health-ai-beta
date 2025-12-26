import os
import sys
import pandas as pd
import numpy as np
# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import AUDIT_RESULTS_FILE, RAW_DATA_FILE
from src.analysis.metrics import reduce_dimensions, calculate_risk_density

def main():
    os.makedirs("outputs", exist_ok=True)
    
    # 1. Load Data from Snapshot
    if not os.path.exists(RAW_DATA_FILE):
        print(f"Error: Snapshot file {RAW_DATA_FILE} not found.")
        print("Please run scripts/ingest_data.py first.")
        sys.exit(1)
        
    print(f"Loading data from {RAW_DATA_FILE}...")
    # Read pickle (preserves types, no need to parse strings)
    df_all = pd.read_pickle(RAW_DATA_FILE)
    
    # Label them for binary classification (Risk vs Control)
    # Assumes dataset_type contains 'mental_health' for risk
    df_all['binary_label'] = df_all['dataset_type'].apply(lambda x: 1 if 'mental_health' in str(x) else 0)
    
    embeddings = np.stack(df_all['embedding_vec'].values)
    binary_labels = df_all['binary_label'].values
    
    # 2. Dimensionality Reduction
    reduced_data = reduce_dimensions(embeddings)
    
    # 3. Significance Filter (Risk Density)
    # Self-querying: query=reduced_data, reference=reduced_data
    risk_scores = calculate_risk_density(
        query_embeddings=reduced_data,
        reference_embeddings=reduced_data,
        reference_labels=binary_labels,
        k=100
    )
    
    df_all['risk_density'] = risk_scores
    
    print("\n--- Risk Density Audit ---")
    print(f"Risk Density calculated for {len(df_all)} items.")
    
    # 4. Save/Report results
    output_file = AUDIT_RESULTS_FILE
    df_all[['subreddit', 'input', 'dataset_type', 'risk_density']].to_pickle(output_file)
    print(f"\nDetailed results saved to {output_file}")

if __name__ == "__main__":
    main()
