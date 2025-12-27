import os
import sys
import pandas as pd
import json

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import DATA_DIR

INPUT_FILE = os.path.join(DATA_DIR, "lmsys_silver_labels.pkl")

def main():
    print("--- Validating Gold Set (Domain Transfer Check) ---")
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run inference first.")
        return

    df = pd.read_pickle(INPUT_FILE)
    print(f"Loaded {len(df)} silver labels.")
    
    # Select 50 random samples
    sample = df.sample(n=50, random_state=42)
    
    print(f"\n{'TEXT':<80} | {'TOP LABEL':<20} | {'CONF'}")
    print("-" * 110)
    
    for _, row in sample.iterrows():
        text = row['text'].replace('\n', ' ')
        label = row['top_label']
        conf = row['confidence']
        
        print(f"{text[:78]:<80} | {label:<20} | {conf:.1%}")

if __name__ == "__main__":
    main()

