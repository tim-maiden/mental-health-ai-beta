import os
import sys
# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import load_dotenv
load_dotenv()

from src.core.clients import supabase
import pandas as pd

TABLE = 'reddit_mental_health_embeddings'
START_UTC = 1627776000 # Aug 1 2021
END_UTC = 1661990400   # Aug 31 2022

def main():
    print(f"--- Checking {TABLE} for date consistency ---")
    print(f"Target Range: {START_UTC} to {END_UTC}")

    # 1. Total Count
    try:
        res = supabase.table(TABLE).select('id', count='exact').limit(1).execute()
        count = res.count
        print(f"Total rows in DB: {count}")
    except Exception as e:
        print(f"Error getting count: {e}")
        count = 0

    if count == 0:
        print("Table is empty.")
        return

    # 2. Check Head (Oldest Data)
    print("\nChecking First 100 rows (Oldest)...")
    res_head = supabase.table(TABLE).select('id, created_utc').order('id', desc=False).limit(100).execute()
    df_head = pd.DataFrame(res_head.data)
    
    if not df_head.empty:
        df_head['created_utc'] = pd.to_numeric(df_head['created_utc'], errors='coerce').fillna(0)
        bad_head = df_head[ (df_head['created_utc'] < START_UTC) | (df_head['created_utc'] > END_UTC) ]
        print(f"  > {len(bad_head)} / {len(df_head)} rows are outside date range.")
        if not bad_head.empty:
            print(f"  > Sample bad date: {bad_head.iloc[0]['created_utc']}")
            print("  > CONCLUSION: Old data is CORRUPTED (likely from before filter was active).")
        else:
            print("  > CONCLUSION: Old data is VALID.")
    else:
        print("  > No data returned.")

    # 3. Check Tail (Newest Data)
    print("\nChecking Last 100 rows (Newest)...")
    res_tail = supabase.table(TABLE).select('id, created_utc').order('id', desc=True).limit(100).execute()
    df_tail = pd.DataFrame(res_tail.data)
    
    if not df_tail.empty:
        df_tail['created_utc'] = pd.to_numeric(df_tail['created_utc'], errors='coerce').fillna(0)
        bad_tail = df_tail[ (df_tail['created_utc'] < START_UTC) | (df_tail['created_utc'] > END_UTC) ]
        print(f"  > {len(bad_tail)} / {len(df_tail)} rows are outside date range.")
        if not bad_tail.empty:
            print(f"  > Sample bad date: {bad_tail.iloc[0]['created_utc']}")
            print("  > CONCLUSION: New data is CORRUPTED (Filter is NOT working).")
        else:
            print("  > CONCLUSION: New data is VALID (Filter IS working).")
    else:
        print("  > No data returned.")

if __name__ == "__main__":
    main()

