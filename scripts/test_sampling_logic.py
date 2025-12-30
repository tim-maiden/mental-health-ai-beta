import random

def mock_yield_data(sample_rate=1.0):
    """
    Simulates the logic in loaders.py but isolated to test sampling behavior.
    """
    random.seed(42)  # Reset seed at start of generator
    
    # Mock "raw data" - imagine 100 posts
    total_posts = 100
    
    yielded_count = 0
    
    print(f"\n--- Generator Start (Sample Rate: {sample_rate}) ---")
    
    for i in range(total_posts):
        # Logic from loaders.py:
        # if sample_rate < 1.0 and random.random() > sample_rate: continue
        
        is_sampled = True
        rand_val = random.random()
        
        if sample_rate < 1.0:
            if rand_val > sample_rate:
                is_sampled = False
        
        status = "KEEP" if is_sampled else "SKIP"
        # print(f"Post {i}: {status} (Random: {rand_val:.4f})")
        
        if is_sampled:
            yielded_count += 1
            yield i  # Yield the "post"
            
    print(f"--- Generator End. Yielded {yielded_count} / {total_posts} ---")

print("Testing deterministic behavior...")

print("\nRUN 1: sample_rate=1.0")
list(mock_yield_data(1.0))

print("\nRUN 2: sample_rate=0.25")
gen_25 = list(mock_yield_data(0.25))

print("\nRUN 3: sample_rate=0.25 (Again - should be identical)")
gen_25_retry = list(mock_yield_data(0.25))

assert gen_25 == gen_25_retry, "Sampling is NOT deterministic!"
print("\nDeterministic check passed: 25% runs are identical.")

# Now the critical check:
# If we change sample rate, does the "stream" of accepted items maintain order/identity?
# No, because random.random() is called sequentially.
# If we process item 0:
#  - At rate 1.0: random() is NOT called (or check is skipped). Actually, logic is:
#    if sample_rate < 1.0 and random.random() > sample_rate:
#  - So at rate 1.0, random() is NEVER called.
#  - At rate 0.25, random() IS called for every item.

# Wait, look at loaders.py line 309:
# if sample_rate < 1.0 and random.random() > sample_rate:
#     continue

# IMPLICATION:
# When sample_rate = 1.0, random.random() is NOT called.
# When sample_rate = 0.25, random.random() IS called for every row.

# This means the state of the random generator is completely different if we were to mix logic,
# but here the random seed is reset at the start of the function.
# So `yield_reddit_mental_health_dataset` resets the seed every time it is called.

# THE PROBLEM:
# If you run with sample_rate=1.0, you get ALL 3.8M rows.
# If you run with sample_rate=0.25, you get ~960k rows.

# If your `processed_rows` (progress.json) says "1,000,000", that implies you have uploaded 1M rows.
# If you switch to 0.25, the TOTAL dataset size is only 960k.
# So if you try to "Resume from 1,000,000" on a dataset that is only 960k long...
# You skip the entire dataset (0 to 960k) and wait for row 1,000,001... which never comes.

print("\nCONCLUSION:")
print("If progress is 1,000,000 and total items at 25% is < 1,000,000,")
print("the script will skip everything and finish immediately.")

