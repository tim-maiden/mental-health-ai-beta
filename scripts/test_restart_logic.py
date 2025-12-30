import random

def mock_yield_data(sample_rate=1.0, stop_at=None):
    random.seed(42)  # Reset seed at start of generator
    total_posts = 100
    yielded_count = 0
    
    print(f"\n--- Generator Start (Sample Rate: {sample_rate}, Stop At: {stop_at}) ---")
    
    for i in range(total_posts):
        is_sampled = True
        rand_val = random.random() # Always call random to maintain state if we were streaming
        
        # Logic matches loaders.py exactly
        if sample_rate < 1.0:
            if rand_val > sample_rate:
                is_sampled = False
        
        if is_sampled:
            yielded_count += 1
            yield i
            if stop_at and yielded_count >= stop_at:
                print("--- External Stop Triggered ---")
                return # Simulate "break" or script exit

print("Testing restart behavior...")

print("\nRUN 1: sample_rate=0.25, stop after 10 items")
part1 = list(mock_yield_data(0.25, stop_at=10))
print(f"Part 1 yielded: {part1}")

print("\nRUN 2: sample_rate=0.25, resume from 10 (skip first 10)")
# Simulate "resuming" by running the generator again and discarding the first 10
gen = mock_yield_data(0.25)
part2 = []
skipped = 0
for item in gen:
    if skipped < 10:
        skipped += 1
        continue
    part2.append(item)

print(f"Part 2 yielded (first 5): {part2[:5]}")

# Verification:
# Does Part 1 + Part 2 equal the full sequence?
full_sequence = list(mock_yield_data(0.25))
combined = part1 + part2
print(f"\nFull Sequence (first 15): {full_sequence[:15]}")
print(f"Combined (first 15):      {combined[:15]}")

assert combined == full_sequence, "Restarting with skip BROKE the sequence!"
print("\nSuccess: Resuming works perfectly IF the configuration (sample_rate) is identical.")

