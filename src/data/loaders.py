import os
import glob
import random
import pandas as pd
import kagglehub
from datasets import load_dataset
from src.core.clients import encoding
from src.core.utils import chunk_text_sliding_window

def load_lmsys_chat_dataset():
    """Loads and processes the lmsys-chat-1m dataset."""
    print("Loading lmsys-chat-1m dataset...")
    ds = load_dataset("lmsys/lmsys-chat-1m")
    df = ds['train'].to_pandas()
    
    exploded_rows = []
    for index, row in df.iterrows():
        conversation = row['conversation']
        for i in range(len(conversation) - 1):
            current_turn = conversation[i]
            next_turn = conversation[i+1]
            if current_turn.get('role') == 'user' and next_turn.get('role') == 'assistant':
                exploded_rows.append({
                    'conversation_id': row['conversation_id'],
                    'model': row['model'],
                    'language': row['language'],
                    'turn_id': (i // 2) + 1,
                    'input': current_turn.get('content'),
                    'output': next_turn.get('content')
                })
    df_chat_turns = pd.DataFrame(exploded_rows)
    
    # Calculate token counts for the input
    df_chat_turns['input_tokens'] = df_chat_turns['input'].apply(
        lambda text: len(encoding.encode(text, allowed_special={"<|endofprompt|>", "<|endoftext|>"})) if isinstance(text, str) else 0
    )

    print("Finished processing lmsys-chat-1m dataset.")
    return df_chat_turns

def load_wildchat_dataset(limit=5000):
    """Loads and processes the WildChat dataset (allenai/WildChat-1M)."""
    print(f"Loading allenai/WildChat-1M dataset (First {limit} conversations)...")
    # Load the dataset
    # We can't easily limit 'load_dataset' for non-streaming unless we slice afterwards, 
    # but streaming=True + taking N is efficient.
    # However, to keep it simple with pandas, we can load it (lazy if possible) or just slice the df.
    # The dataset is large, so let's try streaming to avoid memory issues if we only want a small slice.
    
    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    
    exploded_rows = []
    conversation_count = 0
    
    for row in ds:
        # Stop if we hit the limit
        if conversation_count >= limit:
            break
            
        # Only process English rows
        if row['language'] != 'English':
            continue
            
        conversation = row['conversation']
        if conversation is None: continue

        # Iterate through turns looking for User -> Assistant pairs
        has_valid_turn = False
        for i in range(len(conversation) - 1):
            current_turn = conversation[i]
            next_turn = conversation[i+1]
            
            if current_turn.get('role') == 'user' and next_turn.get('role') == 'assistant':
                user_content = current_turn.get('content')
                
                # CHUNK the user input into sentences
                chunks = chunk_text_sliding_window(user_content)
                chunk_order_id = 1
                
                for chunk in chunks:
                    if not chunk.strip(): continue
                    
                    exploded_rows.append({
                        'conversation_hash': row['conversation_hash'], # Using hash as ID
                        'model': row['model'],
                        'language': row['language'],
                        'turn_id': (i // 2) + 1,
                        'chunk_id': chunk_order_id,
                        'input': chunk.strip(),
                        # We keep the full output context if needed, though we are embedding the input chunk
                        'output': next_turn.get('content') 
                    })
                    chunk_order_id += 1
                has_valid_turn = True
        
        if has_valid_turn:
            conversation_count += 1
            if conversation_count % 100 == 0:
                print(f"Processed {conversation_count} conversations...", end="\r")

    df_chat_chunks = pd.DataFrame(exploded_rows)
    
    # Calculate token counts for the input chunk
    print(f"\nCalculating token counts for {len(df_chat_chunks)} chunks...")
    df_chat_chunks['input_tokens'] = df_chat_chunks['input'].apply(
        lambda text: len(encoding.encode(text, allowed_special={"<|endofprompt|>", "<|endoftext|>"})) if isinstance(text, str) else 0
    )

    print(f"Finished processing WildChat dataset. Generated {len(df_chat_chunks)} chunks from {conversation_count} conversations.")
    return df_chat_chunks

def load_reddit_mental_health_dataset():
    """Loads and processes the Reddit mental health dataset with cleaning."""
    print("Loading Reddit Mental Health dataset (Risk Data)...")
    path = kagglehub.dataset_download("entenam/reddit-mental-health-dataset")
    data_path = os.path.join(path, "Original Reddit Data", "Labelled Data")
    
    csv_files = glob.glob(os.path.join(data_path, "*.csv"))
    df_list = [pd.read_csv(file) for file in csv_files]
    df = pd.concat(df_list, ignore_index=True)
    df = df.reset_index().rename(columns={'index': 'post_id'})

    # --- LABEL CLEANING ---
    print("Cleaning labels...")
    # Convert to string, strip whitespace, and drop NaN/empty labels
    df['Label'] = df['Label'].astype(str).str.strip()
    df = df[df['Label'].str.lower() != 'nan']
    df = df[df['Label'] != '']
    print(f"Rows after label cleaning: {len(df)}")

    multiturn_rows = []
    for index, row in df.iterrows():
        selftext = str(row['selftext'])
        chunks = chunk_text_sliding_window(selftext) # Use the sentence chunking function
        chunk_order_id = 1
        for chunk in chunks:
            multiturn_rows.append({
                'post_id': row['post_id'],
                'chunk_id': chunk_order_id,
                'input': chunk.strip(),
                'score': row['score'],
                'subreddit': row['subreddit'],
                'title': row['title'],
                'label': row['Label'],
                'cat_1': row['CAT 1']
            })
            chunk_order_id += 1
    df_reddit_chunks = pd.DataFrame(multiturn_rows)

    # Calculate token counts for the input
    df_reddit_chunks['input_tokens'] = df_reddit_chunks['input'].apply(
        lambda text: len(encoding.encode(text, allowed_special={"<|endofprompt|>", "<|endoftext|>"})) if isinstance(text, str) else 0
    )

    print(f"Finished processing Reddit mental health dataset. Total chunks: {len(df_reddit_chunks)}")
    return df_reddit_chunks

def load_reddit_control_dataset():
    """Loads and processes the Reddit control (safe) dataset from HuggingFace."""
    print("Loading Reddit Control dataset (Safe Data)...")
    
    # Target approximately 200 samples per subreddit on average (Total ~10,000)
    # Sampling proportional to split size from the source dataset.
    
    SUBREDDIT_SIZES = {
        "tifu": 526000,
        "explainlikeimfive": 1810000,
        "WritingPrompts": 1000000,
        "changemyview": 257000,
        "LifeProTips": 715000,
        "todayilearned": 2150000,
        "science": 873000,
        "askscience": 1560000,
        "ifyoulikeblank": 221000,
        "Foodforthought": 70600,
        "IWantToLearn": 103000,
        "bestof": 341000,
        "IAmA": 436000,
        "socialskills": 260000,
        "relationship_advice": 3280000,
        "philosophy": 213000,
        "YouShouldKnow": 94600,
        "history": 284000,
        "books": 693000,
        "Showerthoughts": 6360000,
        "personalfinance": 1350000,
        "buildapc": 3030000,
        "EatCheapAndHealthy": 79700,
        "boardgames": 287000,
        "malefashionadvice": 549000,
        "femalefashionadvice": 131000,
        "scifi": 135000,
        "Fantasy": 176000,
        "Games": 831000,
        "bodyweightfitness": 145000,
        "SkincareAddiction": 890000,
        "podcasts": 114000,
        "suggestmeabook": 301000,
        "AskHistorians": 592000,
        "gaming": 6420000,
        "DIY": 506000,
        "mildlyinteresting": 1970000,
        "sports": 784000,
        "space": 416000,
        "gadgets": 284000,
        "Documentaries": 301000,
        "GetMotivated": 396000,
        "UpliftingNews": 285000,
        "technology": 2110000,
        "Fitness": 1040000,
        "travel": 1010000,
        "lifehacks": 117000,
        "Damnthatsinteresting": 397000,
        "gardening": 723000,
        "programming": 571000
    }
    
    TOTAL_SOURCE_ROWS = sum(SUBREDDIT_SIZES.values())
    TARGET_TOTAL_SAMPLES = 1000 * len(SUBREDDIT_SIZES) # Aim for avg 1000 per subreddit
    
    TARGET_SUBREDDITS = {}
    for sub, count in SUBREDDIT_SIZES.items():
        # Calculate proportional target
        target = int((count / TOTAL_SOURCE_ROWS) * TARGET_TOTAL_SAMPLES)
        # Ensure at least a minimal sample if the subreddit exists in our list
        # And cap at a reasonable maximum to avoid one subreddit dominating
        target = max(500, min(target, 5000)) 
        TARGET_SUBREDDITS[sub] = target

    MIN_LENGTH = 50
    
    print(f"Targeting {sum(TARGET_SUBREDDITS.values())} rows across {len(TARGET_SUBREDDITS)} subreddits...")
    
    collected_data = {sub: [] for sub in TARGET_SUBREDDITS.keys()}
    
    print("Streaming and filtering data (this may take a moment)...")
    
    for subreddit, target_count in TARGET_SUBREDDITS.items():
        print(f"Processing r/{subreddit}...")
        try:
            # The dataset uses subreddit names as splits
            dataset_stream = load_dataset("HuggingFaceGECLM/REDDIT_submissions", split=subreddit, streaming=True)
        except ValueError as e:
            print(f"Warning: Could not load split for r/{subreddit}: {e}")
            continue

        for i, row in enumerate(dataset_stream):
            # Check if we have enough for this sub
            if len(collected_data[subreddit]) >= target_count:
                break

            # Content Filter: Combine Title + Selftext (Body)
            title = row.get('title', "")
            body = row.get('selftext', "")
            full_text = f"{title}\n{body}".strip()
            
            # Quality Check
            if len(full_text) < MIN_LENGTH or "[removed]" in body or "[deleted]" in body:
                continue
                
            # Add to collection
            # Chunk the text to match mental health dataset processing
            chunks = chunk_text_sliding_window(full_text)
            
            # We need a unique ID for each post to track chunks
            # Since we don't have a post_id from this dataset, we'll generate one
            # Or we can just use a simple counter for now, or hash the text
            # Let's use a simple tuple of (subreddit, index_in_stream) to make a pseudo-ID if we need
            # But for simplicity, we'll just carry over the chunk logic.
            # The mental health dataset has 'post_id' and 'chunk_id'. 
            # Let's assume we can generate a unique ID for the post here.
            post_id = f"{subreddit}_{i}" 
            
            chunk_order_id = 1
            for chunk in chunks:
                collected_data[subreddit].append({
                    "post_id": post_id,
                    "chunk_id": chunk_order_id,
                    "input": chunk.strip(),
                    "subreddit": subreddit,
                    "label": "Safe_Control"
                })
                chunk_order_id += 1

            if i % 1000 == 0:
                print(f"  Scanned {i} rows for r/{subreddit}, collected {len(collected_data[subreddit])} chunks...", end="\r")

        print(f" -> Finished collecting r/{subreddit} ({len(collected_data[subreddit])} chunks)")

    # Flatten
    final_rows = []
    for sub_rows in collected_data.values():
        final_rows.extend(sub_rows)
    
    # Shuffle to mix them up
    random.shuffle(final_rows)
    
    df = pd.DataFrame(final_rows)
    
    # Calculate token counts for the input
    df['input_tokens'] = df['input'].apply(
        lambda text: len(encoding.encode(text, allowed_special={"<|endofprompt|>", "<|endoftext|>"})) if isinstance(text, str) else 0
    )
    
    print(f"\nFinished processing Reddit Control dataset. Total rows: {len(df)}")
    return df

