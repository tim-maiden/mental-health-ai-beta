import os
import glob
import random
import pandas as pd
import kagglehub
from datasets import load_dataset
from src.core.clients import encoding, supabase
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
    # Load dataset in streaming mode to efficiently handle large file sizes.
    
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

def load_wildchat_dataset_from_supabase(limit=50000):
    """Fetches WildChat chunks from Supabase instead of Hugging Face."""
    print(f"Fetching WildChat data from Supabase (limit={limit})...")
    
    # Supabase fetch loop to handle limits (default max is usually 1000)
    all_rows = []
    batch_size = 1000
    offset = 0
    
    while True:
        try:
            # We select the columns needed for inference/distillation
            # Assuming table is named 'wildchat_embeddings' per upload script
            response = supabase.table("wildchat_embeddings")\
                .select("id, input, input_tokens")\
                .range(offset, offset + batch_size - 1)\
                .execute()
            
            rows = response.data
            if not rows:
                break
                
            all_rows.extend(rows)
            offset += len(rows)
            
            print(f"Fetched {len(all_rows)} rows...", end="\r")
            
            if len(all_rows) >= limit:
                break
                
        except Exception as e:
            print(f"\nError fetching from Supabase: {e}")
            break
            
    df = pd.DataFrame(all_rows)
    print(f"\nLoaded {len(df)} rows from Supabase.")
    return df

def load_goemotions_dataset():
    """Loads and processes the GoEmotions dataset from Kaggle."""
    print("Loading GoEmotions dataset from Kaggle...")
    path = kagglehub.dataset_download("debarshichanda/goemotions")
    
    # The dataset might be split into multiple CSVs (train, test, val or just data)
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    if not csv_files:
        # Sometimes it's in a subdirectory
        csv_files = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
    
    # Filter out known non-data CSVs
    csv_files = [f for f in csv_files if os.path.basename(f) not in ['emotion_words.csv', 'ekman_labels.csv']]
        
    print(f"Found {len(csv_files)} data CSV files: {[os.path.basename(f) for f in csv_files]}")
    
    df_list = [pd.read_csv(file) for file in csv_files]
    df = pd.concat(df_list, ignore_index=True)
    
    print(f"Total raw rows: {len(df)}")

    # Standard GoEmotions columns
    emotion_cols = [
        "admiration", "amusement", "anger", "annoyance", "approval", "caring",
        "confusion", "curiosity", "desire", "disappointment", "disapproval",
        "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
        "joy", "love", "nervousness", "optimism", "pride", "realization",
        "relief", "remorse", "sadness", "surprise", "neutral"
    ]
    
    # Filter for valid emotion columns that exist in this version of the dataset
    existing_emotion_cols = [col for col in emotion_cols if col in df.columns]
    
    # Optional: Filter out unclear examples if the column exists
    if 'example_very_unclear' in df.columns:
        initial_len = len(df)
        df = df[df['example_very_unclear'] == False]
        print(f"Filtered out {initial_len - len(df)} unclear examples.")

    processed_rows = []
    
    print("Processing and chunking rows...")
    for idx, row in df.iterrows():
        text = str(row['text'])
        
        # Identify active emotions for this row
        active_emotions = [emo for emo in existing_emotion_cols if row[emo] == 1]
        
        # Chunk the text using the standard sliding window
        chunks = chunk_text_sliding_window(text)
        chunk_order_id = 1
        
        # Use existing ID or generate one
        post_id = row.get('id', f"goemotions_{idx}")
        
        for chunk in chunks:
            if not chunk.strip(): continue
            
            processed_rows.append({
                'post_id': post_id,
                'chunk_id': chunk_order_id,
                'input': chunk.strip(),
                'subreddit': row.get('subreddit', 'unknown'),
                'emotions': active_emotions, # Store as list
            })
            chunk_order_id += 1
            
    df_processed = pd.DataFrame(processed_rows)
    
    # Calculate token counts
    print(f"Calculating token counts for {len(df_processed)} chunks...")
    df_processed['input_tokens'] = df_processed['input'].apply(
        lambda text: len(encoding.encode(text, allowed_special={"<|endofprompt|>", "<|endoftext|>"})) if isinstance(text, str) else 0
    )
    
    print(f"Finished processing GoEmotions dataset. Total chunks: {len(df_processed)}")
    return df_processed

def yield_reddit_mental_health_dataset(batch_size=1000, sample_rate=1.0, limit=None):
    """
    Yields batches of processed chunks from the Reddit mental health dataset.
    Processes files iteratively to avoid memory issues.
    ONLY loads from the 'raw data' directory to ensure metadata consistency (author, date).
    
    Args:
        batch_size (int): Number of chunks per yielded batch.
        sample_rate (float): Fraction of posts to process (0.0 to 1.0). Used for downsampling.
        limit (int): Maximum number of chunks to yield in total.
    """
    print(f"Initializing Reddit Mental Health dataset loader (Generator). Sample Rate: {sample_rate}, Limit: {limit}...")
    path = kagglehub.dataset_download("entenam/reddit-mental-health-dataset")
    # STRICTLY target "raw data" folder inside Original Reddit Data
    # This avoids the "Labelled Data" folder which has missing metadata
    data_path = os.path.join(path, "Original Reddit Data", "raw data")
    
    # Recursive search for all CSVs in raw data
    csv_files = glob.glob(os.path.join(data_path, "**", "*.csv"), recursive=True)
    print(f"Found {len(csv_files)} raw data CSV files.")
    
    # Sort files for deterministic order
    csv_files.sort()

    # Set seed for reproducible sampling
    random.seed(42)

    current_batch_rows = []
    total_chunks_yielded = 0
    
    # Author tracking for cap
    author_counts = {}
    MAX_POSTS_PER_AUTHOR = 5

    for file_idx, file_path in enumerate(csv_files):
        # Stop if we hit the global limit
        if limit is not None and total_chunks_yielded >= limit:
            print(f"Global limit of {limit} chunks reached. Stopping.")
            break

        try:
            file_name = os.path.basename(file_path)
            
            # Date Filtering at File Level (Optimization)
            # Filter rows by 'created_utc' rather than relying on inconsistent filenames.
            
            # Read CSV
            df = pd.read_csv(file_path, on_bad_lines='skip', low_memory=False)
            
            # Normalize Columns
            # Required columns: author, created_utc, score, selftext, subreddit, title
            
            # Check for essential content columns
            if 'selftext' not in df.columns:
                 print(f"Skipping {file_name}: Missing 'selftext' column")
                 continue

            # Check dates if column exists
            # Aug 1 2021 - Aug 31 2022
            START_UTC = 1627776000
            END_UTC = 1661990400

            for index, row in df.iterrows():
                # Stop if we hit the global limit (check inside loop for precision)
                if limit is not None and total_chunks_yielded >= limit:
                    break

                selftext = str(row['selftext'])
                
                # Content Filtering
                if not selftext or selftext.lower() in ['nan', 'none', '', '[removed]', '[deleted]']:
                    continue

                # Downsampling: Randomly skip posts based on sample_rate
                # We check this BEFORE date filtering to simulate a smaller dataset of the same distribution
                if sample_rate < 1.0 and random.random() > sample_rate:
                    continue
                
                # Date Filtering
                try:
                    created_utc = float(row.get('created_utc', 0.0))
                except (ValueError, TypeError):
                    created_utc = 0.0
                
                if created_utc < START_UTC or created_utc > END_UTC:
                    continue

                # Chunking
                chunks = chunk_text_sliding_window(selftext)
                chunk_order_id = 1
                
                # Create Post ID
                post_id = str(row.get('post_id', f"{file_name}_{index}"))
                
                # Get other fields - Raw data is expected to have these
                author = str(row.get('author', 'unknown'))

                # Author Cap Check (Prioritize Diversity)
                # We check this BEFORE yielding but we must be careful:
                # If we filter here, we save processing time.
                if author != 'unknown':
                    current_count = author_counts.get(author, 0)
                    if current_count >= MAX_POSTS_PER_AUTHOR:
                        continue # Skip power users / helpers
                    
                    # Check author cap before yielding to optimize processing time.
                    author_counts[author] = current_count + 1

                try:
                    created_utc = float(row.get('created_utc', 0.0))
                except (ValueError, TypeError):
                    created_utc = 0.0
                    
                score = row.get('score', 0)
                # Handle case where score might be string or missing
                try:
                    score = int(score)
                except (ValueError, TypeError):
                    score = 0
                    
                subreddit = str(row.get('subreddit', 'unknown'))
                title = str(row.get('title', ''))

                for chunk in chunks:
                    if not chunk.strip(): continue
                    
                    current_batch_rows.append({
                        'post_id': post_id,
                        'chunk_id': chunk_order_id,
                        'input': chunk.strip(),
                        'author': author,
                        'created_utc': created_utc,
                        'score': score,
                        'subreddit': subreddit,
                        'title': title
                    })
                    chunk_order_id += 1
                    
                    # YIELD BATCH IF FULL
                    if len(current_batch_rows) >= batch_size:
                        df_batch = pd.DataFrame(current_batch_rows)
                        
                        # Calculate tokens
                        df_batch['input_tokens'] = df_batch['input'].apply(
                            lambda text: len(encoding.encode(text, allowed_special={"<|endofprompt|>", "<|endoftext|>"})) if isinstance(text, str) else 0
                        )
                        
                        yield df_batch
                        total_chunks_yielded += len(df_batch)
                        current_batch_rows = []
                        
                        if limit is not None and total_chunks_yielded >= limit:
                            break
                        
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue
            
    # Yield remaining rows
    if current_batch_rows and (limit is None or total_chunks_yielded < limit):
        df_batch = pd.DataFrame(current_batch_rows)
        df_batch['input_tokens'] = df_batch['input'].apply(
            lambda text: len(encoding.encode(text, allowed_special={"<|endofprompt|>", "<|endoftext|>"})) if isinstance(text, str) else 0
        )
        yield df_batch
        total_chunks_yielded += len(df_batch)

def load_reddit_mental_health_dataset():
    """
    DEPRECATED: Use yield_reddit_mental_health_dataset for iterative processing.
    Maintained for backward compatibility but warns.
    """
    print("WARNING: load_reddit_mental_health_dataset loads everything into memory.")
    print("Use yield_reddit_mental_health_dataset instead.")
    # Consumes the generator fully
    batches = []
    for batch in yield_reddit_mental_health_dataset(batch_size=10000):
        batches.append(batch)
    if not batches:
        return pd.DataFrame()
    return pd.concat(batches, ignore_index=True)

def load_reddit_control_dataset():
    """Loads and processes the Reddit control (safe) dataset from HuggingFace."""
    # Check for cached parquet file
    CACHE_FILE = "data/reddit_control_cache.parquet"
    if os.path.exists(CACHE_FILE):
        print(f"Found cached control dataset at {CACHE_FILE}. Loading...")
        try:
            df = pd.read_parquet(CACHE_FILE)
            print(f"Loaded {len(df)} rows from cache.")
            return df
        except Exception as e:
            print(f"Error loading cache: {e}. Reloading from source.")
    
    print("Loading Reddit Control dataset (Safe Data)...")
    
    # Target approximately 60,000 samples per subreddit on average (Total ~3M target to land 1.5M actual)
    # Oversample by 4x to account for low yield (~60%) and data cleaning attrition.
    
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
    # Increase target to ~3M total (avg 60k per sub) to achieve 1.5M valid chunks
    TARGET_TOTAL_SAMPLES = 60000 * len(SUBREDDIT_SIZES) 
    
    TARGET_SUBREDDITS = {}
    for sub, count in SUBREDDIT_SIZES.items():
        # Calculate proportional target
        target = int((count / TOTAL_SOURCE_ROWS) * TARGET_TOTAL_SAMPLES)
        # Ensure at least a minimal sample if the subreddit exists in our list
        # And cap at a reasonable maximum to avoid one subreddit dominating
        # Cap targets: Min 10k, Max 200k
        target = max(10000, min(target, 200000)) 
        TARGET_SUBREDDITS[sub] = target
    
    print(f"Targeting {sum(TARGET_SUBREDDITS.values())} rows across {len(TARGET_SUBREDDITS)} subreddits...")
    
    collected_data = {sub: [] for sub in TARGET_SUBREDDITS.keys()}
    # Track post counts per author to prevent power-user bias
    author_counts = {} 
    MAX_POSTS_PER_AUTHOR = 5
    
    print("Streaming and filtering data (this may take a moment)...")
    
    for subreddit, target_count in TARGET_SUBREDDITS.items():
        print(f"Processing r/{subreddit}...")
        try:
            # The dataset uses subreddit names as splits
            dataset_stream = load_dataset("HuggingFaceGECLM/REDDIT_submissions", split=subreddit, streaming=True)
        except ValueError as e:
            print(f"Warning: Could not load split for r/{subreddit}: {e}")
            continue

        # Date Filtering (Aug 2021 - Aug 2022)
        # Aug 1, 2021 = 1627776000
        # Aug 31, 2022 = 1661990400
        START_UTC = 1627776000
        END_UTC = 1661990400
        
        print(f"  > Scanning for posts between 2021-08-01 and 2022-08-31...")

        for i, row in enumerate(dataset_stream):
            # Check if we have enough for this sub
            if len(collected_data[subreddit]) >= target_count:
                break
                
            # Date Filter
            try:
                # Ensure safe cast to float for 'created_utc' field.
                created_utc = float(row.get('created_utc', 0))
            except (ValueError, TypeError):
                continue
                
            if created_utc < START_UTC:
                continue # Too old
            if created_utc > END_UTC:
                continue # Too new

            # Strict Selftext Quality Filter
            body = row.get('selftext', "")
            title = row.get('title', "")
            
            # Filter 1: Existence
            if not body or not isinstance(body, str) or not body.strip():
                continue # Skip empty bodies

            # Filter 2: Length (Token count of BODY only)
            if len(encoding.encode(body)) < 20:
                continue # Skip bodies shorter than the single chunk length
            
            # Check for removal markers
            if "[removed]" in body or "[deleted]" in body:
                continue
                
            # Add to collection
            # Chunk the text to match mental health dataset processing (Body only)
            chunks = chunk_text_sliding_window(body.strip())
            
            # Generate a unique pseudo-ID using subreddit and stream index.
            post_id = f"{subreddit}_{i}" 
            
            # Author
            author = str(row.get('author', 'unknown'))

            # Global Author Cap Check (Optimized for actor split)
            if author != 'unknown':
                current_count = author_counts.get(author, 0)
                if current_count >= MAX_POSTS_PER_AUTHOR:
                    continue # Skip this post, this author has enough representation
                
                author_counts[author] = current_count + 1

            chunk_order_id = 1
            for chunk in chunks:
                collected_data[subreddit].append({
                    "post_id": post_id,
                    "chunk_id": chunk_order_id,
                    "input": chunk.strip(),
                    "author": author,
                    "created_utc": created_utc,
                    "subreddit": subreddit,
                    "title": title,
                    "score": row.get('score', 0)
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
    random.seed(42)
    random.shuffle(final_rows)
    
    df = pd.DataFrame(final_rows)
    
    # Calculate token counts for the input
    df['input_tokens'] = df['input'].apply(
        lambda text: len(encoding.encode(text, allowed_special={"<|endofprompt|>", "<|endoftext|>"})) if isinstance(text, str) else 0
    )
    
    print(f"\nFinished processing Reddit Control dataset. Total rows: {len(df)}")
    
    # --- CACHE THE DATASET ---
    try:
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        df.to_parquet(CACHE_FILE)
        print(f"Cached dataset to {CACHE_FILE}")
    except Exception as e:
        print(f"Warning: Could not cache dataset: {e}")

    return df
