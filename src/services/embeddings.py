import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from src.core.clients import openai_client, encoding
from src.services.rate_limiter import rate_limiter

def embed_text(text, large=False, retries=3, delay=5):
    """
    Embeds text using OpenAI's API with retry logic for rate limiting.
    """
    model = "text-embedding-3-large" if large else "text-embedding-3-small"
    dimensions = 3072 if large else 1536
    
    # Proactively manage rate limits before making the API call
    try:
        num_tokens = len(encoding.encode(text, allowed_special={"<|endofprompt|>", "<|endoftext|>"}))
        rate_limiter.acquire(num_tokens)
    except Exception as e:
        print(f"Error during token calculation or rate limiting: {e}")
        return None

    for attempt in range(retries):
        try:
            output = openai_client.embeddings.create(
                input=text, model=model, dimensions=dimensions
            )
            return output.data[0].embedding
        except Exception as e:
            if "rate limit" in str(e).lower():
                print(f"Rate limit hit. Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                print(f"An error occurred during embedding: {e}")
                return None
    print("Failed to embed text after multiple retries.")
    return None

def embed_dataframe(df: pd.DataFrame, text_column: str = 'input', max_workers: int = 24, desc="Embedding rows"):
    """
    Embeds a DataFrame column in parallel using ThreadPoolExecutor.
    """
    embeddings = [None] * len(df)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(embed_text, text): i for i, text in enumerate(df[text_column])}
        
        # Use tqdm to create a progress bar
        for future in tqdm(as_completed(future_to_index), total=len(df), desc=desc):
            index = future_to_index[future]
            try:
                embeddings[index] = future.result()
            except Exception as e:
                print(f"Row {index} failed to embed: {e}")

    df['embedding'] = embeddings
    return df

