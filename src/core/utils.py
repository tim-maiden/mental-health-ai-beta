import re
from src.core.clients import encoding

def chunk_text_sliding_window(text: str, window_size: int = 3, stride: int = 1, min_tokens: int = 20) -> list[str]:
    """
    Splits text into sentences and creates sliding window chunks.
    
    Args:
        text: The input text.
        window_size: Number of sentences per window.
        stride: Step size for the sliding window.
        min_tokens: Minimum tokens required for a window to be kept (unless it's the only one).
    """
    if not text:
        return []

    # 1. Protect ellipses
    text = text.replace('...', '<ELLIPSIS>')
    
    # 2. Split by sentence-ending punctuation
    text_with_delimiters = re.sub(r'([.!?]+)', r'\1<SPLIIT>', text)
    sentences = [chunk.strip() for chunk in text_with_delimiters.split('<SPLIIT>') if chunk.strip()]
    
    # 3. Restore ellipses
    sentences = [s.replace('<ELLIPSIS>', '...') for s in sentences]
    
    if not sentences:
        return []

    chunks = []
    
    # 4. Create Sliding Windows
    # If fewer sentences than window size, take all of them as one chunk
    if len(sentences) <= window_size:
        chunks.append(" ".join(sentences))
    else:
        for i in range(0, len(sentences) - window_size + 1, stride):
            window = sentences[i : i + window_size]
            chunk_text = " ".join(window)
            chunks.append(chunk_text)
            
        # Handle the tail if stride skips the last few and we haven't covered them in a "full" window
        # But range(0, len - size + 1) ensures the last window starts at len-size.
        # With stride 1, we cover everything. With stride > 1, we might miss the very end as a distinct start,
        # but the last window covers the end.
    
    # 5. Enforce Min Tokens (Filter)
    # If a chunk is too small, we might want to drop it, OR merge it?
    # Guidance says: "Constraint: Enforce min_tokens on the window".
    # I'll interpret this as: Filter out windows that are too small, unless it's the ONLY window.
    
    final_chunks = []
    for chunk in chunks:
        # Check token count
        token_count = len(encoding.encode(chunk, allowed_special={"<|endofprompt|>", "<|endoftext|>"}))
        if token_count >= min_tokens:
            final_chunks.append(chunk)
    
    # Fallback: If we filtered everything out (all small), keep the longest one or merge all?
    # If we have no final chunks but we had sentences, it means all windows were too short.
    # In that case, we should probably just return the whole text as one chunk.
    if not final_chunks and chunks:
        # Check if the WHOLE text is long enough?
        # Or just return the largest window we found?
        # Let's return the original full text if it's substantial, or just the list of small chunks?
        # Better: if nothing passed the filter, just return the whole text as a single chunk.
        final_chunks.append(text)

    return final_chunks
