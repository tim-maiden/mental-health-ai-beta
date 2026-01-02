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

    # Protect ellipses -> Split on punctuation -> Restore
    text = text.replace('...', '<ELLIPSIS>')
    
    text_with_delimiters = re.sub(r'([.!?]+)', r'\1<SPLIIT>', text)
    sentences = [chunk.strip() for chunk in text_with_delimiters.split('<SPLIIT>') if chunk.strip()]
    
    sentences = [s.replace('<ELLIPSIS>', '...') for s in sentences]
    
    if not sentences:
        return []

    chunks = []
    
    # Create Sliding Windows
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
    
    # Enforce Min Tokens (Filter)
    # Filter out windows below 'min_tokens' unless it is the only generated chunk.
    
    final_chunks = []
    for chunk in chunks:
        # Check token count
        token_count = len(encoding.encode(chunk, allowed_special={"<|endofprompt|>", "<|endoftext|>"}))
        if token_count >= min_tokens:
            final_chunks.append(chunk)
    
    # Fallback: If all chunks are filtered (too short), return the original text as a single chunk.
    if not final_chunks and chunks:
        final_chunks.append(text)

    return final_chunks
