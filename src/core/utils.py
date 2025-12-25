import re
from src.core.clients import encoding

def _merge_small_chunks(chunks: list[str], min_tokens: int) -> list[str]:
    """Merges chunks that are smaller than the minimum token size."""
    if not chunks:
        return []

    merged_chunks = []
    buffer = ""
    for chunk in chunks:
        # Add the next chunk to the buffer
        if buffer:
            buffer += " " + chunk
        else:
            buffer = chunk
        
        # If buffer is now large enough, add it to the list
        if len(encoding.encode(buffer, allowed_special={"<|endofprompt|>", "<|endoftext|>"})) >= min_tokens:
            merged_chunks.append(buffer)
            buffer = ""  # Reset buffer
            
    # If there's anything left in the buffer at the end
    if buffer:
        # If there are already merged chunks, append the remainder to the last one
        if merged_chunks:
            merged_chunks[-1] += " " + buffer
        # Otherwise, the entire text was smaller than min_tokens, so it's one chunk
        else:
            merged_chunks.append(buffer)
            
    return merged_chunks

def chunk_text_by_sentence(text: str, min_chunk_tokens: int = 10) -> list[str]:
    """
    Splits text into chunks based on sentence-ending punctuation,
    ignoring ellipses and ensuring a minimum chunk size.
    """
    # 1. Protect ellipses by replacing them with a unique placeholder
    text = text.replace('...', '<ELLIPSIS>')
    
    # 2. Split text by sentence-ending punctuation
    text_with_delimiters = re.sub(r'([.!?]+)', r'\1<SPLIIT>', text)
    initial_chunks = [chunk.strip() for chunk in text_with_delimiters.split('<SPLIIT>') if chunk.strip()]
    
    # 3. Restore ellipses in each chunk
    initial_chunks = [chunk.replace('<ELLIPSIS>', '...') for chunk in initial_chunks]
    
    # 4. Merge chunks that are too small
    final_chunks = _merge_small_chunks(initial_chunks, min_chunk_tokens)
    
    return final_chunks

