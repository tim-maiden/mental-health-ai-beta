ALTER TABLE reddit_safe_embeddings DROP COLUMN IF EXISTS emotion_label;
ALTER TABLE reddit_safe_embeddings ADD COLUMN IF NOT EXISTS predicted_emotions text[]; 

