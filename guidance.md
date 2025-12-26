# Implementation Specification: Mental Health Intent Classifier Refactor

**Objective:** Transition the codebase from a "Subreddit Classifier" (prone to noise and topic overfitting) to a robust "Intent Detector" suitable for LLM prompt safety, using Iterative Filtering and Hard Negative Mining.

## 1. Core Logic Update: Granularity (The "Context" Fix)
**File:** `src/core/utils.py` & `src/data/loaders.py`

**The Problem:** Single-sentence chunking strips context (e.g., "I'm done" is ambiguous; "I tried everything. I'm done." is clear).
**The Fix:** Replace `chunk_text_by_sentence` with a sliding window approach.

* **Action:** Rewrite `chunk_text_by_sentence` to `chunk_text_sliding_window`.
    * **Logic:** Split by sentences -> Group into windows of 3 sentences (stride 1).
    * **Example:** `[S1, S2, S3]`, `[S2, S3, S4]`, `[S3, S4, S5]`.
    * **Constraint:** Enforce `min_tokens` on the *window*, not just the sentence.

## 2. Core Logic Update: Sequential Filtering & Hard Mining
**File:** `scripts/compile_dataset.py`

**The Problem:** Current filtering drops "Safe" items that look like "Risk" (Hard Negatives), preventing the model from learning nuance. It also filters independently, letting noisy Risk labels distort the Safe filter.
**The Fix:** Implement **Sequential Recalculation**.

* **Action:** Refactor `main()` to follow this exact sequence:
    1.  **Load Raw Data:** Do not load `AUDIT_RESULTS_FILE`. Load the raw dataframe/snapshot directly.
    2.  **Calc Density (Pass 1):** Compute `risk_density` for all items.
    3.  **Filter Risk (Clean the Teacher):**
        * Keep Risk items only if `risk_density > 0.25` (Stricter threshold).
        * *Rationale:* If a "Risk" post looks 75% Safe, it's likely just neutral chatter. Drop it.
    4.  **Recalculate Density (Pass 2):**
        * Build a **new** NearestNeighbors index using *only* the `risk_cleaned` set + `all_safe` set.
        * Re-compute `risk_density` for the `all_safe` set.
    5.  **Mine Hard Negatives (The Gold Standard Step):**
        * Identify `safe_hard_negatives`: Safe items with `risk_density > 0.5`.
        * Identify `safe_easy`: Safe items with `risk_density < 0.2`.
        * *Crucial:* **Keep the Hard Negatives.** Do not drop them.
    6.  **Balance & Merge:**
        * `final_train` = `risk_cleaned` + `safe_hard_negatives` (weight 1.0 or oversample) + `safe_easy` (undersample to match ratio).

## 3. New Utility: Active Learning Sampler
**File:** Create `scripts/select_active_learning.py`

**The Problem:** Reddit data does not perfectly map to LLM Prompts (Domain Shift).
**The Fix:** A script to select "Uncertain" examples from your target domain (WildChat) for manual review.

* **Action:** Create a script that:
    1.  Loads the inference output (e.g., `outputs/wildchat_risk_scores.pkl`).
    2.  Filters for **Uncertainty**: Scores between `0.2` and `0.8` (or `0.1` and `0.9`).
    3.  Filters for **Diversity**: Use K-Means clustering on the embeddings of these uncertain samples to pick $N$ diverse examples (e.g., 500).
    4.  Exports to `data/active_learning_candidates.csv` for human review (Golden Labeling).

## 4. Engineering Tweaks (Any & All Other Changes)

* **`src/analysis/metrics.py`**:
    * **Tweak:** Increase PCA `n_components` from `50` to `100` (or remove PCA entirely) inside `reduce_dimensions`.
    * *Rationale:* 50 dimensions might compress "High Risk" and "Hard Negative" clusters together. Higher fidelity is needed for the sequential filter.
* **`src/modeling/training.py`**:
    * **Tweak:** Ensure `weight_decay` is sufficiently high (e.g., `0.1`) to prevent overfitting on the specific vocabulary of subreddits.
* **`src/data/processing.py`**:
    * **Deprecate:** You can likely remove `load_audit_results` if you move the density calculation logic directly into the compilation pipeline (Task 2).

***

### Prompt for your IDE

**User:** "Please apply the changes detailed in the 'Implementation Specification' above. Start by refactoring `src/core/utils.py` for sliding windows, then completely rewrite `scripts/compile_dataset.py` to implement the Sequential Filtering logic. Ensure `src/analysis/metrics.py` exposes the necessary functions for the compilation script to run density calculations dynamically."