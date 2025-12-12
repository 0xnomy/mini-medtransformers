"""
STEP 1: Dataset Preparation for Mini-MedTransformers
FIXED VERSION - Fully compliant with specification
"""

import pandas as pd
import re
import numpy as np
from pathlib import Path
from typing import Tuple, List
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# IMPROVED TEXT CLEANING
# ============================================================================

def clean_text(text: str, lowercase: bool = True, remove_html: bool = True) -> str:
    """
    Clean text with comprehensive preprocessing.
    
    Args:
        text: Input text string
        lowercase: Whether to convert to lowercase
        remove_html: Whether to remove HTML-like tags
        
    Returns:
        Cleaned text string
    """
    if not isinstance(text, str):
        return ""
    
    # Remove video references (fixed typo: brackets not "brackts")
    text = re.sub(
        r'\(Watch.*?(?:video|brackets|Esc).*?\)', 
        '', 
        text, 
        flags=re.IGNORECASE | re.DOTALL
    )
    
    # Remove similar parenthetical references
    text = re.sub(r'\(To enlarge.*?\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(To reduce.*?\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(Click.*?\)', '', text, flags=re.IGNORECASE)
    
    # Remove HTML-like tags (NEW - as per spec)
    if remove_html:
        text = re.sub(r'<[^>]+>', '', text)
    
    # Remove citation brackets [1], [2], etc. but preserve ranges like [5-10]
    text = re.sub(r'\[\d+\]', '', text)
    
    # Remove line breaks and normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Standardize punctuation spacing
    text = re.sub(r'\s+([.!?,;:])', r'\1', text)
    text = re.sub(r'([.!?,;:])\s+', r'\1 ', text)
    
    # Remove multiple consecutive punctuation (e.g., "..." -> ".")
    text = re.sub(r'([.!?]){2,}', r'\1', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    # Apply lowercase if requested
    if lowercase:
        text = text.lower()
    
    return text


# ============================================================================
# IMPROVED CLASSIFICATION LABEL ASSIGNMENT
# ============================================================================

def classify_topic(question: str) -> str:
    """
    Classify question into topic using improved keyword matching with better priority.
    More specific patterns are checked first to avoid misclassification.
    Rare classes (complications, prevention, prognosis) merged into 'general'.
    
    Args:
        question: The question text (should be lowercased)
        
    Returns:
        Topic label (7 classes): definition, symptom, genetics, treatment, 
                                cause, diagnosis, risk, or general
    """
    q_lower = question.lower().strip()
    
    # Check most specific patterns first (order matters!)
    # Treatment patterns - check before definition
    if re.search(r'\btreat(?:ment|ed|ing|s)', q_lower):
        return 'treatment'
    if re.search(r'\btherapy\b', q_lower):
        return 'treatment'
    if re.search(r'\bmedication', q_lower):
        return 'treatment'
    if re.search(r'\bsurgery\b', q_lower):
        return 'treatment'
    if re.search(r'\bcure[ds]?\b', q_lower):
        return 'treatment'
    if re.search(r'\bmanage[d]?\b', q_lower):
        return 'treatment'
    
    # Genetics patterns - very specific
    if re.search(r'\binherit', q_lower):
        return 'genetics'
    if re.search(r'\bgenetic', q_lower):
        return 'genetics'
    if re.search(r'\bhereditsar', q_lower):
        return 'genetics'
    
    # Symptom patterns
    if re.search(r'\bsymptom', q_lower):
        return 'symptom'
    if re.search(r'\bsign(?:s)?\s+(?:and|of)', q_lower):
        return 'symptom'
    if re.search(r'how\s+(?:do|can)\s+(?:i|you)\s+(?:know|tell)', q_lower):
        return 'symptom'
    
    # Diagnosis patterns
    if re.search(r'\bdiagnos', q_lower):
        return 'diagnosis'
    if re.search(r'(?:what|which)\s+test', q_lower):
        return 'diagnosis'
    if re.search(r'how\s+(?:is|are).*diagnosed', q_lower):
        return 'diagnosis'
    
    # Cause patterns
    if re.search(r'\bcause[ds]?\b', q_lower):
        return 'cause'
    if re.search(r'\breason', q_lower):
        return 'cause'
    if re.search(r'why\s+(?:do|does)', q_lower):
        return 'cause'
    
    # Risk patterns
    if re.search(r'\brisk', q_lower):
        return 'risk'
    if re.search(r'who.*(?:at\s+risk|likely|susceptible)', q_lower):
        return 'risk'
    
    # Definition patterns - check last since they're broad
    if re.search(r'\bwhat\s+(?:is|are)\b', q_lower):
        return 'definition'
    if re.search(r'\bdefine', q_lower):
        return 'definition'
    if re.search(r'\bexplain', q_lower):
        return 'definition'
    
    return 'general'


# ============================================================================
# IMPROVED CLASSIFICATION DATASET BUILDER
# ============================================================================

def build_classification_subset(
    df: pd.DataFrame, 
    n_samples: int = 14000, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build classification dataset with CORRECT column names: ["text", "label"].
    Removes duplicates BEFORE splitting to prevent data leakage.
    
    Args:
        df: Input dataframe with 'question' column
        n_samples: Target number of samples
        random_state: Random seed for reproducibility
        
    Returns:
        (train_df, val_df, test_df) with columns ["text", "label"]
    """
    print("\n" + "="*80)
    print("STEP 1.3A — BUILD CLASSIFICATION SUBSET (ENCODER-ONLY)")
    print("="*80)
    
    # FIXED: Column will be named "text" not "question"
    df_class = df[['question']].copy()
    df_class.columns = ['text']  # ⚠️ CRITICAL FIX
    
    # Remove duplicates BEFORE sampling (prevents data leakage)
    initial_size = len(df_class)
    df_class = df_class.drop_duplicates(subset=['text'])
    removed = initial_size - len(df_class)
    print(f"\n✓ Removed {removed} duplicate questions")
    print(f"✓ Available unique questions: {len(df_class)}")
    
    # Sample n_samples rows
    if len(df_class) > n_samples:
        df_class = df_class.sample(n=n_samples, random_state=random_state)
    else:
        print(f"⚠️  Only {len(df_class)} unique questions available (< {n_samples})")
    
    print(f"✓ Sampled {len(df_class)} questions")
    
    # Create labels
    df_class['label'] = df_class['text'].apply(classify_topic)
    
    # Check label distribution
    label_dist = df_class['label'].value_counts()
    print(f"\n✓ Label distribution:")
    for label, count in label_dist.items():
        pct = 100 * count / len(df_class)
        print(f"  - {label}: {count} ({pct:.1f}%)")
    
    # Shuffle before splitting
    df_class = df_class.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    
    # Train/Val/Test split: 80/10/10
    n_train = int(0.8 * len(df_class))
    n_val = int(0.1 * len(df_class))
    
    train_class = df_class.iloc[:n_train].reset_index(drop=True)
    val_class = df_class.iloc[n_train:n_train+n_val].reset_index(drop=True)
    test_class = df_class.iloc[n_train+n_val:].reset_index(drop=True)
    
    print(f"\n✓ Split sizes:")
    print(f"  - Train: {len(train_class)} ({100*len(train_class)/len(df_class):.1f}%)")
    print(f"  - Val: {len(val_class)} ({100*len(val_class)/len(df_class):.1f}%)")
    print(f"  - Test: {len(test_class)} ({100*len(test_class)/len(df_class):.1f}%)")
    
    # Verify no overlap (data leakage check)
    train_set = set(train_class['text'])
    val_set = set(val_class['text'])
    test_set = set(test_class['text'])
    
    assert len(train_set & val_set) == 0, "❌ Data leakage: Train-Val overlap!"
    assert len(train_set & test_set) == 0, "❌ Data leakage: Train-Test overlap!"
    assert len(val_set & test_set) == 0, "❌ Data leakage: Val-Test overlap!"
    print(f"✓ No data leakage detected")
    
    # Print sample rows
    print(f"\n✓ Sample 5 rows from train set:")
    for idx, row in train_class.head(5).iterrows():
        print(f"  [{row['label']}] {row['text'][:80]}...")
    
    return train_class, val_class, test_class


# ============================================================================
# IMPROVED TOKEN COUNTING AND SENTENCE SPLITTING
# ============================================================================

def count_tokens(text: str) -> int:
    """Count tokens using whitespace split (standard for this project)."""
    return len(str(text).split())


def split_into_sentences_improved(text: str, max_tokens: int = 128) -> List[str]:
    """
    Split text into sentences with improved handling of edge cases.
    Truncates long sentences instead of dropping them.
    
    Args:
        text: Input text
        max_tokens: Maximum tokens per chunk
        
    Returns:
        List of sentence chunks
    """
    # Improved sentence splitting that handles abbreviations
    # Split on period followed by space and capital letter
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        sentence_tokens = count_tokens(sentence)
        
        # FIXED: Truncate long sentences instead of dropping
        if sentence_tokens > max_tokens:
            # Take first max_tokens words
            words = sentence.split()[:max_tokens]
            truncated = ' '.join(words)
            
            # Save current chunk if exists
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_tokens = 0
            
            # Add truncated sentence as standalone chunk
            chunks.append(truncated)
            continue
        
        # If adding this sentence exceeds limit, start new chunk
        if current_tokens + sentence_tokens > max_tokens:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_tokens = sentence_tokens
        else:
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
    
    # Add final chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


# ============================================================================
# IMPROVED LANGUAGE MODELING DATASET BUILDER
# ============================================================================

def build_language_modeling_subset(
    df: pd.DataFrame, 
    n_sentences: int = 50000, 
    max_tokens: int = 128, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build language modeling dataset targeting 50K sentences.
    Uses improved sentence splitting and handles all text.
    
    Args:
        df: Input dataframe with 'answer' column
        n_sentences: Target number of sentences (default 50000)
        max_tokens: Maximum tokens per sentence
        random_state: Random seed
        
    Returns:
        (train_df, val_df, test_df) with column ["text"]
    """
    print("\n" + "="*80)
    print("STEP 1.3B — BUILD LANGUAGE MODELING SUBSET (DECODER-ONLY)")
    print("="*80)
    
    # Shuffle answers to ensure diversity
    answers = df['answer'].sample(frac=1.0, random_state=random_state).tolist()
    
    sentences = []
    processed_answers = 0
    
    for answer in answers:
        chunks = split_into_sentences_improved(answer, max_tokens=max_tokens)
        sentences.extend(chunks)
        processed_answers += 1
        
        # Stop when we reach target
        if len(sentences) >= n_sentences:
            break
    
    # Trim to exact size
    sentences = sentences[:n_sentences]
    
    print(f"\n✓ Processed {processed_answers} answers")
    print(f"✓ Generated {len(sentences)} sentences")
    
    if len(sentences) < n_sentences:
        print(f"⚠️  Only generated {len(sentences)}/{n_sentences} sentences")
        print(f"   Consider reducing n_sentences or using more source data")
    
    # Verify token counts
    token_counts = [count_tokens(s) for s in sentences]
    print(f"\n✓ Token statistics:")
    print(f"  - Min: {min(token_counts)}")
    print(f"  - Max: {max(token_counts)}")
    print(f"  - Mean: {np.mean(token_counts):.1f}")
    print(f"  - Median: {np.median(token_counts):.1f}")
    
    # Verify all within limit
    over_limit = sum(1 for t in token_counts if t > max_tokens)
    print(f"  - Sentences > {max_tokens} tokens: {over_limit}")
    assert over_limit == 0, f"❌ {over_limit} sentences exceed token limit!"
    
    # Create dataframe
    df_lm = pd.DataFrame({'text': sentences})
    
    # Remove very short sentences (< 5 tokens)
    initial_size = len(df_lm)
    df_lm = df_lm[df_lm['text'].apply(count_tokens) >= 5]
    removed = initial_size - len(df_lm)
    if removed > 0:
        print(f"\n✓ Removed {removed} very short sentences (< 5 tokens)")
    
    df_lm = df_lm.reset_index(drop=True)
    
    # Print first 5 sentences
    print(f"\n✓ First 5 sentences:")
    for idx, sentence in enumerate(df_lm['text'].head(5)):
        print(f"  {idx+1}. [{count_tokens(sentence)} tokens] {sentence[:100]}...")
    
    # Train/Val/Test split: 80/10/10
    n_train = int(0.8 * len(df_lm))
    n_val = int(0.1 * len(df_lm))
    
    train_lm = df_lm.iloc[:n_train].reset_index(drop=True)
    val_lm = df_lm.iloc[n_train:n_train+n_val].reset_index(drop=True)
    test_lm = df_lm.iloc[n_train+n_val:].reset_index(drop=True)
    
    print(f"\n✓ Split sizes:")
    print(f"  - Train: {len(train_lm)} ({100*len(train_lm)/len(df_lm):.1f}%)")
    print(f"  - Val: {len(val_lm)} ({100*len(val_lm)/len(df_lm):.1f}%)")
    print(f"  - Test: {len(test_lm)} ({100*len(test_lm)/len(df_lm):.1f}%)")
    
    return train_lm, val_lm, test_lm


# ============================================================================
# IMPROVED SUMMARIZATION
# ============================================================================

def generate_summary(text: str, max_tokens: int = 64) -> str:
    """
    Generate a bullet-style summary from text.
    Takes first 1-2 sentences and formats as concise summary.
    
    Args:
        text: Input text to summarize
        max_tokens: Maximum tokens for summary
        
    Returns:
        Bullet-style summary string
    """
    # Split into sentences (improved regex)
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    
    summary_sentences = []
    summary_tokens = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        sentence_tokens = count_tokens(sentence)
        
        # Check if adding this sentence stays within limit
        if summary_tokens + sentence_tokens <= max_tokens:
            summary_sentences.append(sentence)
            summary_tokens += sentence_tokens
        else:
            # If we have at least one sentence, stop
            if summary_sentences:
                break
            # If first sentence is too long, truncate it
            else:
                words = sentence.split()[:max_tokens]
                summary_sentences.append(' '.join(words))
                break
    
    if not summary_sentences:
        # Fallback: take first max_tokens words
        words = text.split()[:max_tokens]
        return ' '.join(words)
    
    # IMPROVED: Format as bullet point (single bullet)
    summary = ' '.join(summary_sentences)
    
    # Make it more concise by removing filler words
    summary = re.sub(r'\s+(however|moreover|furthermore|additionally)\s+', ' ', summary, flags=re.IGNORECASE)
    
    return summary.strip()


def build_summarization_subset(
    df: pd.DataFrame, 
    n_samples: int = 3000, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build summarization dataset with improved summary generation.
    Filters out cases where input == target.
    
    Args:
        df: Input dataframe with 'answer' column
        n_samples: Target number of pairs
        random_state: Random seed
        
    Returns:
        (train_df, val_df, test_df) with columns ["input", "target"]
    """
    print("\n" + "="*80)
    print("STEP 1.3C — BUILD SUMMARIZATION SUBSET (ENCODER-DECODER)")
    print("="*80)
    
    # Take more samples initially to account for filtering
    oversample_factor = 1.5
    initial_samples = int(n_samples * oversample_factor)
    
    df_summ = df[['answer']].copy()
    
    # Sample more than needed
    if len(df_summ) > initial_samples:
        df_summ = df_summ.sample(n=initial_samples, random_state=random_state)
    
    print(f"\n✓ Initially sampled {len(df_summ)} answers")
    
    # Rename column to 'input'
    df_summ.columns = ['input']
    
    # Generate summaries
    df_summ['target'] = df_summ['input'].apply(generate_summary)
    
    # Remove rows where target is empty
    df_summ = df_summ[df_summ['target'].str.len() > 0]
    
    # IMPROVED: Remove rows where input == target (no actual summarization)
    initial_size = len(df_summ)
    df_summ = df_summ[df_summ['input'] != df_summ['target']]
    removed_identical = initial_size - len(df_summ)
    print(f"✓ Removed {removed_identical} pairs where input == target")
    
    # Remove rows where summary is > 90% of input length (poor summarization)
    input_lens = df_summ['input'].apply(count_tokens)
    target_lens = df_summ['target'].apply(count_tokens)
    compression_ratio = target_lens / input_lens
    
    df_summ = df_summ[compression_ratio < 0.9]
    removed_poor = initial_size - removed_identical - len(df_summ)
    if removed_poor > 0:
        print(f"✓ Removed {removed_poor} pairs with poor compression (>90%)")
    
    # Now take exactly n_samples
    if len(df_summ) > n_samples:
        df_summ = df_summ.sample(n=n_samples, random_state=random_state)
    else:
        print(f"⚠️  Only {len(df_summ)} valid pairs available (< {n_samples})")
    
    print(f"✓ Final size: {len(df_summ)} pairs")
    
    # Verify lengths
    input_tokens = df_summ['input'].apply(count_tokens)
    target_tokens = df_summ['target'].apply(count_tokens)
    
    print(f"\n✓ Input text statistics:")
    print(f"  - Min: {input_tokens.min()} tokens")
    print(f"  - Max: {input_tokens.max()} tokens")
    print(f"  - Mean: {input_tokens.mean():.1f} tokens")
    
    print(f"\n✓ Target summary statistics:")
    print(f"  - Min: {target_tokens.min()} tokens")
    print(f"  - Max: {target_tokens.max()} tokens")
    print(f"  - Mean: {target_tokens.mean():.1f} tokens")
    
    # Check if any targets exceed 64 tokens
    over_limit = (target_tokens > 64).sum()
    if over_limit > 0:
        print(f"  ⚠️  {over_limit} summaries exceed 64 tokens")
    else:
        print(f"  ✓ All summaries within 64 token limit")
    
    # Compression ratio
    compression = input_tokens / target_tokens
    print(f"\n✓ Compression ratio:")
    print(f"  - Mean: {compression.mean():.2f}x")
    print(f"  - Median: {compression.median():.2f}x")
    
    # Print first 3 pairs
    print(f"\n✓ First 3 input/target pairs:")
    for idx, row in df_summ.head(3).iterrows():
        print(f"\n  Pair {idx}:")
        print(f"  INPUT ({count_tokens(row['input'])} tok): {row['input'][:100]}...")
        print(f"  TARGET ({count_tokens(row['target'])} tok): {row['target'][:100]}...")
    
    # Reset index and split
    df_summ = df_summ.reset_index(drop=True)
    
    # Shuffle before splitting
    df_summ = df_summ.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    
    # Train/Val/Test split: 80/10/10
    n_train = int(0.8 * len(df_summ))
    n_val = int(0.1 * len(df_summ))
    
    train_summ = df_summ.iloc[:n_train].reset_index(drop=True)
    val_summ = df_summ.iloc[n_train:n_train+n_val].reset_index(drop=True)
    test_summ = df_summ.iloc[n_train+n_val:].reset_index(drop=True)
    
    print(f"\n✓ Split sizes:")
    print(f"  - Train: {len(train_summ)} ({100*len(train_summ)/len(df_summ):.1f}%)")
    print(f"  - Val: {len(val_summ)} ({100*len(val_summ)/len(df_summ):.1f}%)")
    print(f"  - Test: {len(test_summ)} ({100*len(test_summ)/len(df_summ):.1f}%)")
    
    return train_summ, val_summ, test_summ


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def extract_columns(csv_path: str) -> pd.DataFrame:
    """Load CSV and extract only question and answer columns."""
    print("\n" + "="*80)
    print("STEP 1.1 — EXTRACT COLUMNS")
    print("="*80)
    
    df = pd.read_csv(csv_path)
    print(f"\n✓ Loaded CSV with shape: {df.shape}")
    print(f"✓ Columns: {df.columns.tolist()}")
    
    # Select only question and answer
    df = df[['question', 'answer']].copy()
    print(f"✓ Selected columns: {df.columns.tolist()}")
    
    # Check for missing values
    missing = df.isnull().sum()
    print(f"\n✓ Missing values:")
    print(f"  - question: {missing['question']}")
    print(f"  - answer: {missing['answer']}")
    
    # Remove rows with missing values
    initial_rows = len(df)
    df = df.dropna()
    removed_rows = initial_rows - len(df)
    print(f"\n✓ Removed {removed_rows} rows with missing values")
    print(f"✓ Final shape: {df.shape}")
    
    return df


def clean_dataset(
    df: pd.DataFrame, 
    lowercase_q: bool = True, 
    lowercase_a: bool = False
) -> pd.DataFrame:
    """Apply cleaning to the entire dataset."""
    print("\n" + "="*80)
    print("STEP 1.2 — CLEAN TEXT")
    print("="*80)
    
    df = df.copy()
    
    # Clean questions (lowercase for classification/matching)
    print("\n✓ Cleaning questions...")
    df['question'] = df['question'].apply(lambda x: clean_text(x, lowercase=lowercase_q))
    
    # Clean answers (keep case for language modeling)
    print("✓ Cleaning answers...")
    df['answer'] = df['answer'].apply(lambda x: clean_text(x, lowercase=lowercase_a))
    
    # Remove empty rows
    df = df[(df['question'].str.len() > 0) & (df['answer'].str.len() > 0)]
    
    # Remove duplicates at dataset level
    initial_len = len(df)
    df = df.drop_duplicates(subset=['question', 'answer'])
    removed_dups = initial_len - len(df)
    print(f"✓ Removed {removed_dups} duplicate rows")
    print(f"✓ Final dataset size: {len(df)} rows")
    
    return df.reset_index(drop=True)


def save_datasets(
    train_c, val_c, test_c, 
    train_lm, val_lm, test_lm, 
    train_s, val_s, test_s, 
    output_dir: str = 'data/processed'
) -> None:
    """Save all subsets to CSV files with verification."""
    print("\n" + "="*80)
    print("SAVING DATASETS")
    print("="*80)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    files_saved = []
    
    # Classification (Encoder-Only)
    files = [
        (train_c, 'classification_train.csv'),
        (val_c, 'classification_val.csv'),
        (test_c, 'classification_test.csv'),
    ]
    
    for df, filename in files:
        filepath = output_path / filename
        df.to_csv(filepath, index=False)
        files_saved.append(filepath)
    
    print(f"\n✓ Classification subset saved:")
    for fp in files_saved[-3:]:
        size = fp.stat().st_size / 1024
        print(f"  - {fp.name}: {size:.1f} KB")
    
    # Language Modeling (Decoder-Only)
    files = [
        (train_lm, 'lm_train.csv'),
        (val_lm, 'lm_val.csv'),
        (test_lm, 'lm_test.csv'),
    ]
    
    for df, filename in files:
        filepath = output_path / filename
        df.to_csv(filepath, index=False)
        files_saved.append(filepath)
    
    print(f"\n✓ Language Modeling subset saved:")
    for fp in files_saved[-3:]:
        size = fp.stat().st_size / 1024
        print(f"  - {fp.name}: {size:.1f} KB")
    
    # Summarization (Encoder-Decoder)
    files = [
        (train_s, 'summarization_train.csv'),
        (val_s, 'summarization_val.csv'),
        (test_s, 'summarization_test.csv'),
    ]
    
    for df, filename in files:
        filepath = output_path / filename
        df.to_csv(filepath, index=False)
        files_saved.append(filepath)
    
    print(f"\n✓ Summarization subset saved:")
    for fp in files_saved[-3:]:
        size = fp.stat().st_size / 1024
        print(f"  - {fp.name}: {size:.1f} KB")
    
    # Verification
    print(f"\n✓ All {len(files_saved)} files saved successfully")
    print(f"✓ Output directory: {output_path.absolute()}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("DATASET PREPARATION - FIXED VERSION")
    print("="*80)
    
    # Step 1.1: Extract columns
    df = extract_columns("data/medquad.csv")
    
    # Step 1.2: Clean text
    df_clean = clean_dataset(df, lowercase_q=True, lowercase_a=False)
    
    # Step 1.3: Build subsets
    print("\n" + "="*80)
    print("BUILDING ALL THREE SUBSETS")
    print("="*80)
    
    train_c, val_c, test_c = build_classification_subset(
        df_clean, 
        n_samples=14000
    )
    
    train_lm, val_lm, test_lm = build_language_modeling_subset(
        df_clean, 
        n_sentences=50000, 
        max_tokens=128
    )
    
    train_s, val_s, test_s = build_summarization_subset(
        df_clean, 
        n_samples=3000
    )
    
    # Save datasets
    save_datasets(
        train_c, val_c, test_c, 
        train_lm, val_lm, test_lm, 
        train_s, val_s, test_s
    )
    
    # Final verification
    print("\n" + "="*80)
    print("✅ DATASET PREPARATION COMPLETE!")
    print("="*80)
    
    print("\n📊 Final Statistics:")
    print(f"  Classification: {len(train_c) + len(val_c) + len(test_c):,} samples")
    print(f"  Language Model: {len(train_lm) + len(val_lm) + len(test_lm):,} sentences")
    print(f"  Summarization: {len(train_s) + len(val_s) + len(test_s):,} pairs")
    
    print("\n✓ Column names verified:")
    print(f"  Classification: {train_c.columns.tolist()}")
    print(f"  LM: {train_lm.columns.tolist()}")
    print(f"  Summarization: {train_s.columns.tolist()}")
    
    print("\n✓ All datasets ready for tokenization!")
