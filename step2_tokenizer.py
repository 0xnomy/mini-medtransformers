"""
STEP 2: Build Tokenizer from Scratch
Mini-MedTransformers Project

Tokenizer Features:
- Whitespace-based tokenization
- Vocabulary built from training datasets
- Special tokens: <pad>, <unk>, <bos>, <eos>, <cls>, <sep>
- Unicode normalization (NFC)
- Configurable lowercasing
- OOV handling with <unk> token
- Batch encoding with padding/truncation
- Complete reproducibility

Usage:
    python step2_tokenizer.py [--max_vocab_size 10000] [--lowercase True]
"""

import json
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, List, Iterable, Tuple, Optional
import numpy as np
import pandas as pd


# ============================================================================
# UTILITIES
# ============================================================================

def normalize_text(text: str) -> str:
    """Apply Unicode NFC normalization and clean whitespace."""
    # Unicode NFC normalization
    text = unicodedata.normalize('NFC', text)
    
    # Replace various whitespace characters with space
    text = re.sub(r'[\u00A0\u1680\u2000-\u200B\u202F\u205F\u3000]', ' ', text)
    
    # Remove zero-width characters
    text = re.sub(r'[\u200C\u200D\u200E\u200F\uFEFF]', '', text)
    
    # Normalize multiple spaces to single space
    text = re.sub(r' +', ' ', text)
    
    return text.strip()


def load_dataset_texts(
    classification_path: str,
    lm_path: str,
    summarization_path: str,
    split: str = 'train'
) -> List[str]:
    """
    Load texts from all three datasets for vocabulary building.
    
    Args:
        classification_path: Path to classification CSV
        lm_path: Path to LM CSV
        summarization_path: Path to summarization CSV
        split: 'train', 'val', or 'test'
    
    Returns:
        List of all texts from all datasets
    """
    texts = []
    
    # Classification: use 'text' column (questions)
    if os.path.exists(classification_path):
        df = pd.read_csv(classification_path)
        if 'text' in df.columns:
            texts.extend(df['text'].dropna().tolist())
    
    # Language Modeling: use 'text' column (sentences)
    if os.path.exists(lm_path):
        df = pd.read_csv(lm_path)
        if 'text' in df.columns:
            texts.extend(df['text'].dropna().tolist())
    
    # Summarization: use 'input' and 'target' columns
    if os.path.exists(summarization_path):
        df = pd.read_csv(summarization_path)
        if 'input' in df.columns:
            texts.extend(df['input'].dropna().tolist())
        if 'target' in df.columns:
            texts.extend(df['target'].dropna().tolist())
    
    return texts


# ============================================================================
# TOKENIZER CLASS
# ============================================================================

class Tokenizer:
    """
    Whitespace-based tokenizer with vocabulary management.
    
    Features:
    - Builds vocabulary from training data
    - Special tokens: <pad>, <unk>, <bos>, <eos>, <cls>, <sep>
    - Unicode normalization
    - Configurable lowercasing
    - Batch encoding with padding/truncation
    - Save/load functionality
    """
    
    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        special_tokens: Optional[List[str]] = None,
        max_seq_len: int = 128,
        lowercase: bool = True
    ):
        """
        Initialize tokenizer.
        
        Args:
            vocab: Pre-built token-to-id mapping (if None, build from data)
            special_tokens: List of special tokens (default: standard set)
            max_seq_len: Maximum sequence length
            lowercase: Whether to lowercase text before tokenization
        """
        self.max_seq_len = max_seq_len
        self.lowercase = lowercase
        
        # Default special tokens in order
        self.special_tokens = special_tokens or [
            '<pad>',  # Index 0 (must be first)
            '<unk>',  # Index 1
            '<bos>',  # Index 2
            '<eos>',  # Index 3
            '<cls>',  # Index 4
            '<sep>',  # Index 5
        ]
        
        # Initialize vocabulary
        if vocab is None:
            self.token_to_id = {}
            self.id_to_token = {}
        else:
            self.token_to_id = vocab
            self.id_to_token = {v: k for k, v in vocab.items()}
        
        # Get special token indices
        self.pad_id = self.token_to_id.get('<pad>', 0)
        self.unk_id = self.token_to_id.get('<unk>', 1)
        self.bos_id = self.token_to_id.get('<bos>', 2)
        self.eos_id = self.token_to_id.get('<eos>', 3)
        self.cls_id = self.token_to_id.get('<cls>', 4)
        self.sep_id = self.token_to_id.get('<sep>', 5)
    
    def build_vocab(
        self,
        texts: Iterable[str],
        max_vocab_size: int = 10000,
        min_freq: int = 1
    ) -> None:
        """
        Build vocabulary from texts.
        
        Args:
            texts: Iterable of text strings
            max_vocab_size: Maximum vocabulary size (excluding specials)
            min_freq: Minimum token frequency to include
        """
        print("\n" + "="*80)
        print("BUILDING VOCABULARY")
        print("="*80)
        
        # Count token frequencies
        token_counter = Counter()
        total_texts = 0
        
        for text in texts:
            total_texts += 1
            if total_texts % 10000 == 0:
                print(f"  Processed {total_texts} texts...")
            
            # Normalize and optionally lowercase
            text = normalize_text(text)
            if self.lowercase:
                text = text.lower()
            
            # Whitespace tokenization
            tokens = text.split()
            token_counter.update(tokens)
        
        print(f"✓ Processed {total_texts} texts")
        print(f"✓ Found {len(token_counter):,} unique tokens")
        
        # Filter by frequency
        frequent_tokens = [
            token for token, count in token_counter.items()
            if count >= min_freq
        ]
        print(f"✓ Tokens with freq >= {min_freq}: {len(frequent_tokens):,}")
        
        # Sort by frequency
        frequent_tokens.sort(key=lambda t: token_counter[t], reverse=True)
        
        # Limit to max_vocab_size
        frequent_tokens = frequent_tokens[:max_vocab_size]
        print(f"✓ Keeping top {len(frequent_tokens):,} tokens")
        
        # Build vocab: special tokens first, then frequent tokens
        self.token_to_id = {}
        idx = 0
        
        # Reserve indices for special tokens (order matters)
        for special_token in self.special_tokens:
            self.token_to_id[special_token] = idx
            idx += 1
        
        # Add frequent tokens
        for token in frequent_tokens:
            if token not in self.token_to_id:  # Skip if already special token
                self.token_to_id[token] = idx
                idx += 1
        
        # Build reverse mapping
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        
        # Update special token indices
        self.pad_id = self.token_to_id['<pad>']
        self.unk_id = self.token_to_id['<unk>']
        self.bos_id = self.token_to_id['<bos>']
        self.eos_id = self.token_to_id['<eos>']
        self.cls_id = self.token_to_id['<cls>']
        self.sep_id = self.token_to_id['<sep>']
        
        print(f"✓ Final vocabulary size: {self.vocab_size():,}")
        print(f"✓ <pad> index: {self.pad_id}")
        print(f"✓ <unk> index: {self.unk_id}")
        
        # Print top tokens
        print(f"\n✓ Top 50 most frequent tokens:")
        top_tokens = sorted(
            token_counter.items(),
            key=lambda x: x[1],
            reverse=True
        )[:50]
        for i, (token, count) in enumerate(top_tokens, 1):
            print(f"  {i:2d}. {token:30s} {count:8,d}")
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text using whitespace splitting.
        
        Args:
            text: Input text
        
        Returns:
            List of tokens
        """
        # Normalize
        text = normalize_text(text)
        
        # Optionally lowercase
        if self.lowercase:
            text = text.lower()
        
        # Whitespace split
        tokens = text.split()
        
        return tokens
    
    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        add_cls: bool = False,
        pad: bool = True,
        truncation_strategy: str = 'tail'
    ) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_bos: Add <bos> token at start
            add_eos: Add <eos> token at end
            add_cls: Add <cls> token at start (for classification)
            pad: Whether to pad/truncate to max_seq_len
            truncation_strategy: 'head' or 'tail' (which end to truncate)
        
        Returns:
            List of token IDs
        """
        # Tokenize
        tokens = self.tokenize(text)
        
        # Convert to IDs (OOV -> unk)
        ids = [self.token_to_id.get(token, self.unk_id) for token in tokens]
        
        # Add special tokens
        if add_cls:
            ids = [self.cls_id] + ids
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        
        # Pad or truncate
        if pad:
            ids = self.pad_sequence(ids, pad_to_max=True, truncation_strategy=truncation_strategy)
        
        return ids
    
    def decode(
        self,
        ids: List[int],
        skip_special_tokens: bool = True
    ) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
        
        Returns:
            Decoded text
        """
        tokens = []
        for idx in ids:
            if idx not in self.id_to_token:
                continue
            
            token = self.id_to_token[idx]
            
            # Skip special tokens if requested
            if skip_special_tokens and token in self.special_tokens:
                continue
            
            tokens.append(token)
        
        return ' '.join(tokens)
    
    def pad_sequence(
        self,
        ids: List[int],
        pad_to_max: bool = True,
        truncation_strategy: str = 'tail'
    ) -> List[int]:
        """
        Pad or truncate sequence to max_seq_len.
        
        Args:
            ids: List of token IDs
            pad_to_max: Whether to pad/truncate
            truncation_strategy: 'head' or 'tail'
        
        Returns:
            Padded/truncated sequence
        """
        if not pad_to_max:
            return ids
        
        if len(ids) > self.max_seq_len:
            # Truncate
            if truncation_strategy == 'head':
                ids = ids[-(self.max_seq_len):]
            else:  # 'tail'
                ids = ids[:self.max_seq_len]
        else:
            # Pad with pad_id
            ids = ids + [self.pad_id] * (self.max_seq_len - len(ids))
        
        return ids
    
    def batch_encode(
        self,
        texts: List[str],
        **kwargs
    ) -> np.ndarray:
        """
        Encode multiple texts as a batch.
        
        Args:
            texts: List of texts
            **kwargs: Additional arguments for encode()
        
        Returns:
            2D numpy array of shape (N, max_seq_len) with dtype int32
        """
        encoded = [self.encode(text, **kwargs) for text in texts]
        
        # Stack into array
        array = np.array(encoded, dtype=np.int32)
        
        return array
    
    def vocab_size(self) -> int:
        """Return vocabulary size including special tokens."""
        return len(self.token_to_id)
    
    def token_to_idx(self, token: str) -> int:
        """Convert token to ID."""
        return self.token_to_id.get(token, self.unk_id)
    
    def idx_to_token(self, idx: int) -> str:
        """Convert ID to token."""
        return self.id_to_token.get(idx, '<unk>')
    
    def save(self, path: str) -> None:
        """Save tokenizer to JSON."""
        config = {
            'token_to_id': self.token_to_id,
            'special_tokens': self.special_tokens,
            'max_seq_len': self.max_seq_len,
            'lowercase': self.lowercase,
        }
        
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Tokenizer saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'Tokenizer':
        """Load tokenizer from JSON."""
        with open(path, 'r') as f:
            config = json.load(f)
        
        tokenizer = cls(
            vocab=config['token_to_id'],
            special_tokens=config['special_tokens'],
            max_seq_len=config['max_seq_len'],
            lowercase=config['lowercase']
        )
        
        print(f"✓ Tokenizer loaded from {path}")
        return tokenizer


# ============================================================================
# VOCABULARY UTILITIES
# ============================================================================

def save_vocab_file(tokenizer: Tokenizer, path: str) -> None:
    """Save vocabulary as one token per line."""
    with open(path, 'w') as f:
        for idx in range(tokenizer.vocab_size()):
            token = tokenizer.idx_to_token(idx)
            f.write(f"{token}\n")
    
    print(f"✓ Vocabulary saved to {path} ({tokenizer.vocab_size():,} tokens)")


# ============================================================================
# TESTING AND VALIDATION
# ============================================================================

def compute_oov_rate(
    tokenizer: Tokenizer,
    texts: List[str]
) -> float:
    """
    Compute OOV (out-of-vocabulary) rate for a list of texts.
    
    Args:
        tokenizer: Tokenizer instance
        texts: List of texts to evaluate
    
    Returns:
        Fraction of tokens mapped to <unk>
    """
    total_tokens = 0
    oov_tokens = 0
    
    for text in texts:
        tokens = tokenizer.tokenize(text)
        total_tokens += len(tokens)
        
        for token in tokens:
            if token not in tokenizer.token_to_id:
                oov_tokens += 1
    
    if total_tokens == 0:
        return 0.0
    
    return oov_tokens / total_tokens


def test_tokenizer(tokenizer: Tokenizer, data_dir: str = 'data/processed') -> Dict:
    """
    Run comprehensive tokenizer tests.
    
    Args:
        tokenizer: Tokenizer instance
        data_dir: Directory containing processed datasets
    
    Returns:
        Dictionary of test results
    """
    print("\n" + "="*80)
    print("TOKENIZER VALIDATION")
    print("="*80)
    
    results = {}
    
    # =====================================================================
    # Test 1: Pad ID
    # =====================================================================
    print(f"\n✓ Test 1: Special Token Indices")
    print(f"  - <pad> ID: {tokenizer.pad_id}")
    assert tokenizer.pad_id == 0, "❌ <pad> ID must be 0"
    print(f"  ✅ <pad> ID is correctly 0")
    
    # =====================================================================
    # Test 2: Vocabulary Size
    # =====================================================================
    print(f"\n✓ Test 2: Vocabulary Size")
    vocab_size = tokenizer.vocab_size()
    print(f"  - Total vocab size: {vocab_size:,}")
    results['vocab_size'] = vocab_size
    
    # =====================================================================
    # Test 3: ID Collision Check
    # =====================================================================
    print(f"\n✓ Test 3: ID Collision Check (top 100 tokens)")
    collisions = 0
    for token, idx in list(tokenizer.token_to_id.items())[:100]:
        if tokenizer.id_to_token.get(idx) != token:
            collisions += 1
    
    if collisions == 0:
        print(f"  ✅ No ID collisions found")
    else:
        print(f"  ❌ Found {collisions} ID collisions")
    results['collisions'] = collisions
    
    # =====================================================================
    # Test 4: OOV Rates on All Datasets
    # =====================================================================
    print(f"\n✓ Test 4: OOV Rates by Dataset")
    oov_results = {}
    
    # Classification
    for split in ['train', 'val', 'test']:
        path = f"{data_dir}/classification_{split}.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            texts = df['text'].dropna().tolist()
            oov_rate = compute_oov_rate(tokenizer, texts)
            oov_results[f'classification_{split}'] = oov_rate
            status = "✅" if oov_rate < 0.10 else "⚠️"
            print(f"  {status} Classification {split:5s}: {oov_rate:.4f} ({oov_rate*100:.2f}%)")
    
    # Language Modeling
    for split in ['train', 'val', 'test']:
        path = f"{data_dir}/lm_{split}.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            texts = df['text'].dropna().tolist()
            oov_rate = compute_oov_rate(tokenizer, texts)
            oov_results[f'lm_{split}'] = oov_rate
            status = "✅" if oov_rate < 0.10 else "⚠️"
            print(f"  {status} LM {split:5s}:                {oov_rate:.4f} ({oov_rate*100:.2f}%)")
    
    # Summarization
    for split in ['train', 'val', 'test']:
        path = f"{data_dir}/summarization_{split}.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            texts = []
            if 'input' in df.columns:
                texts.extend(df['input'].dropna().tolist())
            if 'target' in df.columns:
                texts.extend(df['target'].dropna().tolist())
            oov_rate = compute_oov_rate(tokenizer, texts)
            oov_results[f'summarization_{split}'] = oov_rate
            status = "✅" if oov_rate < 0.10 else "⚠️"
            print(f"  {status} Summarization {split:5s}: {oov_rate:.4f} ({oov_rate*100:.2f}%)")
    
    results['oov_rates'] = oov_results
    
    # =====================================================================
    # Test 5: Sample Encodings & Decodings
    # =====================================================================
    print(f"\n✓ Test 5: Sample Encodings & Decodings")
    
    samples_tested = 0
    decoding_errors = 0
    
    for dataset_name, path in [
        ('Classification', f"{data_dir}/classification_train.csv"),
        ('LM', f"{data_dir}/lm_train.csv"),
        ('Summarization', f"{data_dir}/summarization_train.csv"),
    ]:
        if not os.path.exists(path):
            continue
        
        df = pd.read_csv(path)
        
        if dataset_name == 'Classification' and 'text' in df.columns:
            texts = df['text'].head(5).tolist()
            sample_col = 'text'
        elif dataset_name == 'LM' and 'text' in df.columns:
            texts = df['text'].head(5).tolist()
            sample_col = 'text'
        elif dataset_name == 'Summarization':
            texts = []
            if 'input' in df.columns:
                texts.extend(df['input'].head(3).tolist())
            if 'target' in df.columns:
                texts.extend(df['target'].head(2).tolist())
            sample_col = 'input/target'
        else:
            continue
        
        print(f"\n  {dataset_name}:")
        for i, text in enumerate(texts[:5], 1):
            # Encode
            ids = tokenizer.encode(text, pad=True)
            
            # Decode
            decoded = tokenizer.decode(ids, skip_special_tokens=True)
            
            # Show sample
            print(f"    Sample {i}:")
            print(f"      Original:  {text[:80]}...")
            print(f"      Tokens:    {tokenizer.tokenize(text)[:10]}...")
            print(f"      IDs:       {ids[:15]}...")
            print(f"      Decoded:   {decoded[:80]}...")
            
            samples_tested += 1
    
    results['samples_tested'] = samples_tested
    
    # =====================================================================
    # Test 6: Batch Encoding
    # =====================================================================
    print(f"\n✓ Test 6: Batch Encoding")
    
    # Load classification test set
    path = f"{data_dir}/classification_test.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        texts = df['text'].head(10).tolist()
        
        # Batch encode
        batch = tokenizer.batch_encode(texts, pad=True)
        
        print(f"  - Input: {len(texts)} texts")
        print(f"  - Output shape: {batch.shape}")
        print(f"  - Output dtype: {batch.dtype}")
        print(f"  - Sample IDs (first text): {batch[0, :15]}")
        
        assert batch.shape == (len(texts), tokenizer.max_seq_len), \
            "❌ Batch shape mismatch"
        assert batch.dtype == np.int32, "❌ Batch dtype must be int32"
        print(f"  ✅ Batch encoding OK")
        results['batch_ok'] = True
    
    # =====================================================================
    # Test 7: Edge Cases
    # =====================================================================
    print(f"\n✓ Test 7: Edge Cases")
    
    # Empty string
    empty_ids = tokenizer.encode("", pad=True)
    assert len(empty_ids) == tokenizer.max_seq_len, "❌ Empty string not padded correctly"
    assert all(id == tokenizer.pad_id for id in empty_ids), "❌ Empty string should be all pad tokens"
    print(f"  ✅ Empty string handling OK")
    
    # Very long string
    long_text = " ".join(["word"] * 1000)
    long_ids = tokenizer.encode(long_text, pad=True)
    assert len(long_ids) == tokenizer.max_seq_len, "❌ Long string not truncated correctly"
    print(f"  ✅ Long string truncation OK")
    
    # Unknown tokens
    oov_text = "xyzabc123 unknown_token_xyz"
    oov_ids = tokenizer.encode(oov_text, pad=True)
    assert tokenizer.unk_id in oov_ids, "❌ OOV tokens not mapped to <unk>"
    print(f"  ✅ OOV token handling OK")
    
    results['edge_cases_ok'] = True
    
    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("STEP 2: BUILD TOKENIZER FROM SCRATCH")
    print("="*80)
    
    # Configuration
    max_vocab_size = 10000
    max_seq_len = 128
    lowercase = True
    
    # =====================================================================
    # Load Training Data
    # =====================================================================
    print("\n✓ Loading training datasets...")
    
    data_dir = 'data/processed'
    
    texts = load_dataset_texts(
        classification_path=f"{data_dir}/classification_train.csv",
        lm_path=f"{data_dir}/lm_train.csv",
        summarization_path=f"{data_dir}/summarization_train.csv",
        split='train'
    )
    
    print(f"✓ Loaded {len(texts):,} texts for vocabulary building")
    
    # =====================================================================
    # Build Tokenizer
    # =====================================================================
    print("\n✓ Building tokenizer...")
    tokenizer = Tokenizer(max_seq_len=max_seq_len, lowercase=lowercase)
    tokenizer.build_vocab(texts, max_vocab_size=max_vocab_size, min_freq=1)
    
    # =====================================================================
    # Save Artifacts
    # =====================================================================
    print("\n" + "="*80)
    print("SAVING ARTIFACTS")
    print("="*80)
    
    os.makedirs('artifacts', exist_ok=True)
    
    tokenizer.save('artifacts/tokenizer.json')
    save_vocab_file(tokenizer, 'artifacts/vocab.txt')
    
    # =====================================================================
    # Run Tests
    # =====================================================================
    results = test_tokenizer(tokenizer, data_dir=data_dir)
    
    # =====================================================================
    # Final Report
    # =====================================================================
    print("\n" + "="*80)
    print("✅ FINAL VALIDATION REPORT")
    print("="*80)
    
    print(f"\n✓ Configuration:")
    print(f"  - Max vocab size: {max_vocab_size:,}")
    print(f"  - Max sequence length: {max_seq_len}")
    print(f"  - Lowercase: {lowercase}")
    print(f"  - Actual vocab size: {results['vocab_size']:,}")
    
    print(f"\n✓ Quality Checks:")
    print(f"  - ID collisions: {results['collisions']} (expected: 0) {'✅' if results['collisions'] == 0 else '❌'}")
    print(f"  - Batch encoding: {'✅' if results['batch_ok'] else '❌'}")
    print(f"  - Edge cases: {'✅' if results['edge_cases_ok'] else '❌'}")
    
    print(f"\n✓ OOV Rates (should be < 10% for train sets):")
    oov_ok = True
    for dataset, rate in results['oov_rates'].items():
        status = "✅" if rate < 0.10 else "⚠️"
        print(f"  {status} {dataset:30s}: {rate:.4f}")
        if 'train' in dataset and rate >= 0.10:
            oov_ok = False
    
    print(f"\n✓ Samples tested: {results['samples_tested']}")
    
    print(f"\n✓ Artifacts saved:")
    print(f"  - artifacts/tokenizer.json")
    print(f"  - artifacts/vocab.txt")
    
    # Final status
    print("\n" + "="*80)
    all_pass = (
        results['collisions'] == 0 and
        results['batch_ok'] and
        results['edge_cases_ok'] and
        oov_ok
    )
    
    if all_pass:
        print("✅ ALL CHECKS PASSED - TOKENIZER READY FOR USE")
    else:
        print("⚠️  SOME CHECKS FAILED - REVIEW OUTPUT ABOVE")
    
    print("="*80 + "\n")
    
    return tokenizer


if __name__ == '__main__':
    tokenizer = main()
