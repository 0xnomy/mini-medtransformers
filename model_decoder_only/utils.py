"""
Utilities for MiniGPT-MedLM training.

Includes:
- Dataset loading and preprocessing
- Tokenization utilities
- DataLoader creation
- Model save/load helpers
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys

# Add parent directory to path to import tokenizer
sys.path.append(str(Path(__file__).parent.parent))
from step2_tokenizer import Tokenizer


class MedLMDataset(Dataset):
    """
    Dataset for language modeling on medical text.
    
    Prepares data for next-token prediction:
        input_ids:  tokens[:-1]
        target_ids: tokens[1:]
    
    Args:
        texts: List of answer texts
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
    """
    
    def __init__(self, texts: list, tokenizer: Tokenizer, max_length: int = 128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Preprocess all texts
        self.samples = []
        for text in texts:
            tokens = tokenizer.encode(text, add_cls=False, pad=False)
            
            # Skip if too short
            if len(tokens) < 2:
                continue
            
            # Truncate if too long
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            
            # Pad to max_length
            tokens = tokens + [tokenizer.pad_id] * (max_length - len(tokens))
            
            self.samples.append(tokens)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        tokens = self.samples[idx]
        
        # Shift for next-token prediction
        input_ids = tokens[:-1]   # All tokens except last
        target_ids = tokens[1:]   # All tokens except first
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long)
        }


def load_medqa_answers(file_path: str, max_samples: int = None) -> list:
    """
    Load medical answer texts from JSONL file.
    
    Expected format:
        {"answer": "Medical answer text..."}
    
    Args:
        file_path: Path to JSONL file
        max_samples: Limit number of samples (None = load all)
    
    Returns:
        List of answer texts
    """
    texts = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            
            try:
                data = json.loads(line.strip())
                answer = data.get('answer', '').strip()
                
                if answer and len(answer) > 10:  # Filter very short answers
                    texts.append(answer)
            except json.JSONDecodeError:
                continue
    
    return texts


def create_dataloaders(
    texts: list,
    tokenizer: Tokenizer,
    batch_size: int = 32,
    max_length: int = 128,
    train_split: float = 0.9,
    shuffle: bool = True
):
    """
    Create train and validation dataloaders.
    
    Args:
        texts: List of text samples
        tokenizer: Tokenizer instance
        batch_size: Batch size
        max_length: Maximum sequence length
        train_split: Fraction for training (rest is validation)
        shuffle: Whether to shuffle training data
    
    Returns:
        train_loader, val_loader
    """
    # Split data
    split_idx = int(len(texts) * train_split)
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]
    
    print(f"Train samples: {len(train_texts)}")
    print(f"Val samples: {len(val_texts)}")
    
    # Create datasets
    train_dataset = MedLMDataset(train_texts, tokenizer, max_length)
    val_dataset = MedLMDataset(val_texts, tokenizer, max_length)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader


def save_model(model, optimizer, epoch, loss, save_path: str):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss
        save_path: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, save_path)
    print(f"✓ Model saved to {save_path}")


def load_model(model, optimizer, load_path: str, device):
    """
    Load model checkpoint.
    
    Args:
        model: Model to load into
        optimizer: Optimizer to load state into
        load_path: Path to checkpoint
        device: Device to load on
    
    Returns:
        epoch, loss
    """
    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"✓ Model loaded from {load_path} (epoch {epoch}, loss {loss:.4f})")
    return epoch, loss


def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def extract_answers_from_medquad(medquad_csv: str, output_jsonl: str, max_samples: int = 50000):
    """
    Extract answer texts from MedQuAD CSV and save to JSONL format.
    
    This is a utility to prepare the dataset from the existing MedQuAD data.
    
    Args:
        medquad_csv: Path to data/medquad.csv
        output_jsonl: Path to save JSONL file
        max_samples: Maximum number of samples to extract
    """
    import pandas as pd
    
    print(f"Extracting answers from {medquad_csv}...")
    
    # Load MedQuAD
    df = pd.read_csv(medquad_csv)
    
    # Extract answer column
    if 'answer' not in df.columns:
        raise ValueError("CSV must have 'answer' column")
    
    answers = df['answer'].dropna().tolist()
    
    # Limit samples
    if max_samples:
        answers = answers[:max_samples]
    
    # Save to JSONL
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for answer in answers:
            answer = str(answer).strip()
            if len(answer) > 10:  # Filter very short answers
                json.dump({'answer': answer}, f)
                f.write('\n')
    
    print(f"✓ Saved {len(answers)} answers to {output_jsonl}")


def test_dataset():
    """Test dataset and dataloader creation."""
    print("Testing MedLMDataset...")
    
    # Create mock tokenizer
    from step2_tokenizer import Tokenizer
    tokenizer = Tokenizer.load('../artifacts/tokenizer.json')
    
    # Mock data
    texts = [
        "Diabetes is a chronic condition that affects blood sugar levels.",
        "Treatment includes insulin therapy and lifestyle modifications.",
        "Symptoms include increased thirst, frequent urination, and fatigue."
    ]
    
    # Test 1: Dataset creation
    print("\n1. Testing dataset creation...")
    dataset = MedLMDataset(texts, tokenizer, max_length=32)
    print(f"✓ Dataset size: {len(dataset)}")
    
    # Test 2: Sample retrieval
    print("\n2. Testing sample retrieval...")
    sample = dataset[0]
    assert 'input_ids' in sample and 'target_ids' in sample
    assert sample['input_ids'].shape[0] == 31  # max_length - 1
    assert sample['target_ids'].shape[0] == 31
    print(f"✓ Sample shapes: input={sample['input_ids'].shape}, target={sample['target_ids'].shape}")
    
    # Test 3: DataLoader creation
    print("\n3. Testing dataloader creation...")
    train_loader, val_loader = create_dataloaders(
        texts, tokenizer, batch_size=2, max_length=32, train_split=0.67
    )
    
    batch = next(iter(train_loader))
    print(f"✓ Batch keys: {batch.keys()}")
    print(f"✓ Batch input_ids shape: {batch['input_ids'].shape}")
    
    print("\n✅ All dataset tests passed!")


if __name__ == "__main__":
    test_dataset()
