"""
Training script for MiniGPT-MedLM (Decoder-Only Transformer).

Trains a GPT-style autoregressive language model on medical answer texts.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import json
import os
import sys
import time
from pathlib import Path

# Add model_decoder_only to path
sys.path.append(str(Path(__file__).parent))

from gpt_model import MiniGPT_MedLM
from masks import generate_causal_mask, generate_padding_mask, combine_masks
from utils import (
    MedLMDataset,
    save_model,
    count_parameters
)

# Add parent directory for tokenizer
sys.path.append(str(Path(__file__).parent.parent))
from step2_tokenizer import Tokenizer


# ============================================================================
# CONFIGURATION
# ============================================================================

# Model hyperparameters
D_MODEL = 128
NUM_HEADS = 4
NUM_LAYERS = 2
FFN_DIM = 512
MAX_SEQ_LEN = 128
DROPOUT = 0.1

# Training hyperparameters
BATCH_SIZE = 32
EPOCHS = 8
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

# Paths
TOKENIZER_PATH = 'artifacts/tokenizer.json'
MODEL_SAVE_DIR = 'artifacts/medLM_models'
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_dataset():
    """Load dataset from existing processed CSV files."""
    import pandas as pd
    
    train_path = 'data/processed/lm_train.csv'
    val_path = 'data/processed/lm_val.csv'
    test_path = 'data/processed/lm_test.csv'
    
    # Check if files exist
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"{train_path} not found!\n"
            "Expected processed language modeling datasets in data/processed/"
        )
    
    # Load texts from CSV files
    print(f"   Loading from: {train_path}")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    train_texts = train_df['text'].tolist()
    val_texts = val_df['text'].tolist()
    test_texts = test_df['text'].tolist()
    
    print(f"   ✓ Train: {len(train_texts)} texts")
    print(f"   ✓ Val: {len(val_texts)} texts")
    print(f"   ✓ Test: {len(test_texts)} texts")
    
    # Combine train + val for training (use test for validation)
    combined_texts = train_texts + val_texts
    
    return combined_texts, test_texts


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_epoch(model, train_loader, optimizer, criterion, device, pad_token_id):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)  # [batch, seq_len-1]
        target_ids = batch['target_ids'].to(device)  # [batch, seq_len-1]
        
        # Generate masks
        seq_len = input_ids.size(1)
        causal_mask = generate_causal_mask(seq_len, device=device)
        padding_mask = generate_padding_mask(input_ids, pad_token_id)
        mask = combine_masks(causal_mask, padding_mask)
        
        # Forward pass
        logits = model(input_ids, mask=mask)  # [batch, seq_len, vocab_size]
        
        # Compute loss (reshape for CrossEntropyLoss)
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),  # [batch*seq_len, vocab_size]
            target_ids.reshape(-1)  # [batch*seq_len]
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def validate(model, val_loader, criterion, device, pad_token_id):
    """Validate model."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            
            # Generate masks
            seq_len = input_ids.size(1)
            causal_mask = generate_causal_mask(seq_len, device=device)
            padding_mask = generate_padding_mask(input_ids, pad_token_id)
            mask = combine_masks(causal_mask, padding_mask)
            
            # Forward pass
            logits = model(input_ids, mask=mask)
            
            # Compute loss
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1)
            )
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


# ============================================================================
# TEXT GENERATION DEMO
# ============================================================================

def generate_sample_text(model, tokenizer, prompts, device, max_new_tokens=20):
    """Generate text continuations for sample prompts."""
    model.eval()
    
    print("\n" + "="*80)
    print("SAMPLE TEXT GENERATION")
    print("="*80)
    
    for prompt_text in prompts:
        # Tokenize prompt
        tokens = tokenizer.encode(prompt_text, add_cls=False, pad=False)
        input_ids = torch.tensor([tokens], device=device)
        
        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.8,
                top_k=50,
                pad_token_id=tokenizer.pad_id
            )
        
        # Decode
        generated_tokens = output_ids[0].cpu().tolist()
        generated_text = tokenizer.decode(generated_tokens)
        
        print(f"\nPrompt: \"{prompt_text}\"")
        print(f"Generated: \"{generated_text}\"")
        print("-" * 80)


# ============================================================================
# MAIN TRAINING
# ============================================================================

def main():
    print("="*80)
    print("TRAINING MiniGPT-MedLM (Decoder-Only Transformer)")
    print("="*80)
    
    # 1. Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = Tokenizer.load(TOKENIZER_PATH)
    vocab_size = tokenizer.vocab_size()
    pad_token_id = tokenizer.pad_id
    print(f"   ✓ Vocabulary size: {vocab_size}")
    print(f"   ✓ Pad token ID: {pad_token_id}")
    
    # 2. Prepare dataset
    print("\n2. Preparing dataset...")
    train_texts, test_texts = prepare_dataset()
    print(f"   ✓ Total training texts: {len(train_texts)}")
    print(f"   ✓ Test (validation) texts: {len(test_texts)}")
    
    # 3. Create dataloaders
    print("\n3. Creating dataloaders...")
    from utils import MedLMDataset
    
    train_dataset = MedLMDataset(train_texts, tokenizer, MAX_SEQ_LEN)
    test_dataset = MedLMDataset(test_texts, tokenizer, MAX_SEQ_LEN)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    val_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    print(f"   ✓ Train batches: {len(train_loader)} ({len(train_dataset)} samples)")
    print(f"   ✓ Val batches: {len(val_loader)} ({len(test_dataset)} samples)")
    
    # 4. Initialize model
    print("\n4. Initializing MiniGPT-MedLM...")
    model = MiniGPT_MedLM(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        ffn_dim=FFN_DIM,
        max_seq_len=MAX_SEQ_LEN,
        dropout=DROPOUT
    ).to(DEVICE)
    
    num_params = count_parameters(model)
    print(f"   ✓ Model parameters: {num_params:,}")
    
    # 5. Setup training
    print("\n5. Setting up training...")
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    print(f"   ✓ Loss: CrossEntropyLoss (ignore_index={pad_token_id})")
    print(f"   ✓ Optimizer: AdamW (lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY})")
    print(f"   ✓ Gradient clipping: {GRAD_CLIP}")
    
    # 6. Training loop
    print("\n6. Training...")
    print("="*80)
    
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE, pad_token_id)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, DEVICE, pad_token_id)
        
        # Track history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        elapsed = time.time() - start_time
        
        # Print progress
        print(f"Epoch {epoch+1}/{EPOCHS}  "
              f"Train Loss: {train_loss:.4f}  "
              f"Val Loss: {val_loss:.4f}  "
              f"({elapsed:.1f}s)")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(
                model, optimizer, epoch, val_loss,
                os.path.join(MODEL_SAVE_DIR, 'miniGPT_medLM.pt')
            )
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # 7. Save training history
    print("\n7. Saving training history...")
    with open(os.path.join(MODEL_SAVE_DIR, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    print(f"   ✓ Saved to {MODEL_SAVE_DIR}/training_history.json")
    
    # 8. Plot loss curves
    print("\n8. Plotting loss curves...")
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss', marker='o')
    plt.plot(history['val_loss'], label='Val Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('MiniGPT-MedLM Training Progress')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plot_path = os.path.join(MODEL_SAVE_DIR, 'training_curve.png')
    plt.savefig(plot_path, dpi=300)
    print(f"   ✓ Saved to {plot_path}")
    
    # 9. Generate sample text
    print("\n9. Generating sample text...")
    prompts = [
        "Diabetes is a condition",
        "Treatment options include",
        "Symptoms may include"
    ]
    generate_sample_text(model, tokenizer, prompts, DEVICE, max_new_tokens=20)
    
    print("\n" + "="*80)
    print("✓ ALL DONE!")
    print("="*80)
    print(f"\nModel saved to: {MODEL_SAVE_DIR}/miniGPT_medLM.pt")
    print(f"Training curve: {MODEL_SAVE_DIR}/training_curve.png")
    print(f"History: {MODEL_SAVE_DIR}/training_history.json")


if __name__ == "__main__":
    main()
