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
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sklearn.metrics import accuracy_score

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
    total_accuracy = 0
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
        
        # Calculate token accuracy
        with torch.no_grad():
            accuracy = calculate_token_accuracy(logits, target_ids, pad_token_id)
            total_accuracy += accuracy
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches, total_accuracy / num_batches


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


def calculate_perplexity(loss):
    """Calculate perplexity from loss."""
    return np.exp(loss)


def calculate_token_accuracy(logits, targets, pad_token_id):
    """Calculate token-level accuracy (ignoring padding)."""
    predictions = torch.argmax(logits, dim=-1)
    
    # Create mask for non-padding tokens
    mask = (targets != pad_token_id)
    
    # Calculate accuracy only on non-padding tokens
    correct = (predictions == targets) & mask
    accuracy = correct.sum().item() / mask.sum().item() if mask.sum().item() > 0 else 0.0
    
    return accuracy


def evaluate_generation_metrics(model, val_loader, tokenizer, device, pad_token_id, max_samples=50):
    """Evaluate generation quality using BLEU, ROUGE, and BERTScore."""
    model.eval()
    
    references_list = []
    hypotheses_list = []
    references_text = []
    hypotheses_text = []
    
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    sample_count = 0
    with torch.no_grad():
        for batch in val_loader:
            if sample_count >= max_samples:
                break
                
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            
            # Generate predictions for each sequence in batch
            for i in range(input_ids.size(0)):
                if sample_count >= max_samples:
                    break
                    
                # Take first few tokens as prompt
                prompt_len = min(10, input_ids.size(1) // 2)
                prompt = input_ids[i:i+1, :prompt_len]
                
                # Generate continuation
                generated = model.generate(
                    prompt,
                    max_new_tokens=input_ids.size(1) - prompt_len,
                    temperature=0.8,
                    top_k=50,
                    pad_token_id=pad_token_id
                )
                
                # Get reference (full sequence)
                reference_tokens = target_ids[i].cpu().tolist()
                # Remove padding
                reference_tokens = [t for t in reference_tokens if t != pad_token_id]
                
                # Get generated tokens (remove prompt)
                generated_tokens = generated[0, prompt_len:].cpu().tolist()
                # Remove padding
                generated_tokens = [t for t in generated_tokens if t != pad_token_id]
                
                # Decode to text
                reference_text = tokenizer.decode(reference_tokens)
                hypothesis_text = tokenizer.decode(generated_tokens)
                
                references_list.append([reference_tokens])
                hypotheses_list.append(generated_tokens)
                references_text.append(reference_text)
                hypotheses_text.append(hypothesis_text)
                
                sample_count += 1
    
    # Calculate BLEU score
    smoothing = SmoothingFunction().method1
    bleu_score = corpus_bleu(references_list, hypotheses_list, smoothing_function=smoothing) * 100
    
    # Calculate ROUGE scores
    rouge_scores = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
    for ref, hyp in zip(references_text, hypotheses_text):
        scores = rouge_scorer_obj.score(ref, hyp)
        rouge_scores['rouge1'] += scores['rouge1'].fmeasure
        rouge_scores['rouge2'] += scores['rouge2'].fmeasure
        rouge_scores['rougeL'] += scores['rougeL'].fmeasure
    
    # Average ROUGE scores
    for key in rouge_scores:
        rouge_scores[key] = (rouge_scores[key] / len(references_text)) * 100
    
    # Calculate BERTScore
    if len(hypotheses_text) > 0 and len(references_text) > 0:
        P, R, F1 = bert_score(hypotheses_text, references_text, lang='en', verbose=False, device=device)
        bert_score_f1 = F1.mean().item() * 100
    else:
        bert_score_f1 = 0.0
    
    return {
        'bleu': bleu_score,
        'rouge1': rouge_scores['rouge1'],
        'rouge2': rouge_scores['rouge2'],
        'rougeL': rouge_scores['rougeL'],
        'bert_score_f1': bert_score_f1
    }


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
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_accuracy': [], 'val_accuracy': [],
        'train_perplexity': [], 'val_perplexity': [],
        'bleu': [], 'rouge1': [], 'rouge2': [], 'rougeL': [], 'bert_score_f1': []
    }
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        # Train
        train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, criterion, DEVICE, pad_token_id)
        train_perplexity = calculate_perplexity(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, DEVICE, pad_token_id)
        val_perplexity = calculate_perplexity(val_loss)
        
        # Calculate validation accuracy
        model.eval()
        val_accuracy = 0
        num_val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                target_ids = batch['target_ids'].to(DEVICE)
                seq_len = input_ids.size(1)
                causal_mask = generate_causal_mask(seq_len, device=DEVICE)
                padding_mask = generate_padding_mask(input_ids, pad_token_id)
                mask = combine_masks(causal_mask, padding_mask)
                logits = model(input_ids, mask=mask)
                val_accuracy += calculate_token_accuracy(logits, target_ids, pad_token_id)
                num_val_batches += 1
        val_accuracy = val_accuracy / num_val_batches if num_val_batches > 0 else 0
        
        # Calculate generation metrics (every 2 epochs to save time)
        if (epoch + 1) % 2 == 0 or epoch == EPOCHS - 1:
            gen_metrics = evaluate_generation_metrics(model, val_loader, tokenizer, DEVICE, pad_token_id, max_samples=30)
            history['bleu'].append(gen_metrics['bleu'])
            history['rouge1'].append(gen_metrics['rouge1'])
            history['rouge2'].append(gen_metrics['rouge2'])
            history['rougeL'].append(gen_metrics['rougeL'])
            history['bert_score_f1'].append(gen_metrics['bert_score_f1'])
            gen_str = f"  BLEU: {gen_metrics['bleu']:.2f}  ROUGE-L: {gen_metrics['rougeL']:.2f}  BERTScore: {gen_metrics['bert_score_f1']:.2f}"
        else:
            gen_str = ""
        
        # Track history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_accuracy'].append(val_accuracy)
        history['train_perplexity'].append(train_perplexity)
        history['val_perplexity'].append(val_perplexity)
        
        elapsed = time.time() - start_time
        
        # Print progress
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"  Loss - Train: {train_loss:.4f}  Val: {val_loss:.4f}")
        print(f"  Accuracy - Train: {train_accuracy:.4f}  Val: {val_accuracy:.4f}")
        print(f"  Perplexity - Train: {train_perplexity:.2f}  Val: {val_perplexity:.2f}")
        if gen_str:
            print(gen_str)
        print(f"  Time: {elapsed:.1f}s")
        print("-" * 80)
        
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
    print(f"\nFinal Metrics:")
    print(f"  Train Loss: {history['train_loss'][-1]:.4f}  |  Val Loss: {history['val_loss'][-1]:.4f}")
    print(f"  Train Accuracy: {history['train_accuracy'][-1]:.4f}  |  Val Accuracy: {history['val_accuracy'][-1]:.4f}")
    print(f"  Train Perplexity: {history['train_perplexity'][-1]:.2f}  |  Val Perplexity: {history['val_perplexity'][-1]:.2f}")
    if len(history['bleu']) > 0:
        print(f"  BLEU: {history['bleu'][-1]:.2f}  |  ROUGE-L: {history['rougeL'][-1]:.2f}  |  BERTScore: {history['bert_score_f1'][-1]:.2f}")
    
    # 7. Save training history
    print("\n7. Saving training history...")
    with open(os.path.join(MODEL_SAVE_DIR, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    print(f"   ✓ Saved to {MODEL_SAVE_DIR}/training_history.json")
    
    # 8. Plot comprehensive metrics
    print("\n8. Plotting evaluation metrics...")
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('MiniGPT-MedLM Comprehensive Evaluation Metrics', fontsize=16, fontweight='bold')
    
    epochs_range = range(1, len(history['train_loss']) + 1)
    
    # Plot 1: Loss
    axes[0, 0].plot(epochs_range, history['train_loss'], label='Train Loss', marker='o', linewidth=2)
    axes[0, 0].plot(epochs_range, history['val_loss'], label='Val Loss', marker='s', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=11)
    axes[0, 0].set_ylabel('Loss', fontsize=11)
    axes[0, 0].set_title('Loss', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Accuracy
    axes[0, 1].plot(epochs_range, history['train_accuracy'], label='Train Accuracy', marker='o', linewidth=2, color='green')
    axes[0, 1].plot(epochs_range, history['val_accuracy'], label='Val Accuracy', marker='s', linewidth=2, color='lime')
    axes[0, 1].set_xlabel('Epoch', fontsize=11)
    axes[0, 1].set_ylabel('Accuracy', fontsize=11)
    axes[0, 1].set_title('Token Accuracy', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Perplexity
    axes[0, 2].plot(epochs_range, history['train_perplexity'], label='Train Perplexity', marker='o', linewidth=2, color='red')
    axes[0, 2].plot(epochs_range, history['val_perplexity'], label='Val Perplexity', marker='s', linewidth=2, color='orange')
    axes[0, 2].set_xlabel('Epoch', fontsize=11)
    axes[0, 2].set_ylabel('Perplexity', fontsize=11)
    axes[0, 2].set_title('Perplexity', fontsize=12, fontweight='bold')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: BLEU Score
    if len(history['bleu']) > 0:
        gen_epochs = [i*2 if i > 0 else EPOCHS for i in range(len(history['bleu']))]
        if gen_epochs[-1] != EPOCHS:
            gen_epochs[-1] = EPOCHS
        axes[1, 0].plot(gen_epochs, history['bleu'], label='BLEU Score', marker='D', linewidth=2, color='purple', markersize=8)
        axes[1, 0].set_xlabel('Epoch', fontsize=11)
        axes[1, 0].set_ylabel('BLEU Score (%)', fontsize=11)
        axes[1, 0].set_title('BLEU Score', fontsize=12, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: ROUGE Scores
    if len(history['rouge1']) > 0:
        gen_epochs = [i*2 if i > 0 else EPOCHS for i in range(len(history['rouge1']))]
        if gen_epochs[-1] != EPOCHS:
            gen_epochs[-1] = EPOCHS
        axes[1, 1].plot(gen_epochs, history['rouge1'], label='ROUGE-1', marker='o', linewidth=2)
        axes[1, 1].plot(gen_epochs, history['rouge2'], label='ROUGE-2', marker='s', linewidth=2)
        axes[1, 1].plot(gen_epochs, history['rougeL'], label='ROUGE-L', marker='^', linewidth=2)
        axes[1, 1].set_xlabel('Epoch', fontsize=11)
        axes[1, 1].set_ylabel('ROUGE Score (%)', fontsize=11)
        axes[1, 1].set_title('ROUGE Scores', fontsize=12, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: BERTScore
    if len(history['bert_score_f1']) > 0:
        gen_epochs = [i*2 if i > 0 else EPOCHS for i in range(len(history['bert_score_f1']))]
        if gen_epochs[-1] != EPOCHS:
            gen_epochs[-1] = EPOCHS
        axes[1, 2].plot(gen_epochs, history['bert_score_f1'], label='BERTScore F1', marker='*', linewidth=2, color='teal', markersize=10)
        axes[1, 2].set_xlabel('Epoch', fontsize=11)
        axes[1, 2].set_ylabel('BERTScore F1 (%)', fontsize=11)
        axes[1, 2].set_title('BERTScore F1', fontsize=12, fontweight='bold')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(MODEL_SAVE_DIR, 'comprehensive_metrics.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved to {plot_path}")
    
    # Create additional plot for loss only (backward compatibility)
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss', marker='o')
    plt.plot(history['val_loss'], label='Val Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('MiniGPT-MedLM Training Progress')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plot_path_simple = os.path.join(MODEL_SAVE_DIR, 'training_curve.png')
    plt.savefig(plot_path_simple, dpi=300)
    print(f"   ✓ Saved to {plot_path_simple}")
    
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
    print(f"Comprehensive metrics plot: {MODEL_SAVE_DIR}/comprehensive_metrics.png")
    print(f"Training curve: {MODEL_SAVE_DIR}/training_curve.png")
    print(f"History: {MODEL_SAVE_DIR}/training_history.json")


if __name__ == "__main__":
    main()
