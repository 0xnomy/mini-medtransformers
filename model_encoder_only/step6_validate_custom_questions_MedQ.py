"""
Custom Question Validation Script
Test MiniBERT on manually written questions to verify real generalization
Uses existing trained model - NO TRAINING
"""

import torch
import torch.nn as nn
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from step3_core_transformer import (
    TokenEmbedding, PositionalEncoding, TransformerEncoderLayer, generate_padding_mask,
    D_MODEL, NUM_HEADS, FFN_DIM, MAX_SEQ_LEN
)
from step2_tokenizer import Tokenizer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_REPO = 'artifacts/medQ_models'

# Model architecture constants
NUM_CLASSES = 8
DROPOUT = 0.4

# Define model class (same as in training script)
class MiniBERTMedQ(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, ffn_dim, num_classes, dropout=0.4):
        super().__init__()
        self.embedding = TokenEmbedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, MAX_SEQ_LEN)
        self.encoder1 = TransformerEncoderLayer(d_model, num_heads, ffn_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.encoder2 = TransformerEncoderLayer(d_model, num_heads, ffn_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.cls_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, x, padding_mask=None):
        x = self.embedding(x)
        x = self.pos_enc(x)
        x = self.encoder1(x, padding_mask)
        x = self.dropout1(x)
        x = self.encoder2(x, padding_mask)
        x = self.dropout2(x)
        cls_emb = x[:, 0, :]
        cls_emb = self.cls_dropout(cls_emb)
        logits = self.classifier(cls_emb)
        return logits

# ==================== CUSTOM TEST QUESTIONS ====================
# These questions are manually written and NOT from the training/test set
CUSTOM_QUESTIONS = [
    # Definition questions
    ("What is hypertension?", "definition"),
    ("Can you explain what diabetes mellitus is?", "definition"),
    ("What does rheumatoid arthritis mean?", "definition"),
    
    # Symptom questions
    ("What are the main symptoms of asthma?", "symptom"),
    ("How do I know if I have a fever?", "symptom"),
    ("What signs should I look for in pneumonia?", "symptom"),
    
    # Treatment questions
    ("What is the best treatment for a migraine?", "treatment"),
    ("How can I treat a common cold?", "treatment"),
    ("What medications help with arthritis?", "treatment"),
    
    # Cause questions
    ("What causes type 2 diabetes?", "cause"),
    ("Why do people get heart disease?", "cause"),
    ("What causes migraines?", "cause"),
    
    # Diagnosis questions
    ("How is cancer diagnosed?", "diagnosis"),
    ("What tests are needed to diagnose thyroid disease?", "diagnosis"),
    ("How do doctors test for COVID-19?", "diagnosis"),
    
    # Genetics questions
    ("Is cystic fibrosis inherited?", "genetics"),
    ("Can hemophilia be genetic?", "genetics"),
    ("Are autism spectrum disorders hereditary?", "genetics"),
    
    # Risk questions
    ("Who is at risk for stroke?", "risk"),
    ("What populations are susceptible to malaria?", "risk"),
    ("Who is more likely to get Alzheimer's?", "risk"),
    
    # General/unclear questions
    ("Tell me about health.", "general"),
    ("What do you know about medicine?", "general"),
    ("Information about hospitals?", "general"),
]

# ==================== LOAD MODEL & TOKENIZER ====================

print("="*80)
print("CUSTOM QUESTION VALIDATION")
print("="*80)

print("\n1. Loading tokenizer...")
tokenizer = Tokenizer.load('artifacts/tokenizer.json')
vocab_size = tokenizer.vocab_size()
print(f"   ✓ Tokenizer loaded (vocab size: {vocab_size})")

print("\n2. Loading trained model...")
# Check if model exists
model_path = os.path.join(MODEL_REPO, 'miniBERT_medQ_best.pt')
if not os.path.exists(model_path):
    print(f"   ✗ Model not found at {model_path}")
    print("   Please train the model first: python miniBERT_medQ_final.py --use_improved")
    exit(1)

# Load model architecture and weights
model = MiniBERTMedQ(vocab_size, D_MODEL, NUM_HEADS, FFN_DIM, NUM_CLASSES, DROPOUT).to(DEVICE)
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.eval()
print(f"   ✓ Model loaded from {model_path}")

# Load label mapping
print("\n3. Loading label mapping...")
results_path = os.path.join(MODEL_REPO, 'results.json')
if not os.path.exists(results_path):
    print(f"   ✗ Results file not found at {results_path}")
    print("   Please train the model first: python miniBERT_medQ_final.py --use_improved")
    exit(1)

with open(results_path, 'r') as f:
    results = json.load(f)
    id_to_label = {int(k): v for k, v in results['id_to_label'].items()}
    print(f"   ✓ Loaded {len(id_to_label)} class labels")

# ==================== RUN PREDICTIONS ====================

print("\n" + "="*80)
print("TESTING CUSTOM QUESTIONS")
print("="*80)

correct = 0
total = 0
results_list = []

print(f"\n{'Question':<55} | {'Expected':<12} | {'Predicted':<12} | {'Confidence':<10} | {'Status':<8}")
print("-" * 115)

with torch.no_grad():
    for question, expected_label in CUSTOM_QUESTIONS:
        # Tokenize
        tokens = tokenizer.encode(question, add_cls=True, pad=True)
        x = torch.tensor([tokens], dtype=torch.long).to(DEVICE)
        mask = generate_padding_mask(x).to(DEVICE)
        
        # Predict
        logits = model(x, mask)
        probs = torch.softmax(logits, dim=1)
        pred_id = probs.argmax(dim=1).item()
        confidence = probs.squeeze().max().item()
        pred_label = id_to_label[pred_id]
        
        # Check if correct
        is_correct = (pred_label == expected_label)
        status = "✓ PASS" if is_correct else "✗ FAIL"
        if is_correct:
            correct += 1
        total += 1
        
        # Truncate question for display
        q_display = question[:52] + "..." if len(question) > 52 else question
        print(f"{q_display:<55} | {expected_label:<12} | {pred_label:<12} | {confidence:<10.3f} | {status:<8}")
        
        results_list.append({
            "question": question,
            "expected": expected_label,
            "predicted": pred_label,
            "confidence": float(confidence),
            "correct": is_correct
        })

# ==================== SUMMARY ====================

accuracy = correct / total if total > 0 else 0
print("\n" + "="*80)
print("VALIDATION SUMMARY")
print("="*80)

print(f"\n✓ Correct predictions: {correct}/{total}")
print(f"✓ Custom Question Accuracy: {accuracy*100:.1f}%")

# Breakdown by category
print(f"\n✓ Breakdown by Question Type:")
categories = {}
for result in results_list:
    cat = result['expected']
    if cat not in categories:
        categories[cat] = {'correct': 0, 'total': 0}
    categories[cat]['total'] += 1
    if result['correct']:
        categories[cat]['correct'] += 1

for cat in sorted(categories.keys()):
    cat_acc = categories[cat]['correct'] / categories[cat]['total']
    print(f"  - {cat:<12}: {categories[cat]['correct']}/{categories[cat]['total']} ({cat_acc*100:.1f}%)")

# ==================== ANALYSIS ====================

print(f"\n" + "="*80)
print("ANALYSIS")
print("="*80)

if accuracy >= 0.90:
    print("✅ EXCELLENT: Model generalizes very well to custom questions!")
    print("   High real-world applicability confirmed.")
elif accuracy >= 0.80:
    print("✅ GOOD: Model shows decent generalization to new questions.")
    print("   Some patterns may need refinement, but acceptable for deployment.")
elif accuracy >= 0.70:
    print("⚠️  FAIR: Model has moderate generalization capability.")
    print("   Consider collecting more diverse training data or adjusting thresholds.")
else:
    print("❌ POOR: Model may be overfitting to training data.")
    print("   Recommend data augmentation or model architecture changes.")

# Save detailed results
with open(os.path.join(MODEL_REPO, 'custom_validation_results.json'), 'w') as f:
    json.dump({
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'dropout': DROPOUT,
        'num_classes': NUM_CLASSES,
        'categories': categories,
        'detailed_results': results_list
    }, f, indent=2)

print(f"\n✓ Detailed results saved to {os.path.join(MODEL_REPO, 'custom_validation_results.json')}")
print("\n✓ Done!")
