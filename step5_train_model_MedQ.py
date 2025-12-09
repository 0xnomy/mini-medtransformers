"""
MiniBERT-MedQ Final: Optimized training with pre-tokenized data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from step3_core_transformer import (
    TokenEmbedding, PositionalEncoding, TransformerEncoderLayer, generate_padding_mask,
    D_MODEL, NUM_HEADS, FFN_DIM, MAX_SEQ_LEN
)
from step2_tokenizer import Tokenizer
import os
import sys
from sklearn.metrics import precision_recall_fscore_support, classification_report, f1_score, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import json
import time
import argparse

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_REPO = 'artifacts/medQ_models'
os.makedirs(MODEL_REPO, exist_ok=True)

# CONFIG
BATCH_SIZE = 20
EPOCHS = 10
EARLY_STOP_PATIENCE = 3
NUM_CLASSES = 8  # Core semantic labels: cause, definition, diagnosis, general, genetics, risk, symptom, treatment
DROPOUT = 0.4  # Increased from 0.2 for better regularization

print(f"Device: {DEVICE}")
print(f"Batch size: {BATCH_SIZE}, Epochs: {EPOCHS}\n")

# ==================== MODEL ====================

class MiniBERTMedQ(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, ffn_dim, num_classes, dropout=0.2):
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

# ==================== PREPROCESS ====================

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train MiniBERT-MedQ classifier')
parser.add_argument('--train_file', type=str, default='data/processed/classification_train.csv',
                    help='Path to training data CSV file')
parser.add_argument('--use_augmented', action='store_true',
                    help='Use augmented training data (classification_train_augmented.csv)')
parser.add_argument('--use_improved', action='store_true',
                    help='Use improved training data with targeted augmentations (classification_train_improved.csv)')
parser.add_argument('--use_conversational', action='store_true',
                    help='Use conversational training data (classification_train_conversational.csv)')
parser.add_argument('--validate_on_custom', action='store_true',
                    help='Use custom questions as validation set instead of random split')
args = parser.parse_args()

# Determine which training file to use
if args.use_conversational and os.path.exists('data/processed/classification_train_conversational.csv'):
    train_file = 'data/processed/classification_train_conversational.csv'
    print("Using conversational training data (natural language augmentations)")
elif args.use_improved and os.path.exists('data/processed/classification_train_improved.csv'):
    train_file = 'data/processed/classification_train_improved.csv'
    print("Using improved training data with targeted augmentations")
elif args.use_augmented:
    train_file = 'data/processed/classification_train_augmented.csv'
    print("Using augmented training data")
else:
    train_file = args.train_file

print("Step 1: Load and preprocess data...")
print(f"Training file: {train_file}")
start = time.time()

tokenizer = Tokenizer.load('artifacts/tokenizer.json')
vocab_size = tokenizer.vocab_size()

# Custom validation questions (conversational/natural language)
CUSTOM_QUESTIONS = [
    ("What is hypertension?", "definition"),
    ("Can you explain what diabetes mellitus is?", "definition"),
    ("What does rheumatoid arthritis mean?", "definition"),
    ("What are the main symptoms of asthma?", "symptom"),
    ("How do I know if I have a fever?", "symptom"),
    ("What signs should I look for in pneumonia?", "symptom"),
    ("What is the best treatment for a migraine?", "treatment"),
    ("How can I treat a common cold?", "treatment"),
    ("What medications help with arthritis?", "treatment"),
    ("What causes type 2 diabetes?", "cause"),
    ("Why do people get heart disease?", "cause"),
    ("What causes migraines?", "cause"),
    ("How is cancer diagnosed?", "diagnosis"),
    ("What tests are needed to diagnose thyroid disease?", "diagnosis"),
    ("How do doctors test for COVID-19?", "diagnosis"),
    ("Is cystic fibrosis inherited?", "genetics"),
    ("Can hemophilia be genetic?", "genetics"),
    ("Are autism spectrum disorders hereditary?", "genetics"),
    ("Who is at risk for stroke?", "risk"),
    ("What populations are susceptible to malaria?", "risk"),
    ("Who is more likely to get Alzheimer's?", "risk"),
    ("Tell me about health.", "general"),
    ("What do you know about medicine?", "general"),
    ("Information about hospitals?", "general"),
]

# Use standard classification datasets (now with semantic labels from step1)
df_train = pd.read_csv(train_file)

# Choose validation set
if args.validate_on_custom:
    print("Using custom questions as validation set")
    # Create dataframe from custom questions
    df_val = pd.DataFrame(CUSTOM_QUESTIONS, columns=['text', 'label'])
    # Use standard test set
    df_test = pd.read_csv('data/processed/classification_test.csv')
else:
    # Use standard validation/test split
    df_val = pd.read_csv('data/processed/classification_val.csv')
    df_test = pd.read_csv('data/processed/classification_test.csv')

label_to_id = {label: idx for idx, label in enumerate(sorted(set(
    list(df_train['label'].unique()) + list(df_val['label'].unique()) + list(df_test['label'].unique())
)))}
id_to_label = {v: k for k, v in label_to_id.items()}

# Add label IDs
df_train['label_id'] = df_train['label'].map(label_to_id)
df_val['label_id'] = df_val['label'].map(label_to_id)
df_test['label_id'] = df_test['label'].map(label_to_id)

print(f"Label mapping: {label_to_id}")
print(f"\nLabel distribution:")
for idx, lbl in id_to_label.items():
    count = (df_train['label_id'] == idx).sum()
    print(f"  {lbl:15s}: {count:4d} ({count/len(df_train)*100:5.1f}%)")

# Tokenize training data
print(f"\nTokenizing {len(df_train)} training texts...")
tokens_list = []
for text in df_train['text'].values:
    tokens = tokenizer.encode(text, add_cls=True, pad=True)
    tokens_list.append(tokens)

tokens_array = np.array(tokens_list, dtype=np.int64)
labels_array = df_train['label_id'].values

# Tokenize validation and test data
print(f"Tokenizing {len(df_val)} validation texts...")
val_tokens_list = [np.array(tokenizer.encode(text, add_cls=True, pad=True), dtype=np.int64) 
                   for text in df_val['text'].values]
val_tokens_array = np.array(val_tokens_list, dtype=np.int64)
val_labels_array = df_val['label_id'].values

print(f"Tokenizing {len(df_test)} test texts...")
test_tokens_list = [np.array(tokenizer.encode(text, add_cls=True, pad=True), dtype=np.int64) 
                    for text in df_test['text'].values]
test_tokens_array = np.array(test_tokens_list, dtype=np.int64)
test_labels_array = df_test['label_id'].values

print(f"\nTokens shapes:")
print(f"  Train: {tokens_array.shape}")
print(f"  Val:   {val_tokens_array.shape}")
print(f"  Test:  {test_tokens_array.shape}")

# Create tensors directly
train_x = torch.from_numpy(tokens_array).long()
train_y = torch.from_numpy(labels_array).long()

val_x = torch.from_numpy(val_tokens_array).long()
val_y = torch.from_numpy(val_labels_array).long()

test_x = torch.from_numpy(test_tokens_array).long()
test_y = torch.from_numpy(test_labels_array).long()

print(f"\nData split (already split - using pre-made files):")
print(f"  Train: {len(train_x)}")
print(f"  Val:   {len(val_x)}")
print(f"  Test:  {len(test_x)}")

train_ds = TensorDataset(train_x, train_y)
val_ds = TensorDataset(val_x, val_y)
test_ds = TensorDataset(test_x, test_y)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

elapsed = time.time() - start
print(f"✓ Preprocessing done in {elapsed:.2f}s")

# ==================== TRAIN ====================

print("\n" + "="*70)
print("TRAINING")
print("="*70)

model = MiniBERTMedQ(vocab_size, D_MODEL, NUM_HEADS, FFN_DIM, NUM_CLASSES, DROPOUT).to(DEVICE)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

best_val_f1 = 0
patience = 0
history = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'train_acc': [], 'val_acc': []}

for epoch in range(EPOCHS):
    start = time.time()
    
    # Train
    model.train()
    train_loss = 0
    train_preds, train_labels = [], []
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        mask = generate_padding_mask(x_batch).to(DEVICE)
        
        logits = model(x_batch, mask)
        loss = criterion(logits, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item()
        pred = logits.argmax(dim=1)
        train_preds.extend(pred.cpu().numpy())
        train_labels.extend(y_batch.cpu().numpy())
    
    train_loss /= len(train_loader)
    train_acc = accuracy_score(train_labels, train_preds)
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    
    # Val
    model.eval()
    val_loss = 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            mask = generate_padding_mask(x_batch).to(DEVICE)
            
            logits = model(x_batch, mask)
            loss = criterion(logits, y_batch)
            val_loss += loss.item()
            
            pred = logits.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    val_loss /= len(val_loader)
    val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    val_acc = accuracy_score(all_labels, all_preds)
    
    history['val_loss'].append(val_loss)
    history['val_f1'].append(val_f1)
    history['val_acc'].append(val_acc)
    
    elapsed = time.time() - start
    print(f"Epoch {epoch+1:2d}/{EPOCHS}  Tr Loss: {train_loss:.4f}  Tr Acc: {train_acc:.4f}  Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}  F1: {val_f1:.4f}  ({elapsed:.1f}s)", end="")
    
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        patience = 0
        torch.save(model.state_dict(), os.path.join(MODEL_REPO, 'miniBERT_medQ_best.pt'))
        print(" ✓")
    else:
        patience += 1
        print(f" ({patience}/{EARLY_STOP_PATIENCE})")
        if patience >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping")
            break

# ==================== TEST ====================

print("\n" + "="*70)
print("TEST EVALUATION")
print("="*70)

model.load_state_dict(torch.load(os.path.join(MODEL_REPO, 'miniBERT_medQ_best.pt')))
model.eval()

test_preds, test_labels_list = [], []
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        mask = generate_padding_mask(x_batch).to(DEVICE)
        logits = model(x_batch, mask)
        pred = logits.argmax(dim=1)
        test_preds.extend(pred.cpu().numpy())
        test_labels_list.extend(y_batch.cpu().numpy())

test_preds_names = [id_to_label[p] for p in test_preds]
test_labels_names = [id_to_label[l] for l in test_labels_list]

print("\nClassification Report:")
print(classification_report(test_labels_names, test_preds_names, zero_division=0))

test_f1 = f1_score(test_labels_names, test_preds_names, average='macro', zero_division=0)
test_acc = accuracy_score(test_labels_names, test_preds_names)
print(f"\nTest Macro F1: {test_f1:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# Print training metrics summary
print("\n" + "="*70)
print("TRAINING METRICS SUMMARY")
print("="*70)
print(f"\nFinal Training Accuracy: {history['train_acc'][-1]:.4f}")
print(f"Final Validation Accuracy: {history['val_acc'][-1]:.4f}")
print(f"Best Validation F1: {best_val_f1:.4f}")
print(f"\nTraining Loss: {history['train_loss'][0]:.4f} → {history['train_loss'][-1]:.4f}")
print(f"Validation Loss: {history['val_loss'][0]:.4f} → {history['val_loss'][-1]:.4f}")

# Save
results = {
    'history': history,
    'test_f1': test_f1,
    'test_accuracy': test_acc,
    'confusion_matrix': confusion_matrix(test_labels_list, test_preds).tolist(),
    'id_to_label': id_to_label,
    'num_classes': NUM_CLASSES,
    'note': 'Semantic labels (8 classes): cause, definition, diagnosis, general, genetics, risk, symptom, treatment (merged rare classes)'
}
with open(os.path.join(MODEL_REPO, 'results.json'), 'w') as f:
    json.dump(results, f, indent=2)

torch.save(model.state_dict(), os.path.join(MODEL_REPO, 'miniBERT_medQ_best.pt'))
print(f"\n✓ Model saved to {os.path.join(MODEL_REPO, 'miniBERT_medQ_best.pt')}")
print(f"✓ Results saved to {os.path.join(MODEL_REPO, 'results.json')}")

# ==================== PREDICT ====================

print("\n" + "="*70)
print("SAMPLE PREDICTIONS")
print("="*70)

test_samples = [
    "What are early symptoms of diabetes?",
    "How do you treat a broken arm?",
    "What causes high blood pressure?",
]

model.eval()
with torch.no_grad():
    for text in test_samples:
        tokens = tokenizer.encode(text, add_cls=True, pad=True)
        x = torch.tensor([tokens], dtype=torch.long).to(DEVICE)
        mask = generate_padding_mask(x).to(DEVICE)
        logits = model(x, mask)
        probs = torch.softmax(logits, dim=1)
        pred_id = probs.argmax(dim=1).item()
        confidence = probs.squeeze().max().item()
        print(f"  Q: {text}")
        print(f"  A: {id_to_label[pred_id]} (conf: {confidence:.3f})\n")

print("✓ Done!")
