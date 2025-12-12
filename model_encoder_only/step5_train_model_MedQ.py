"""
MiniBERT-MedQ Final: Optimized training with pre-tokenized data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

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
import matplotlib.pyplot as plt
import seaborn as sns

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
history = {
    'train_loss': [], 'val_loss': [],
    'val_f1': [], 'train_acc': [], 'val_acc': [],
    'train_precision': [], 'train_recall': [], 'train_f1': [],
    'val_precision': [], 'val_recall': [],
    'per_class_f1': []  # Will store per-epoch per-class F1 scores
}

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
    train_precision, train_recall, train_f1_per_class, _ = precision_recall_fscore_support(
        train_labels, train_preds, average=None, zero_division=0
    )
    train_precision_macro = train_precision.mean()
    train_recall_macro = train_recall.mean()
    train_f1_macro = f1_score(train_labels, train_preds, average='macro', zero_division=0)
    
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['train_precision'].append(train_precision_macro)
    history['train_recall'].append(train_recall_macro)
    history['train_f1'].append(train_f1_macro)
    
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
    val_precision, val_recall, val_f1_per_class, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    val_precision_macro = val_precision.mean()
    val_recall_macro = val_recall.mean()
    
    # Store per-class F1 scores
    per_class_f1_dict = {id_to_label[i]: float(val_f1_per_class[i]) for i in range(len(val_f1_per_class))}
    history['per_class_f1'].append(per_class_f1_dict)
    
    history['val_loss'].append(val_loss)
    history['val_f1'].append(val_f1)
    history['val_acc'].append(val_acc)
    history['val_precision'].append(val_precision_macro)
    history['val_recall'].append(val_recall_macro)
    
    elapsed = time.time() - start
    print(f"Epoch {epoch+1:2d}/{EPOCHS}")
    print(f"  Loss - Train: {train_loss:.4f}  Val: {val_loss:.4f}")
    print(f"  Acc - Train: {train_acc:.4f}  Val: {val_acc:.4f}")
    print(f"  F1 - Train: {train_f1_macro:.4f}  Val: {val_f1:.4f}")
    print(f"  Precision - Train: {train_precision_macro:.4f}  Val: {val_precision_macro:.4f}")
    print(f"  Recall - Train: {train_recall_macro:.4f}  Val: {val_recall_macro:.4f}")
    print(f"  Time: {elapsed:.1f}s", end="")
    
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

# Calculate test precision and recall
test_precision, test_recall, test_f1_per_class, _ = precision_recall_fscore_support(
    test_labels_list, test_preds, average=None, zero_division=0
)
test_precision_macro = test_precision.mean()
test_recall_macro = test_recall.mean()

# Print training metrics summary
print("\n" + "="*70)
print("TRAINING METRICS SUMMARY")
print("="*70)
print(f"\nFinal Training Metrics:")
print(f"  Accuracy: {history['train_acc'][-1]:.4f}")
print(f"  F1 Score: {history['train_f1'][-1]:.4f}")
print(f"  Precision: {history['train_precision'][-1]:.4f}")
print(f"  Recall: {history['train_recall'][-1]:.4f}")
print(f"\nFinal Validation Metrics:")
print(f"  Accuracy: {history['val_acc'][-1]:.4f}")
print(f"  F1 Score: {history['val_f1'][-1]:.4f}")
print(f"  Precision: {history['val_precision'][-1]:.4f}")
print(f"  Recall: {history['val_recall'][-1]:.4f}")
print(f"\nTest Metrics:")
print(f"  Accuracy: {test_acc:.4f}")
print(f"  F1 Score: {test_f1:.4f}")
print(f"  Precision: {test_precision_macro:.4f}")
print(f"  Recall: {test_recall_macro:.4f}")
print(f"\nBest Validation F1: {best_val_f1:.4f}")
print(f"\nTraining Loss: {history['train_loss'][0]:.4f} → {history['train_loss'][-1]:.4f}")
print(f"Validation Loss: {history['val_loss'][0]:.4f} → {history['val_loss'][-1]:.4f}")

# ==================== VISUALIZATION ====================

print("\n" + "="*70)
print("CREATING EVALUATION PLOTS")
print("="*70)

# Create comprehensive visualization with multiple subplots
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

epochs_range = range(1, len(history['train_loss']) + 1)

# Plot 1: Loss
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(epochs_range, history['train_loss'], label='Train Loss', marker='o', linewidth=2)
ax1.plot(epochs_range, history['val_loss'], label='Val Loss', marker='s', linewidth=2)
ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('Loss', fontsize=11)
ax1.set_title('Loss', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Accuracy
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(epochs_range, history['train_acc'], label='Train Accuracy', marker='o', linewidth=2, color='green')
ax2.plot(epochs_range, history['val_acc'], label='Val Accuracy', marker='s', linewidth=2, color='lime')
ax2.set_xlabel('Epoch', fontsize=11)
ax2.set_ylabel('Accuracy', fontsize=11)
ax2.set_title('Accuracy', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: F1 Score
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(epochs_range, history['train_f1'], label='Train F1', marker='o', linewidth=2, color='purple')
ax3.plot(epochs_range, history['val_f1'], label='Val F1', marker='s', linewidth=2, color='magenta')
ax3.set_xlabel('Epoch', fontsize=11)
ax3.set_ylabel('F1 Score', fontsize=11)
ax3.set_title('F1 Score (Macro)', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Precision
ax4 = fig.add_subplot(gs[1, 0])
ax4.plot(epochs_range, history['train_precision'], label='Train Precision', marker='o', linewidth=2, color='orange')
ax4.plot(epochs_range, history['val_precision'], label='Val Precision', marker='s', linewidth=2, color='red')
ax4.set_xlabel('Epoch', fontsize=11)
ax4.set_ylabel('Precision', fontsize=11)
ax4.set_title('Precision (Macro)', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Recall
ax5 = fig.add_subplot(gs[1, 1])
ax5.plot(epochs_range, history['train_recall'], label='Train Recall', marker='o', linewidth=2, color='cyan')
ax5.plot(epochs_range, history['val_recall'], label='Val Recall', marker='s', linewidth=2, color='blue')
ax5.set_xlabel('Epoch', fontsize=11)
ax5.set_ylabel('Recall', fontsize=11)
ax5.set_title('Recall (Macro)', fontsize=12, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: Per-class F1 Scores (final epoch)
ax6 = fig.add_subplot(gs[1, 2])
if len(history['per_class_f1']) > 0:
    final_per_class_f1 = history['per_class_f1'][-1]
    classes = list(final_per_class_f1.keys())
    f1_values = list(final_per_class_f1.values())
    bars = ax6.barh(classes, f1_values, color='steelblue')
    ax6.set_xlabel('F1 Score', fontsize=11)
    ax6.set_title('Per-Class F1 Score (Final Epoch)', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='x')
    # Add value labels on bars
    for bar, value in zip(bars, f1_values):
        ax6.text(value + 0.01, bar.get_y() + bar.get_height()/2, f'{value:.3f}', 
                va='center', fontsize=9)

# Plot 7: Confusion Matrix
ax7 = fig.add_subplot(gs[2, :2])
cm = confusion_matrix(test_labels_list, test_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(id_to_label.values()), 
            yticklabels=list(id_to_label.values()), ax=ax7, cbar_kws={'label': 'Count'})
ax7.set_xlabel('Predicted', fontsize=11)
ax7.set_ylabel('Actual', fontsize=11)
ax7.set_title('Confusion Matrix (Test Set)', fontsize=12, fontweight='bold')

# Plot 8: Metrics Summary Bar Chart
ax8 = fig.add_subplot(gs[2, 2])
metrics_names = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
train_metrics = [history['train_acc'][-1], history['train_f1'][-1], 
                history['train_precision'][-1], history['train_recall'][-1]]
val_metrics = [history['val_acc'][-1], history['val_f1'][-1], 
              history['val_precision'][-1], history['val_recall'][-1]]
test_metrics = [test_acc, test_f1, test_precision_macro, test_recall_macro]

x = np.arange(len(metrics_names))
width = 0.25
ax8.bar(x - width, train_metrics, width, label='Train', color='green', alpha=0.8)
ax8.bar(x, val_metrics, width, label='Val', color='orange', alpha=0.8)
ax8.bar(x + width, test_metrics, width, label='Test', color='red', alpha=0.8)
ax8.set_ylabel('Score', fontsize=11)
ax8.set_title('Final Metrics Comparison', fontsize=12, fontweight='bold')
ax8.set_xticks(x)
ax8.set_xticklabels(metrics_names, rotation=45, ha='right')
ax8.legend()
ax8.grid(True, alpha=0.3, axis='y')
ax8.set_ylim(0, 1.0)

fig.suptitle('MiniBERT-MedQ Comprehensive Evaluation Metrics', fontsize=16, fontweight='bold', y=0.995)
plt.savefig(os.path.join(MODEL_REPO, 'comprehensive_metrics.png'), dpi=300, bbox_inches='tight')
print(f"✓ Comprehensive metrics plot saved to {os.path.join(MODEL_REPO, 'comprehensive_metrics.png')}")

# Create separate confusion matrix plot for better visibility
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(id_to_label.values()), 
            yticklabels=list(id_to_label.values()), cbar_kws={'label': 'Count'})
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.title('Confusion Matrix - MiniBERT-MedQ (Test Set)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(MODEL_REPO, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
print(f"✓ Confusion matrix saved to {os.path.join(MODEL_REPO, 'confusion_matrix.png')}")

# Save
results = {
    'history': history,
    'test_f1': test_f1,
    'test_accuracy': test_acc,
    'test_precision': test_precision_macro,
    'test_recall': test_recall_macro,
    'test_per_class_f1': {id_to_label[i]: float(test_f1_per_class[i]) for i in range(len(test_f1_per_class))},
    'confusion_matrix': confusion_matrix(test_labels_list, test_preds).tolist(),
    'id_to_label': id_to_label,
    'num_classes': NUM_CLASSES,
    'note': 'Semantic labels (8 classes): cause, definition, diagnosis, general, genetics, risk, symptom, treatment (merged rare classes)'
}
with open(os.path.join(MODEL_REPO, 'results.json'), 'w') as f:
    json.dump(results, f, indent=2)

torch.save(model.state_dict(), os.path.join(MODEL_REPO, 'miniBERT_medQ_best.pt'))

print("\n" + "="*70)
print("✓ TRAINING COMPLETE")
print("="*70)
print(f"\n✓ Model saved to {os.path.join(MODEL_REPO, 'miniBERT_medQ_best.pt')}")
print(f"✓ Results saved to {os.path.join(MODEL_REPO, 'results.json')}")
print(f"✓ Comprehensive metrics plot: {os.path.join(MODEL_REPO, 'comprehensive_metrics.png')}")
print(f"✓ Confusion matrix: {os.path.join(MODEL_REPO, 'confusion_matrix.png')}")

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
