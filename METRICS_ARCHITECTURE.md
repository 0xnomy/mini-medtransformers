# Evaluation Metrics Architecture

## Model 1: MiniGPT-MedLM (Decoder-Only / Language Model)

```
┌─────────────────────────────────────────────────────────────────┐
│                    MiniGPT-MedLM Training Loop                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │        Per-Epoch Metric Collection       │
        └─────────────────────────────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        │                                            │
        ▼                                            ▼
┌──────────────────┐                     ┌──────────────────┐
│  Training Phase  │                     │ Validation Phase │
└──────────────────┘                     └──────────────────┘
        │                                            │
        ├─► Loss (CrossEntropy)                     ├─► Loss (CrossEntropy)
        ├─► Token Accuracy                          ├─► Token Accuracy
        └─► Perplexity = exp(loss)                  ├─► Perplexity = exp(loss)
                                                     ├─► BLEU Score (every 2 epochs)
                                                     ├─► ROUGE-1, ROUGE-2, ROUGE-L
                                                     └─► BERTScore F1
                              │
                              ▼
        ┌──────────────────────────────────────────┐
        │         Save to training_history.json     │
        │  • train_loss, val_loss                   │
        │  • train_accuracy, val_accuracy           │
        │  • train_perplexity, val_perplexity       │
        │  • bleu, rouge1, rouge2, rougeL           │
        │  • bert_score_f1                          │
        └──────────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────────┐
        │      Generate Visualization Plots         │
        │                                           │
        │  comprehensive_metrics.png (6 panels):    │
        │  ┌────────┬────────┬────────┐            │
        │  │  Loss  │Accuracy│Perplexi│            │
        │  ├────────┼────────┼────────┤            │
        │  │  BLEU  │ ROUGE  │BERTScor│            │
        │  └────────┴────────┴────────┘            │
        │                                           │
        │  training_curve.png (simple loss)         │
        └──────────────────────────────────────────┘
```

## Model 2: MiniBERT-MedQ (Encoder-Only / Classifier)

```
┌─────────────────────────────────────────────────────────────────┐
│                  MiniBERT-MedQ Training Loop                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │        Per-Epoch Metric Collection       │
        └─────────────────────────────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        │                                            │
        ▼                                            ▼
┌──────────────────┐                     ┌──────────────────┐
│  Training Phase  │                     │ Validation Phase │
└──────────────────┘                     └──────────────────┘
        │                                            │
        ├─► Loss (CrossEntropy + Smoothing)         ├─► Loss (CrossEntropy)
        ├─► Accuracy                                ├─► Accuracy
        ├─► F1 Score (Macro)                        ├─► F1 Score (Macro)
        ├─► Precision (Macro)                       ├─► Precision (Macro)
        ├─► Recall (Macro)                          ├─► Recall (Macro)
        └─► Per-Class F1                            └─► Per-Class F1 (8 classes)
                              │
                              ▼
        ┌──────────────────────────────────────────┐
        │            Test Evaluation                │
        │  • Final accuracy, F1, precision, recall  │
        │  • Per-class metrics for all 8 categories │
        │  • Confusion matrix (8x8)                 │
        │  • Classification report                  │
        └──────────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────────┐
        │           Save to results.json            │
        │  • history (all training metrics)         │
        │  • test_f1, test_accuracy                 │
        │  • test_precision, test_recall            │
        │  • test_per_class_f1                      │
        │  • confusion_matrix                       │
        └──────────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────────┐
        │      Generate Visualization Plots         │
        │                                           │
        │  comprehensive_metrics.png (8 panels):    │
        │  ┌────────┬────────┬────────┐            │
        │  │  Loss  │Accuracy│   F1   │            │
        │  ├────────┼────────┼────────┤            │
        │  │Precisio│ Recall │Per-Cls │            │
        │  ├────────┴────────┴────────┤            │
        │  │   Confusion Matrix (8x8) │            │
        │  ├─────────────────────────┤             │
        │  │  Final Metrics Comparison│            │
        │  └─────────────────────────┘             │
        │                                           │
        │  confusion_matrix.png (standalone)        │
        └──────────────────────────────────────────┘
```

## Metrics Calculation Pipeline

### Decoder Model (Language Generation)

```
Input Sequence: "Diabetes is a"
        ↓
  Token Embeddings
        ↓
  Transformer Decoder Layers
        ↓
  Output Logits: [vocab_size]
        ↓
  ┌────────────────────────────┐
  │   Metric Calculations      │
  ├────────────────────────────┤
  │ 1. Loss = CrossEntropy(    │
  │      logits, targets)      │
  │                            │
  │ 2. Accuracy =              │
  │      correct_tokens /      │
  │      total_non_pad_tokens  │
  │                            │
  │ 3. Perplexity = exp(loss)  │
  │                            │
  │ 4. Generate Text:          │
  │    "Diabetes is a chronic  │
  │     metabolic disorder"    │
  │                            │
  │ 5. Compare with Reference: │
  │    - BLEU (n-grams)        │
  │    - ROUGE (recall)        │
  │    - BERTScore (semantic)  │
  └────────────────────────────┘
```

### Encoder Model (Text Classification)

```
Input Text: "What causes diabetes?"
        ↓
  [CLS] + Token Embeddings
        ↓
  Transformer Encoder Layers
        ↓
  [CLS] Representation
        ↓
  Classification Head
        ↓
  Class Probabilities: [8 classes]
        ↓
  Predicted: "cause" (85% confidence)
  True Label: "cause"
        ↓
  ┌────────────────────────────┐
  │   Metric Calculations      │
  ├────────────────────────────┤
  │ 1. Loss = CrossEntropy(    │
  │      logits, true_class)   │
  │                            │
  │ 2. Accuracy = 1            │
  │      (correct prediction)  │
  │                            │
  │ 3. For "cause" class:      │
  │    TP = 1, FP = 0, FN = 0  │
  │                            │
  │ 4. Precision = TP/(TP+FP)  │
  │    Recall = TP/(TP+FN)     │
  │    F1 = 2*P*R/(P+R)        │
  │                            │
  │ 5. Aggregate across all    │
  │    8 classes (macro avg)   │
  └────────────────────────────┘
```

## Metrics Formulas

### Universal Metrics

```
Loss (Cross-Entropy):
  L = -Σ y_true * log(y_pred)

Accuracy:
  Acc = (Correct Predictions) / (Total Predictions)
```

### Generation Metrics (Decoder Only)

```
Perplexity:
  PPL = exp(Loss)
  • Range: [1, ∞)
  • Lower is better

BLEU:
  BLEU = BP * exp(Σ w_n * log(p_n))
  • BP: Brevity penalty
  • p_n: n-gram precision
  • Range: [0, 1] or [0, 100]

ROUGE-L:
  ROUGE-L = LCS(reference, hypothesis) / len(reference)
  • LCS: Longest Common Subsequence
  • Range: [0, 1] or [0, 100]

BERTScore:
  F1 = 2 * (Precision * Recall) / (Precision + Recall)
  where scores based on BERT embeddings
  • Range: [0, 1] or [0, 100]
```

### Classification Metrics (Encoder Only)

```
Precision:
  P = TP / (TP + FP)
  • Of predicted positives, how many correct?

Recall:
  R = TP / (TP + FN)
  • Of actual positives, how many found?

F1 Score:
  F1 = 2 * (P * R) / (P + R)
  • Harmonic mean of precision and recall

Macro-Average:
  Metric_macro = (Σ Metric_class_i) / num_classes
  • Treats all classes equally
```

## Data Flow During Training

```
┌──────────────┐
│ Load Dataset │
└──────┬───────┘
       │
       ▼
┌──────────────────┐      ┌────────────────┐
│ Tokenize Texts   │─────▶│ Create Batches │
└──────────────────┘      └───────┬────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │   Training Loop         │
                    │   (Multiple Epochs)     │
                    └────────┬────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
   ┌─────────┐         ┌─────────┐         ┌─────────┐
   │ Forward │         │Backward │         │ Update  │
   │  Pass   │────────▶│  Pass   │────────▶│ Weights │
   └────┬────┘         └─────────┘         └─────────┘
        │
        ▼
   ┌──────────────────┐
   │ Calculate Metrics│
   │  • Loss          │
   │  • Accuracy      │
   │  • F1/Perplexity │
   │  • etc.          │
   └────┬─────────────┘
        │
        ▼
   ┌──────────────────┐
   │ Store in History │
   └────┬─────────────┘
        │
        ▼
   ┌──────────────────┐
   │ Print Progress   │
   └────┬─────────────┘
        │
        ▼ (After all epochs)
   ┌──────────────────┐
   │ Save Results     │
   │ • Model weights  │
   │ • History JSON   │
   │ • Plots (PNG)    │
   └──────────────────┘
```

## File Structure After Training

```
project/
├── artifacts/
│   ├── medLM_models/                  (Decoder Model)
│   │   ├── miniGPT_medLM.pt          [Model checkpoint]
│   │   ├── training_history.json     [All metrics]
│   │   ├── comprehensive_metrics.png [6-panel plot]
│   │   └── training_curve.png        [Loss curve]
│   │
│   └── medQ_models/                   (Encoder Model)
│       ├── miniBERT_medQ_best.pt     [Model checkpoint]
│       ├── results.json               [All metrics + test results]
│       ├── comprehensive_metrics.png  [8-panel plot]
│       └── confusion_matrix.png       [Confusion matrix]
│
├── model_decoder_only/
│   └── train_gpt_medLM.py            [Updated with metrics]
│
├── model_encoder_only/
│   └── step5_train_model_MedQ.py     [Updated with metrics]
│
├── EVALUATION_METRICS_SUMMARY.md      [Detailed documentation]
└── METRICS_QUICK_REFERENCE.txt        [Quick lookup guide]
```

## Summary

**Total Metrics Implemented: 13 Unique Metrics**

**Decoder Model (6 primary metrics):**
1. Loss
2. Token Accuracy
3. Perplexity
4. BLEU
5. ROUGE (3 variants: 1, 2, L)
6. BERTScore

**Encoder Model (7 primary metrics):**
1. Loss
2. Accuracy
3. F1 Score (Macro + Per-Class)
4. Precision (Macro)
5. Recall (Macro)
6. Confusion Matrix
7. Per-Class Metrics

**All metrics are:**
- ✅ Tracked per epoch
- ✅ Saved to JSON
- ✅ Visualized in plots
- ✅ Displayed during training
- ✅ Computed for train/val/test splits (where applicable)
