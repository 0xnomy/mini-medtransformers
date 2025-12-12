# MiniGPT-MedLM: Decoder-Only Transformer for Medical Text Generation

A lightweight GPT-style autoregressive language model trained on medical text data for next-token prediction and text generation.

## Architecture

**Model Specifications:**
- **Type:** Decoder-only transformer (GPT architecture)
- **Dimensions:** 128-dim embeddings, 512-dim FFN
- **Layers:** 2 decoder layers with 4 attention heads each
- **Vocabulary:** 10,006 tokens (BPE tokenizer)
- **Max Sequence Length:** 128 tokens
- **Parameters:** 1,677,568 total

**Key Features:**
- Causal attention masking (no future token visibility)
- Padding mask (ignores pad tokens in loss computation)
- Weight tying (embedding layer = LM head)
- Sinusoidal positional encoding (fixed, not learnable)
- Pre-norm architecture (LayerNorm before attention/FFN)

## Implementation

### Files Structure
```
model_decoder_only/
├── masks.py                  # Causal & padding mask generation
├── decoder_layers.py         # MultiHeadSelfAttention, FeedForward, DecoderLayer
├── gpt_model.py             # Complete MiniGPT_MedLM model with generation
├── utils.py                 # MedLMDataset, data loading, I/O utilities
├── train_gpt_medLM.py       # Training pipeline with loss tracking
├── generate_text.py         # Text generation with 3 sampling strategies
└── tests_decoder_only.py    # Unit tests for all components
```

### Training Configuration
- **Objective:** Next-token prediction (autoregressive language modeling)
- **Loss Function:** CrossEntropyLoss with `ignore_index=0` (pad token)
- **Optimizer:** AdamW (lr=3e-4, weight_decay=0.01)
- **Gradient Clipping:** Max norm 1.0
- **Batch Size:** 32
- **Epochs:** 8
- **Device:** CUDA (GPU accelerated)

### Data Pipeline
1. Load medical texts from `data/processed/lm_train.csv`, `lm_val.csv`, `lm_test.csv`
2. Tokenize using BPE tokenizer (from encoder-only model)
3. Create shifted sequences: `input=[tokens[:-1]]`, `target=[tokens[1:]]`
4. Combine train+val for training (31,824 samples), use test for validation (3,536 samples)

## How to Execute

### 1. Run Unit Tests
```bash
python model_decoder_only/tests_decoder_only.py
```
**Expected Output:** All 13 tests pass (masks, decoder components, model, training)

### 2. Train the Model
```bash
python model_decoder_only/train_gpt_medLM.py
```
**What happens:**
- Loads tokenizer and datasets
- Initializes MiniGPT-MedLM model
- Trains for 8 epochs with validation after each epoch
- Saves best model to `artifacts/medLM_models/miniGPT_medLM.pt`
- Generates training curves and sample text outputs

**Training Time:** ~42 seconds on CUDA GPU

### 3. Generate Text
```bash
python model_decoder_only/generate_text.py
```
**What happens:**
- Loads trained model
- Tests 3 prompts with 3 generation strategies:
  1. **Greedy Decoding** (deterministic, always picks most likely token)
  2. **Top-K Sampling** (samples from k=50 most likely tokens, temp=0.8)
  3. **Nucleus Sampling** (top-p=0.9, samples from cumulative probability, temp=1.0)

## Results Achieved

### Training Metrics
| Epoch | Train Loss | Val Loss | Time |
|-------|-----------|----------|------|
| 1/8   | 5.6044    | 4.1720   | 5.4s |
| 2/8   | 4.0315    | 3.5676   | 5.2s |
| 3/8   | 3.5986    | 3.2781   | 5.2s |
| 4/8   | 3.3461    | 3.0927   | 5.2s |
| 5/8   | 3.1760    | 2.9653   | 5.3s |
| 6/8   | 3.0526    | 2.8697   | 5.3s |
| 7/8   | 2.9560    | 2.7959   | 5.3s |
| 8/8   | 2.8792    | 2.7432   | 5.3s |

**Final Performance:**
- ✅ **49% train loss reduction** (5.60 → 2.88)
- ✅ **34% validation loss reduction** (4.17 → 2.74)
- ✅ Smooth convergence with no overfitting
- ✅ Model learns medical vocabulary and text structure

### Sample Generations

**Prompt:** "Diabetes is a condition"
- Model generates continuations mentioning blood sugar, kidneys, blood vessels, symptoms

**Prompt:** "Treatment options include"
- Model generates medical procedures: surgery, chemotherapy, radiation therapy, stem cell transplant

**Prompt:** "Symptoms may include"
- Model generates clinical symptoms: hearing loss, difficulty swallowing, numbness, muscle weakness

## Model Outputs

**Saved Artifacts:**
- `artifacts/medLM_models/miniGPT_medLM.pt` - Trained model weights
- `artifacts/medLM_models/training_history.json` - Loss curves data
- `artifacts/medLM_models/training_curve.png` - Training visualization

## Technical Details

### Causal Masking
```python
# Triangular mask prevents attending to future tokens
mask[i, j] = 0.0   if j <= i  # Can attend to past/current
mask[i, j] = -inf  if j > i   # Cannot attend to future
```

### Next-Token Prediction
```python
# Shift tokens for autoregressive training
input_ids  = [token_1, token_2, ..., token_n-1]
target_ids = [token_2, token_3, ..., token_n]
# Model predicts target_ids from input_ids
```

### Generation Strategies
1. **Greedy:** `output = argmax(logits)` - most likely token always
2. **Top-K:** Sample from k highest probability tokens with temperature scaling
3. **Nucleus:** Sample from smallest set of tokens with cumulative probability ≥ p

## Comparison: Encoder vs Decoder

| Aspect | MiniBERT-MedQ (Encoder) | MiniGPT-MedLM (Decoder) |
|--------|------------------------|------------------------|
| Task | Classification | Text Generation |
| Attention | Bidirectional | Causal (autoregressive) |
| Masking | Padding only | Causal + Padding |
| Output | Class logits | Next-token logits |
| Training | CrossEntropy on [CLS] | CrossEntropy on all tokens |
| Use Case | Question categorization | Answer generation |


## Notes

- Uses same tokenizer as encoder-only model (vocabulary consistency)
- Pre-norm architecture follows modern GPT implementations
- Weight tying reduces parameters and improves generalization
- Temperature controls randomness: lower = more deterministic, higher = more creative
- Model trained on medical question-answer pairs from MedQuAD dataset
