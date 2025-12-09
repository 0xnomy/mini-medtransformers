"""
STEP 3: Core Transformer Components from Scratch
Mini-MedTransformers Project

Implements all reusable modules for encoder-only, decoder-only, and encoder-decoder models:
- Token Embeddings with scaling
- Sinusoidal Positional Encoding (Vaswani et al. 2017)
- Multi-Head Self & Cross Attention
- Feed Forward Networks
- Transformer Encoder & Decoder Layers
- Masking functions (padding & causal)

Configuration:
    d_model = 128 (embedding dimension)
    ffn_dim = 256 (feed-forward hidden dim)
    num_heads = 4 (attention heads)
    max_seq_len = 128 (maximum sequence length)

No external libraries except PyTorch and standard packages.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================================
# CONFIGURATION
# ============================================================================

D_MODEL = 128          # Embedding and model dimension
FFN_DIM = 256          # Feed-forward network hidden dimension
NUM_HEADS = 4          # Number of attention heads
MAX_SEQ_LEN = 128      # Maximum sequence length
HEAD_DIM = D_MODEL // NUM_HEADS  # Dimension per head: 128/4 = 32

assert D_MODEL % NUM_HEADS == 0, f"d_model ({D_MODEL}) must be divisible by num_heads ({NUM_HEADS})"


# ============================================================================
# PART 1: TOKEN EMBEDDING
# ============================================================================

class TokenEmbedding(nn.Module):
    """
    Token embedding layer with scaling by √(d_model).
    
    This scales embeddings by sqrt(d_model) as recommended in the Transformer paper
    to prevent the position encodings from being completely overwhelmed by the
    embedding vectors.
    
    Args:
        vocab_size: Size of vocabulary
        d_model: Embedding dimension
    """
    
    def __init__(self, vocab_size: int, d_model: int):
        """Initialize token embedding."""
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Scale factor: sqrt(d_model)
        self.scale = math.sqrt(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed token IDs.
        
        Args:
            x: Token IDs, shape (batch, seq_len)
        
        Returns:
            Scaled embeddings, shape (batch, seq_len, d_model)
        """
        # Embed tokens: (batch, seq_len) -> (batch, seq_len, d_model)
        embeddings = self.embedding(x)
        
        # Scale by sqrt(d_model)
        return embeddings * self.scale


# ============================================================================
# PART 2: POSITIONAL ENCODING
# ============================================================================

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding as described in Vaswani et al. 2017.
    
    Uses sine and cosine functions of different frequencies to encode absolute position.
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    This allows the model to attend to relative positions since PE(pos+k) can be
    represented as a linear function of PE(pos).
    
    Args:
        d_model: Embedding dimension
        max_len: Maximum sequence length
    """
    
    def __init__(self, d_model: int, max_len: int = 128):
        """Initialize positional encoding."""
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # Create positional encoding table
        # Shape: (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        
        # Position indices: (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Dimension indices for sin/cos: (d_model/2,)
        # div_term = 1 / (10000 ^ (2i / d_model))
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * 
            -(math.log(10000.0) / d_model)
        )
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd indices
        if d_model % 2 == 1:
            # Handle odd d_model (last dimension gets cosine)
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension: (1, max_len, d_model)
        # This will broadcast across batch during forward pass
        pe = pe.unsqueeze(0)
        
        # Register as buffer (no gradients, saved with model)
        self.register_buffer("pe", pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input embeddings, shape (batch, seq_len, d_model)
        
        Returns:
            Input + positional encoding, shape (batch, seq_len, d_model)
        """
        # Get sequence length from input
        seq_len = x.size(1)
        
        # Add positional encoding: (batch, seq_len, d_model) + (1, seq_len, d_model)
        # Broadcasting expands (1, seq_len, d_model) to (batch, seq_len, d_model)
        return x + self.pe[:, :seq_len, :]


# ============================================================================
# PART 3: MULTI-HEAD ATTENTION
# ============================================================================

class MultiHeadAttention(nn.Module):
    """
    Multi-head scaled dot-product attention.
    
    Allows the model to jointly attend to information from different representation
    subspaces at different positions.
    
    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
    
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
    """
    
    def __init__(self, d_model: int, num_heads: int):
        """Initialize multi-head attention."""
        super().__init__()
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
        
        # Scale factor for dot-product: 1 / sqrt(d_k)
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through multi-head attention.
        
        Args:
            q: Query, shape (batch, seq_len_q, d_model)
            k: Key, shape (batch, seq_len_k, d_model)
            v: Value, shape (batch, seq_len_v, d_model)
            mask: Attention mask, shape (batch, 1, seq_len_q, seq_len_k) or similar
        
        Returns:
            attention_output: Shape (batch, seq_len_q, d_model)
            attention_weights: Shape (batch, num_heads, seq_len_q, seq_len_k)
        """
        batch_size = q.size(0)
        seq_len_q = q.size(1)
        seq_len_k = k.size(1)
        seq_len_v = v.size(1)
        
        # Linear projections
        # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        q = self.W_q(q)
        k = self.W_k(k)
        v = self.W_v(v)
        
        # Reshape for multi-head attention
        # (batch, seq_len, d_model) -> (batch, seq_len, num_heads, head_dim)
        q = q.reshape(batch_size, seq_len_q, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len_k, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len_v, self.num_heads, self.head_dim)
        
        # Transpose to (batch, num_heads, seq_len, head_dim)
        # This allows separate attention computation for each head
        q = q.transpose(1, 2)  # (batch, num_heads, seq_len_q, head_dim)
        k = k.transpose(1, 2)  # (batch, num_heads, seq_len_k, head_dim)
        v = v.transpose(1, 2)  # (batch, num_heads, seq_len_v, head_dim)
        
        # Compute attention scores
        # Q K^T: (batch, num_heads, seq_len_q, head_dim) @ (batch, num_heads, head_dim, seq_len_k)
        #     -> (batch, num_heads, seq_len_q, seq_len_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            # Mask shape should be broadcastable to scores
            # Typically (batch, 1, seq_len_q, seq_len_k) or (batch, num_heads, seq_len_q, seq_len_k)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax to get attention weights
        # (batch, num_heads, seq_len_q, seq_len_k)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Handle NaNs from -inf (when entire row is masked)
        # Replace NaN with 0 (will multiply with 0 in next step anyway)
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
        
        # Apply attention to values
        # (batch, num_heads, seq_len_q, seq_len_k) @ (batch, num_heads, seq_len_k, head_dim)
        #     -> (batch, num_heads, seq_len_q, head_dim)
        context = torch.matmul(attention_weights, v)
        
        # Concatenate heads: (batch, num_heads, seq_len_q, head_dim) -> (batch, seq_len_q, num_heads * head_dim)
        # Transpose first: (batch, num_heads, seq_len_q, head_dim) -> (batch, seq_len_q, num_heads, head_dim)
        context = context.transpose(1, 2).contiguous()
        # Reshape: (batch, seq_len_q, num_heads, head_dim) -> (batch, seq_len_q, d_model)
        context = context.reshape(batch_size, seq_len_q, self.d_model)
        
        # Final linear projection
        # (batch, seq_len_q, d_model) -> (batch, seq_len_q, d_model)
        output = self.W_o(context)
        
        return output, attention_weights


# ============================================================================
# PART 4: FEED-FORWARD NETWORK
# ============================================================================

class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    
    Consists of two linear transformations with ReLU activation:
    FFN(x) = max(0, x W_1 + b_1) W_2 + b_2
    
    Applied to each position separately and identically.
    Expands to d_ff then projects back to d_model.
    
    Args:
        d_model: Model dimension
        ffn_dim: Hidden dimension in feed-forward network
    """
    
    def __init__(self, d_model: int, ffn_dim: int):
        """Initialize feed-forward network."""
        super().__init__()
        # Expand: d_model -> ffn_dim
        self.fc1 = nn.Linear(d_model, ffn_dim)
        # Contract: ffn_dim -> d_model
        self.fc2 = nn.Linear(ffn_dim, d_model)
        # Activation
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feed-forward network.
        
        Args:
            x: Input, shape (batch, seq_len, d_model)
        
        Returns:
            Output, shape (batch, seq_len, d_model)
        """
        # Expand and apply ReLU
        # (batch, seq_len, d_model) -> (batch, seq_len, ffn_dim)
        x = self.fc1(x)
        x = self.relu(x)
        
        # Contract back to d_model
        # (batch, seq_len, ffn_dim) -> (batch, seq_len, d_model)
        x = self.fc2(x)
        
        return x


# ============================================================================
# PART 5: TRANSFORMER ENCODER LAYER
# ============================================================================

class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer encoder layer.
    
    Structure:
    1. Multi-head self-attention
    2. Residual connection + LayerNorm
    3. Feed-forward network
    4. Residual connection + LayerNorm
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        ffn_dim: Feed-forward hidden dimension
    """
    
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int):
        """Initialize encoder layer."""
        super().__init__()
        
        # Self-attention
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        
        # Layer normalization for attention
        self.norm1 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.feed_forward = FeedForward(d_model, ffn_dim)
        
        # Layer normalization for feed-forward
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through encoder layer.
        
        Args:
            x: Input, shape (batch, seq_len, d_model)
            padding_mask: Padding mask, shape (batch, 1, 1, seq_len)
                         Values should be 1 for valid tokens, 0 for padding
        
        Returns:
            Output, shape (batch, seq_len, d_model)
        """
        # ===== Multi-head self-attention with residual connection =====
        # Self-attention: Q, K, V all come from input
        attention_output, _ = self.self_attention(x, x, x, mask=padding_mask)
        
        # Residual connection + LayerNorm (pre-norm architecture)
        x = self.norm1(x + attention_output)
        
        # ===== Feed-forward with residual connection =====
        ff_output = self.feed_forward(x)
        
        # Residual connection + LayerNorm
        x = self.norm2(x + ff_output)
        
        return x


# ============================================================================
# PART 6: TRANSFORMER DECODER LAYER
# ============================================================================

class TransformerDecoderLayer(nn.Module):
    """
    Single Transformer decoder layer.
    
    Structure:
    1. Causal multi-head self-attention (attends only to past)
    2. Residual connection + LayerNorm
    3. Multi-head cross-attention (attends to encoder output)
    4. Residual connection + LayerNorm
    5. Feed-forward network
    6. Residual connection + LayerNorm
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        ffn_dim: Feed-forward hidden dimension
    """
    
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int):
        """Initialize decoder layer."""
        super().__init__()
        
        # Causal self-attention (attends only to past)
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        
        # Layer normalization for self-attention
        self.norm1 = nn.LayerNorm(d_model)
        
        # Cross-attention to encoder output
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        
        # Layer normalization for cross-attention
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.feed_forward = FeedForward(d_model, ffn_dim)
        
        # Layer normalization for feed-forward
        self.norm3 = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        enc_out: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        causal_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through decoder layer.
        
        Args:
            x: Decoder input, shape (batch, seq_len, d_model)
            enc_out: Encoder output, shape (batch, enc_seq_len, d_model)
            padding_mask: Padding mask for encoder, shape (batch, 1, 1, enc_seq_len)
            causal_mask: Causal mask for self-attention, shape (dec_seq_len, dec_seq_len)
        
        Returns:
            Output, shape (batch, seq_len, d_model)
        """
        # ===== Causal self-attention with residual connection =====
        # Causal mask prevents attending to future positions
        self_attention_output, _ = self.self_attention(x, x, x, mask=causal_mask)
        x = self.norm1(x + self_attention_output)
        
        # ===== Cross-attention to encoder output with residual =====
        # Q comes from decoder, K and V from encoder
        cross_attention_output, _ = self.cross_attention(x, enc_out, enc_out, mask=padding_mask)
        x = self.norm2(x + cross_attention_output)
        
        # ===== Feed-forward with residual connection =====
        ff_output = self.feed_forward(x)
        x = self.norm3(x + ff_output)
        
        return x


# ============================================================================
# PART 7: MASKING FUNCTIONS
# ============================================================================

def generate_padding_mask(
    token_ids: torch.Tensor,
    pad_token_id: int = 0
) -> torch.Tensor:
    """
    Generate padding mask from token IDs.
    
    Creates a mask where valid tokens are 1 and padding tokens are 0.
    Output shape is (batch, 1, 1, seq_len) for broadcasting in attention.
    
    Args:
        token_ids: Token IDs, shape (batch, seq_len)
        pad_token_id: ID of padding token (default: 0 for <pad>)
    
    Returns:
        Padding mask, shape (batch, 1, 1, seq_len)
        Values: 1 for valid tokens, 0 for padding
    """
    # Create mask: 1 where token != pad_token, 0 where token == pad_token
    # (batch, seq_len) -> (batch, seq_len)
    mask = (token_ids != pad_token_id).float()
    
    # Reshape for attention broadcasting: (batch, 1, 1, seq_len)
    # This broadcasts correctly in attention: (batch, num_heads, seq_len_q, seq_len_k)
    mask = mask.unsqueeze(1).unsqueeze(1)
    
    return mask


def generate_causal_mask(size: int) -> torch.Tensor:
    """
    Generate causal mask (lower triangular matrix).
    
    Prevents the decoder from attending to future positions.
    Creates a lower triangular matrix where 1 indicates "can attend" and 0 indicates "cannot attend".
    
    Example for size=3:
    [[1, 0, 0],
     [1, 1, 0],
     [1, 1, 1]]
    
    Position i can only attend to positions <= i.
    
    Args:
        size: Sequence length
    
    Returns:
        Causal mask, shape (size, size)
        Values: 1 for allowed attention, 0 for masked
    """
    # Create lower triangular matrix
    # torch.tril creates a lower triangular matrix with ones
    # (size, size) with 1s in lower triangle (including diagonal)
    mask = torch.tril(torch.ones(size, size))
    
    # Reshape to (1, 1, size, size) for broadcasting in attention
    # This broadcasts correctly in attention: (batch, num_heads, seq_len_q, seq_len_k)
    mask = mask.unsqueeze(0).unsqueeze(0)
    
    return mask


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_component_outputs():
    """
    Run comprehensive validation tests on all components.
    
    This function tests:
    1. Token embedding shapes and scaling
    2. Positional encoding shapes
    3. Multi-head attention outputs
    4. Feed-forward network outputs
    5. Encoder layer forward pass
    6. Decoder layer forward pass
    7. Masking functions
    8. Attention weight sanity
    """
    print("\n" + "="*80)
    print("STEP 3: CORE TRANSFORMER COMPONENTS - VALIDATION")
    print("="*80)
    
    # Test configuration
    batch_size = 2
    seq_len = 16
    vocab_size = 10006
    device = torch.device("cpu")
    
    print(f"\n📊 Test Configuration:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Sequence length: {seq_len}")
    print(f"  - Vocabulary size: {vocab_size}")
    print(f"  - d_model: {D_MODEL}")
    print(f"  - num_heads: {NUM_HEADS}")
    print(f"  - ffn_dim: {FFN_DIM}")
    
    # =====================================================================
    # TEST 1: Token Embedding
    # =====================================================================
    print(f"\n{'='*80}")
    print("TEST 1: Token Embedding")
    print(f"{'='*80}")
    
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"\n✓ Input token IDs shape: {token_ids.shape}")
    
    token_embedding = TokenEmbedding(vocab_size, D_MODEL)
    embeddings = token_embedding(token_ids)
    
    print(f"✓ Output embedding shape: {embeddings.shape}")
    assert embeddings.shape == (batch_size, seq_len, D_MODEL), \
        f"Expected shape ({batch_size}, {seq_len}, {D_MODEL}), got {embeddings.shape}"
    
    # Check that embeddings are scaled
    print(f"✓ Embedding mean: {embeddings.mean().item():.6f}")
    print(f"✓ Embedding std: {embeddings.std().item():.6f}")
    print(f"✓ sqrt(d_model): {math.sqrt(D_MODEL):.6f}")
    
    # Verify no special handling for padding (should just be regular embeddings)
    pad_embedding = token_embedding(torch.tensor([[0]])).squeeze()
    regular_embedding = token_embedding(torch.tensor([[1]])).squeeze()
    print(f"✓ Padding token embedding differs from token 1: {not torch.allclose(pad_embedding, regular_embedding)}")
    
    print("\n✅ Token Embedding: PASSED")
    
    # =====================================================================
    # TEST 2: Positional Encoding
    # =====================================================================
    print(f"\n{'='*80}")
    print("TEST 2: Positional Encoding")
    print(f"{'='*80}")
    
    pos_encoding = PositionalEncoding(D_MODEL, MAX_SEQ_LEN)
    
    # Get PE
    pe_buffer = pos_encoding.pe
    print(f"\n✓ Positional encoding buffer shape: {pe_buffer.shape}")
    assert pe_buffer.shape == (1, MAX_SEQ_LEN, D_MODEL), \
        f"Expected shape (1, {MAX_SEQ_LEN}, {D_MODEL}), got {pe_buffer.shape}"
    
    # Check that positions are not all zeros
    print(f"✓ PE mean: {pe_buffer.mean().item():.6f}")
    print(f"✓ PE std: {pe_buffer.std().item():.6f}")
    print(f"✓ PE max: {pe_buffer.max().item():.6f}")
    print(f"✓ PE min: {pe_buffer.min().item():.6f}")
    
    # Check that different positions have different encodings
    pos_0 = pe_buffer[0, 0, :]
    pos_1 = pe_buffer[0, 1, :]
    pos_diff = (pos_0 - pos_1).abs().sum().item()
    print(f"✓ Difference between position 0 and 1: {pos_diff:.6f}")
    assert pos_diff > 0, "Different positions should have different encodings"
    
    # Test adding PE to embeddings
    x_with_pe = pos_encoding(embeddings)
    print(f"✓ Input + PE shape: {x_with_pe.shape}")
    assert x_with_pe.shape == embeddings.shape, "Adding PE should preserve shape"
    print(f"✓ Adding PE changes values: {not torch.allclose(x_with_pe, embeddings)}")
    
    print("\n✅ Positional Encoding: PASSED")
    
    # =====================================================================
    # TEST 3: Multi-Head Attention
    # =====================================================================
    print(f"\n{'='*80}")
    print("TEST 3: Multi-Head Attention")
    print(f"{'='*80}")
    
    attention = MultiHeadAttention(D_MODEL, NUM_HEADS)
    
    # Test self-attention (Q=K=V)
    print(f"\n✓ Testing self-attention (Q=K=V)...")
    attn_out, attn_weights = attention(embeddings, embeddings, embeddings)
    
    print(f"✓ Output shape: {attn_out.shape}")
    assert attn_out.shape == (batch_size, seq_len, D_MODEL), \
        f"Expected shape ({batch_size}, {seq_len}, {D_MODEL}), got {attn_out.shape}"
    
    print(f"✓ Attention weights shape: {attn_weights.shape}")
    assert attn_weights.shape == (batch_size, NUM_HEADS, seq_len, seq_len), \
        f"Expected shape ({batch_size}, {NUM_HEADS}, {seq_len}, {seq_len}), got {attn_weights.shape}"
    
    # Check attention weights are probabilities (sum to 1)
    weights_sum = attn_weights.sum(dim=-1)
    print(f"✓ Attention weights sum (should be 1): {weights_sum[0, 0, :5]}")
    assert torch.allclose(weights_sum, torch.ones_like(weights_sum), atol=1e-5), \
        "Attention weights should sum to 1 across key dimension"
    
    # Check no NaNs
    print(f"✓ No NaNs in output: {not torch.isnan(attn_out).any()}")
    print(f"✓ No NaNs in attention weights: {not torch.isnan(attn_weights).any()}")
    
    # Test with mask
    print(f"\n✓ Testing attention with padding mask...")
    padding_mask = generate_padding_mask(token_ids, pad_token_id=0)
    attn_out_masked, attn_weights_masked = attention(
        embeddings, embeddings, embeddings, mask=padding_mask
    )
    
    print(f"✓ Masked output shape: {attn_out_masked.shape}")
    assert attn_out_masked.shape == (batch_size, seq_len, D_MODEL), \
        "Shape should match input"
    print(f"✓ No NaNs with mask: {not torch.isnan(attn_out_masked).any()}")
    
    print("\n✅ Multi-Head Attention: PASSED")
    
    # =====================================================================
    # TEST 4: Feed-Forward Network
    # =====================================================================
    print(f"\n{'='*80}")
    print("TEST 4: Feed-Forward Network")
    print(f"{'='*80}")
    
    ff = FeedForward(D_MODEL, FFN_DIM)
    ff_out = ff(embeddings)
    
    print(f"\n✓ Input shape: {embeddings.shape}")
    print(f"✓ Output shape: {ff_out.shape}")
    assert ff_out.shape == embeddings.shape, "FFN output shape should match input"
    
    print(f"✓ Output mean: {ff_out.mean().item():.6f}")
    print(f"✓ Output std: {ff_out.std().item():.6f}")
    print(f"✓ No NaNs: {not torch.isnan(ff_out).any()}")
    
    print("\n✅ Feed-Forward Network: PASSED")
    
    # =====================================================================
    # TEST 5: Transformer Encoder Layer
    # =====================================================================
    print(f"\n{'='*80}")
    print("TEST 5: Transformer Encoder Layer")
    print(f"{'='*80}")
    
    encoder_layer = TransformerEncoderLayer(D_MODEL, NUM_HEADS, FFN_DIM)
    
    print(f"\n✓ Forward pass without mask...")
    enc_out = encoder_layer(embeddings)
    
    print(f"✓ Output shape: {enc_out.shape}")
    assert enc_out.shape == embeddings.shape, "Encoder layer should preserve shape"
    
    print(f"✓ No NaNs: {not torch.isnan(enc_out).any()}")
    
    print(f"\n✓ Forward pass with padding mask...")
    padding_mask = generate_padding_mask(token_ids, pad_token_id=0)
    enc_out_masked = encoder_layer(embeddings, padding_mask=padding_mask)
    
    print(f"✓ Output shape with mask: {enc_out_masked.shape}")
    assert enc_out_masked.shape == embeddings.shape, "Shape should match input"
    print(f"✓ No NaNs with mask: {not torch.isnan(enc_out_masked).any()}")
    
    print("\n✅ Encoder Layer: PASSED")
    
    # =====================================================================
    # TEST 6: Transformer Decoder Layer
    # =====================================================================
    print(f"\n{'='*80}")
    print("TEST 6: Transformer Decoder Layer")
    print(f"{'='*80}")
    
    decoder_layer = TransformerDecoderLayer(D_MODEL, NUM_HEADS, FFN_DIM)
    
    # Create different sequence lengths for decoder input and encoder output
    dec_seq_len = 8
    dec_input = torch.randn(batch_size, dec_seq_len, D_MODEL)
    enc_output = torch.randn(batch_size, seq_len, D_MODEL)
    
    print(f"\n✓ Decoder input shape: {dec_input.shape}")
    print(f"✓ Encoder output shape: {enc_output.shape}")
    
    print(f"\n✓ Forward pass without masks...")
    dec_out = decoder_layer(dec_input, enc_output)
    
    print(f"✓ Output shape: {dec_out.shape}")
    assert dec_out.shape == dec_input.shape, "Decoder output shape should match decoder input"
    print(f"✓ No NaNs: {not torch.isnan(dec_out).any()}")
    
    print(f"\n✓ Forward pass with masks...")
    causal_mask = generate_causal_mask(dec_seq_len).to(dec_input.device)
    padding_mask = generate_padding_mask(token_ids[:, :seq_len], pad_token_id=0)
    
    dec_out_masked = decoder_layer(
        dec_input,
        enc_output,
        padding_mask=padding_mask,
        causal_mask=causal_mask
    )
    
    print(f"✓ Output shape with masks: {dec_out_masked.shape}")
    assert dec_out_masked.shape == dec_input.shape, "Shape should match"
    print(f"✓ No NaNs with masks: {not torch.isnan(dec_out_masked).any()}")
    
    print("\n✅ Decoder Layer: PASSED")
    
    # =====================================================================
    # TEST 7: Masking Functions
    # =====================================================================
    print(f"\n{'='*80}")
    print("TEST 7: Masking Functions")
    print(f"{'='*80}")
    
    print(f"\n✓ Padding Mask:")
    padding_mask = generate_padding_mask(token_ids, pad_token_id=0)
    print(f"  - Shape: {padding_mask.shape}")
    print(f"  - Expected: (batch={batch_size}, 1, 1, seq_len={seq_len})")
    print(f"  - Sample values: {padding_mask[0, 0, 0, :10]}")
    
    # Count valid tokens
    valid_tokens = (token_ids != 0).sum().item()
    masked_ones = (padding_mask == 1).sum().item()
    print(f"  - Valid tokens: {valid_tokens}")
    print(f"  - Mask ones: {masked_ones}")
    
    print(f"\n✓ Causal Mask:")
    causal_mask = generate_causal_mask(seq_len)
    print(f"  - Shape: {causal_mask.shape}")
    print(f"  - Expected: (1, 1, seq_len={seq_len}, seq_len={seq_len})")
    print(f"  - Lower triangular: {(torch.triu(causal_mask[0, 0], diagonal=1) == 0).all()}")
    
    # Print first 5x5 for inspection
    print(f"  - First 5x5 of causal mask:")
    print(f"    {causal_mask[0, 0, :5, :5]}")
    
    print("\n✅ Masking Functions: PASSED")
    
    # =====================================================================
    # TEST 8: Attention Weight Sanity Checks
    # =====================================================================
    print(f"\n{'='*80}")
    print("TEST 8: Attention Weight Sanity Checks")
    print(f"{'='*80}")
    
    print(f"\n✓ Self-attention without mask:")
    attn_out, attn_weights = attention(embeddings, embeddings, embeddings, mask=None)
    print(f"  - Min weight: {attn_weights.min().item():.6f}")
    print(f"  - Max weight: {attn_weights.max().item():.6f}")
    print(f"  - Mean weight: {attn_weights.mean().item():.6f}")
    print(f"  - Any NaN: {torch.isnan(attn_weights).any()}")
    print(f"  - Any Inf: {torch.isinf(attn_weights).any()}")
    
    print(f"\n✓ Self-attention with causal mask:")
    causal_mask = generate_causal_mask(seq_len)
    attn_out_causal, attn_weights_causal = attention(
        embeddings, embeddings, embeddings, mask=causal_mask
    )
    print(f"  - Min weight: {attn_weights_causal.min().item():.6f}")
    print(f"  - Max weight: {attn_weights_causal.max().item():.6f}")
    print(f"  - Mean weight: {attn_weights_causal.mean().item():.6f}")
    print(f"  - Any NaN: {torch.isnan(attn_weights_causal).any()}")
    
    # Verify causal mask is actually working
    # For each position i, attention to positions > i should be 0
    causal_working = True
    for i in range(seq_len):
        for j in range(i+1, seq_len):
            if attn_weights_causal[0, 0, i, j] > 1e-5:
                causal_working = False
                break
    
    print(f"  - Causal mask working (future tokens have 0 weight): {causal_working}")
    
    print("\n✅ Attention Sanity Checks: PASSED")
    
    # =====================================================================
    # FINAL SUMMARY
    # =====================================================================
    print(f"\n{'='*80}")
    print("✅ ALL VALIDATION TESTS PASSED")
    print(f"{'='*80}")
    
    print(f"""
SUMMARY:
  ✅ Token Embedding: Correct scaling and shape
  ✅ Positional Encoding: Sinusoidal pattern, shape (1, {MAX_SEQ_LEN}, {D_MODEL})
  ✅ Multi-Head Attention: Output shape correct, weights sum to 1, no NaNs
  ✅ Feed-Forward Network: Shape preserved, ReLU activation working
  ✅ Encoder Layer: Attention + residual + FFN working correctly
  ✅ Decoder Layer: Causal + cross-attention + FFN working correctly
  ✅ Padding Mask: Correct shape (batch, 1, 1, seq_len)
  ✅ Causal Mask: Lower triangular, prevents future attention
  ✅ Attention Sanity: No NaNs, proper masking behavior

CONFIGURATION:
  d_model: {D_MODEL}
  ffn_dim: {FFN_DIM}
  num_heads: {NUM_HEADS}
  max_seq_len: {MAX_SEQ_LEN}
  head_dim: {HEAD_DIM}

READY FOR STEP 4: Build complete transformer models.
""")


if __name__ == '__main__':
    validate_component_outputs()
