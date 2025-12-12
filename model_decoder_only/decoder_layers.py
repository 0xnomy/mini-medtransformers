"""
Decoder Layer for MiniGPT-MedLM.

Implements a single transformer decoder block with:
- Causal multi-head self-attention
- Feed-forward network
- Layer normalization
- Residual connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention for decoder (causal).
    
    Args:
        d_model: Model dimension (128)
        num_heads: Number of attention heads (4)
    """
    
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Q, K, V projections
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        
        # Output projection
        self.out = nn.Linear(d_model, d_model)
        
        # Scale factor for dot product
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Attention mask [seq_len, seq_len] or [batch, 1, seq_len, seq_len]
                  Values should be 0.0 (attend) or -inf (mask)
        
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        # Linear projections and split into heads
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose to [batch, num_heads, seq_len, head_dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Scaled dot-product attention
        # scores: [batch, num_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores + mask  # Additive masking (-inf positions)
        
        # Softmax over key dimension
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        # [batch, num_heads, seq_len, seq_len] × [batch, num_heads, seq_len, head_dim]
        # -> [batch, num_heads, seq_len, head_dim]
        attn_output = torch.matmul(attn_weights, V)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()  # [batch, seq_len, num_heads, head_dim]
        attn_output = attn_output.view(batch_size, seq_len, d_model)  # [batch, seq_len, d_model]
        
        # Final linear projection
        output = self.out(attn_output)
        
        return output


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    
    Two linear layers with GELU activation.
    
    Args:
        d_model: Model dimension (128)
        ffn_dim: Hidden dimension (512, typically 4x d_model)
        dropout: Dropout rate
    """
    
    def __init__(self, d_model: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            [batch, seq_len, d_model]
        """
        x = self.linear1(x)
        x = F.gelu(x)  # GELU activation (standard in GPT models)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class DecoderLayer(nn.Module):
    """
    Single Transformer Decoder Layer.
    
    Architecture:
        x -> LayerNorm -> CausalSelfAttention -> Residual
          -> LayerNorm -> FeedForward -> Residual
    
    Args:
        d_model: Model dimension (128)
        num_heads: Number of attention heads (4)
        ffn_dim: Feed-forward hidden dimension (512)
        dropout: Dropout rate (0.1)
    """
    
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        
        # Layer normalization (pre-norm architecture, standard in GPT)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        # Causal self-attention
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads)
        
        # Feed-forward network
        self.ffn = FeedForward(d_model, ffn_dim, dropout)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Causal + padding mask
        
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        # Self-attention with residual connection (pre-norm)
        residual = x
        x = self.ln1(x)
        x = self.self_attn(x, mask=mask)
        x = self.dropout1(x)
        x = residual + x
        
        # Feed-forward with residual connection (pre-norm)
        residual = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = self.dropout2(x)
        x = residual + x
        
        return x


def test_decoder_layer():
    """Test decoder layer functionality."""
    print("Testing Decoder Layer...")
    
    # Config
    batch_size = 2
    seq_len = 10
    d_model = 128
    num_heads = 4
    ffn_dim = 512
    
    # Create decoder layer
    layer = DecoderLayer(d_model, num_heads, ffn_dim, dropout=0.1)
    
    # Random input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Test 1: Forward pass without mask
    print("\n1. Testing forward pass without mask...")
    output = layer(x)
    assert output.shape == (batch_size, seq_len, d_model)
    print(f"✓ Output shape: {output.shape}")
    
    # Test 2: Forward pass with causal mask
    print("\n2. Testing forward pass with causal mask...")
    from masks import generate_causal_mask
    causal_mask = generate_causal_mask(seq_len)
    output = layer(x, mask=causal_mask)
    assert output.shape == (batch_size, seq_len, d_model)
    print(f"✓ Output shape with mask: {output.shape}")
    
    # Test 3: Verify residual connections work
    print("\n3. Testing residual connections...")
    layer.eval()
    with torch.no_grad():
        out1 = layer(x)
        out2 = layer(x)
        assert torch.allclose(out1, out2), "Residual connections should be deterministic in eval mode"
    print("✓ Residual connections working")
    
    print("\n✅ All decoder layer tests passed!")


if __name__ == "__main__":
    test_decoder_layer()
