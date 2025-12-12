"""
MiniGPT-MedLM: Decoder-Only Transformer for Medical Text Generation.

GPT-style autoregressive language model for next-token prediction.
Architecture:
- Token + Positional Embeddings
- 2 Decoder Layers (causal self-attention)
- Language Modeling Head
"""

import torch
import torch.nn as nn
import math
from decoder_layers import DecoderLayer


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding (Vaswani et al. 2017).
    
    Fixed, non-learnable embeddings that encode position information.
    Preferred over learnable for generalization to longer sequences.
    
    Args:
        d_model: Embedding dimension
        max_len: Maximum sequence length
    """
    
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        
        # Create positional encoding matrix [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Compute div_term for sinusoidal functions
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension: [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter, but part of state_dict)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token embeddings [batch, seq_len, d_model]
        Returns:
            x + positional encoding [batch, seq_len, d_model]
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x


class MiniGPT_MedLM(nn.Module):
    """
    Decoder-Only Transformer for Medical Language Modeling.
    
    Architecture:
        Token Embedding (scaled by √d_model)
        + Positional Encoding
        → 2× Decoder Layers (causal self-attention + FFN)
        → LayerNorm
        → Language Modeling Head (Linear to vocab_size)
    
    Args:
        vocab_size: Size of vocabulary (~10,006)
        d_model: Embedding dimension (128)
        num_heads: Number of attention heads (4)
        num_layers: Number of decoder layers (2)
        ffn_dim: Feed-forward hidden dimension (512)
        max_seq_len: Maximum sequence length (128)
        dropout: Dropout rate (0.1)
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        ffn_dim: int = 512,
        max_seq_len: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        # Token embedding (scaled by √d_model as in original Transformer)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.embedding_scale = math.sqrt(d_model)
        
        # Positional encoding (sinusoidal, fixed)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Dropout after embeddings
        self.dropout = nn.Dropout(dropout)
        
        # Stack of decoder layers
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer normalization (post-norm after all layers)
        self.ln_final = nn.LayerNorm(d_model)
        
        # Language modeling head (project to vocabulary)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying: share embedding and output weights (standard practice)
        self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize embeddings
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        
        # Initialize linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass for next-token prediction.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            mask: Combined causal + padding mask (optional)
        
        Returns:
            Logits [batch, seq_len, vocab_size]
        """
        # Token embeddings (scaled)
        x = self.token_embedding(input_ids) * self.embedding_scale  # [batch, seq_len, d_model]
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Dropout
        x = self.dropout(x)
        
        # Pass through decoder layers
        for layer in self.decoder_layers:
            x = layer(x, mask=mask)
        
        # Final layer normalization
        x = self.ln_final(x)
        
        # Project to vocabulary
        logits = self.lm_head(x)  # [batch, seq_len, vocab_size]
        
        return logits
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: int = 50,
        pad_token_id: int = 0
    ) -> torch.Tensor:
        """
        Autoregressive text generation.
        
        Args:
            input_ids: Starting tokens [batch, seq_len]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (1.0 = no change, <1 = sharper, >1 = softer)
            top_k: Keep only top k tokens for sampling (0 = no filtering)
            pad_token_id: Padding token ID
        
        Returns:
            Generated token IDs [batch, seq_len + max_new_tokens]
        """
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get predictions for current sequence
                # Crop if sequence exceeds max_seq_len
                input_ids_crop = input_ids if input_ids.size(1) <= self.max_seq_len else input_ids[:, -self.max_seq_len:]
                
                # Forward pass
                logits = self.forward(input_ids_crop)  # [batch, seq_len, vocab_size]
                
                # Get logits for last position (next token prediction)
                logits = logits[:, -1, :]  # [batch, vocab_size]
                
                # Apply temperature
                logits = logits / temperature
                
                # Top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < top_k_logits[:, -1, None]] = float('-inf')
                
                # Sample from distribution
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # [batch, 1]
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids


def test_model():
    """Test MiniGPT-MedLM model."""
    print("Testing MiniGPT-MedLM...")
    
    # Config
    vocab_size = 10006
    batch_size = 2
    seq_len = 20
    d_model = 128
    num_heads = 4
    num_layers = 2
    ffn_dim = 512
    
    # Create model
    model = MiniGPT_MedLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        ffn_dim=ffn_dim,
        max_seq_len=128
    )
    
    print(f"\nModel created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test 1: Forward pass
    print("\n1. Testing forward pass...")
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = model(input_ids)
    assert logits.shape == (batch_size, seq_len, vocab_size)
    print(f"✓ Output shape: {logits.shape}")
    
    # Test 2: Forward pass with mask
    print("\n2. Testing forward pass with causal mask...")
    from masks import generate_causal_mask
    causal_mask = generate_causal_mask(seq_len, device=input_ids.device)
    logits = model(input_ids, mask=causal_mask)
    assert logits.shape == (batch_size, seq_len, vocab_size)
    print(f"✓ Output shape with mask: {logits.shape}")
    
    # Test 3: Text generation
    print("\n3. Testing text generation...")
    model.eval()
    prompt = torch.randint(0, vocab_size, (1, 5))
    generated = model.generate(prompt, max_new_tokens=10, temperature=1.0, top_k=50)
    assert generated.shape[1] == 15  # 5 + 10
    print(f"✓ Generated sequence length: {generated.shape[1]}")
    
    # Test 4: Weight tying
    print("\n4. Verifying weight tying...")
    assert model.lm_head.weight is model.token_embedding.weight
    print("✓ Embedding and LM head weights are tied")
    
    print("\n✅ All model tests passed!")


if __name__ == "__main__":
    test_model()
