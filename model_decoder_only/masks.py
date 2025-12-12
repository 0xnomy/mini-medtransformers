"""
Masking utilities for Decoder-Only Transformer (MiniGPT-MedLM).

Implements:
- Causal mask (prevents attending to future tokens)
- Padding mask (prevents attending to padding tokens)
- Combined mask application
"""

import torch
import torch.nn as nn


def generate_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """
    Generate causal (triangular) mask for autoregressive attention.
    
    Prevents attention to future tokens. Lower triangular matrix of ones,
    upper triangular filled with -inf (or zeros, then applied as additive mask).
    
    Args:
        seq_len: Sequence length
        device: Device to create tensor on
        
    Returns:
        Causal mask of shape [seq_len, seq_len]
        - 0.0 for allowed positions (present and past)
        - -inf for forbidden positions (future)
    
    Example:
        For seq_len=4:
        [[0, -inf, -inf, -inf],
         [0,    0, -inf, -inf],
         [0,    0,    0, -inf],
         [0,    0,    0,    0]]
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask


def generate_padding_mask(
    input_ids: torch.Tensor, 
    pad_token_id: int
) -> torch.Tensor:
    """
    Generate padding mask from input token IDs.
    
    Args:
        input_ids: Token IDs, shape [batch, seq_len]
        pad_token_id: ID of padding token
        
    Returns:
        Padding mask of shape [batch, 1, 1, seq_len]
        - 0.0 for real tokens
        - -inf for padding tokens
        
    Note:
        Shape [batch, 1, 1, seq_len] broadcasts correctly with attention scores
        of shape [batch, num_heads, seq_len, seq_len]
    """
    # Create mask: True for padding positions
    padding_mask = (input_ids == pad_token_id)  # [batch, seq_len]
    
    # Convert to additive mask: 0.0 for real tokens, -inf for padding
    padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
    padding_mask = padding_mask.float().masked_fill(padding_mask, float('-inf'))
    
    return padding_mask


def combine_masks(
    causal_mask: torch.Tensor,
    padding_mask: torch.Tensor = None
) -> torch.Tensor:
    """
    Combine causal mask and padding mask.
    
    Args:
        causal_mask: Shape [seq_len, seq_len]
        padding_mask: Shape [batch, 1, 1, seq_len] or None
        
    Returns:
        Combined mask that can be added to attention scores
        Shape [batch, 1, seq_len, seq_len] or [seq_len, seq_len]
    """
    if padding_mask is None:
        return causal_mask
    
    # Expand causal mask to match batch dimension
    # causal_mask: [seq_len, seq_len] -> [1, 1, seq_len, seq_len]
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
    
    # Add masks (both use -inf for forbidden positions)
    combined = causal_mask + padding_mask  # Broadcasting handles dimensions
    
    return combined


def test_masks():
    """Unit tests for mask generation."""
    print("Testing mask generation...")
    
    # Test 1: Causal mask
    print("\n1. Testing causal mask (seq_len=4):")
    causal = generate_causal_mask(4)
    print(causal)
    assert causal.shape == (4, 4)
    assert causal[0, 1] == float('-inf'), "Future tokens should be masked"
    assert causal[3, 0] == 0.0, "Past tokens should be visible"
    print("✓ Causal mask correct")
    
    # Test 2: Padding mask
    print("\n2. Testing padding mask:")
    input_ids = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])  # pad_id = 0
    pad_mask = generate_padding_mask(input_ids, pad_token_id=0)
    print(f"Shape: {pad_mask.shape}")
    assert pad_mask.shape == (2, 1, 1, 5)
    print("✓ Padding mask shape correct")
    
    # Test 3: Combined mask
    print("\n3. Testing combined mask:")
    causal = generate_causal_mask(5)
    combined = combine_masks(causal, pad_mask)
    print(f"Combined shape: {combined.shape}")
    assert combined.shape == (2, 1, 5, 5)
    print("✓ Combined mask correct")
    
    print("\n✅ All mask tests passed!")


if __name__ == "__main__":
    test_masks()
