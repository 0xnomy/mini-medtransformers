"""
Text Generation Script for MiniGPT-MedLM.

Autoregressive generation with multiple sampling strategies:
- Greedy decoding
- Top-k sampling  
- Temperature sampling
"""

import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from gpt_model import MiniGPT_MedLM
from step2_tokenizer import Tokenizer


# ============================================================================
# GENERATION FUNCTIONS
# ============================================================================

def generate_greedy(
    model: MiniGPT_MedLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    device: torch.device = None
) -> str:
    """
    Greedy decoding: always select most likely token.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        prompt: Starting text
        max_new_tokens: Number of tokens to generate
        device: Device to run on
    
    Returns:
        Generated text
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    # Tokenize prompt
    tokens = tokenizer.encode(prompt, add_cls=False, pad=False)
    input_ids = torch.tensor([tokens], device=device)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop if too long
            input_crop = input_ids if input_ids.size(1) <= model.max_seq_len else input_ids[:, -model.max_seq_len:]
            
            # Forward pass
            logits = model(input_crop)
            
            # Get most likely next token (greedy)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            
            # Append
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Stop if EOS token (if defined)
            if hasattr(tokenizer, 'eos_id') and next_token.item() == tokenizer.eos_id:
                break
    
    # Decode
    generated_tokens = input_ids[0].cpu().tolist()
    return tokenizer.decode(generated_tokens)


def generate_top_k(
    model: MiniGPT_MedLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 50,
    device: torch.device = None
) -> str:
    """
    Top-k sampling: sample from k most likely tokens.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        prompt: Starting text
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature (lower = more deterministic)
        top_k: Number of top tokens to consider
        device: Device to run on
    
    Returns:
        Generated text
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Tokenize prompt
    tokens = tokenizer.encode(prompt, add_cls=False, pad=False)
    input_ids = torch.tensor([tokens], device=device)
    
    # Use model's built-in generation
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        pad_token_id=tokenizer.pad_id
    )
    
    # Decode
    generated_tokens = output_ids[0].cpu().tolist()
    return tokenizer.decode(generated_tokens)


def generate_nucleus(
    model: MiniGPT_MedLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_p: float = 0.9,
    device: torch.device = None
) -> str:
    """
    Nucleus (top-p) sampling: sample from smallest set with cumulative probability > p.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        prompt: Starting text
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature
        top_p: Cumulative probability threshold
        device: Device to run on
    
    Returns:
        Generated text
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    # Tokenize prompt
    tokens = tokenizer.encode(prompt, add_cls=False, pad=False)
    input_ids = torch.tensor([tokens], device=device)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop if too long
            input_crop = input_ids if input_ids.size(1) <= model.max_seq_len else input_ids[:, -model.max_seq_len:]
            
            # Forward pass
            logits = model(input_crop)[:, -1, :] / temperature
            
            # Sort by probability
            probs = torch.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            
            # Compute cumulative probabilities
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # Remove tokens with cumulative probability > top_p
            sorted_indices_to_remove = cumsum_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Set filtered logits to -inf
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[0, indices_to_remove] = float('-inf')
            
            # Sample
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            input_ids = torch.cat([input_ids, next_token], dim=1)
    
    # Decode
    generated_tokens = input_ids[0].cpu().tolist()
    return tokenizer.decode(generated_tokens)


# ============================================================================
# DEMO
# ============================================================================

def demo_generation():
    """Demo different generation strategies."""
    print("="*80)
    print("MiniGPT-MedLM TEXT GENERATION DEMO")
    print("="*80)
    
    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = Tokenizer.load('artifacts/tokenizer.json')
    print(f"   ✓ Loaded (vocab size: {tokenizer.vocab_size()})")
    
    # Load model
    print("\n2. Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_path = 'artifacts/medLM_models/miniGPT_medLM.pt'
    
    model = MiniGPT_MedLM(
        vocab_size=tokenizer.vocab_size(),
        d_model=128,
        num_heads=4,
        num_layers=2,
        ffn_dim=512,
        max_seq_len=128
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"   ✓ Loaded from {model_path}")
    
    # Test prompts
    prompts = [
        "Diabetes is a condition",
        "Treatment options include",
        "Symptoms may include"
    ]
    
    # Generate with different strategies
    for prompt in prompts:
        print("\n" + "="*80)
        print(f"PROMPT: \"{prompt}\"")
        print("="*80)
        
        # Greedy
        print("\n[GREEDY DECODING]")
        text_greedy = generate_greedy(model, tokenizer, prompt, max_new_tokens=30, device=device)
        print(text_greedy)
        
        # Top-k (temperature=0.8)
        print("\n[TOP-K SAMPLING (k=50, temp=0.8)]")
        text_topk = generate_top_k(model, tokenizer, prompt, max_new_tokens=30, temperature=0.8, top_k=50, device=device)
        print(text_topk)
        
        # Nucleus (top-p=0.9)
        print("\n[NUCLEUS SAMPLING (p=0.9, temp=1.0)]")
        text_nucleus = generate_nucleus(model, tokenizer, prompt, max_new_tokens=30, temperature=1.0, top_p=0.9, device=device)
        print(text_nucleus)
    
    print("\n" + "="*80)
    print("✓ Generation demo complete!")
    print("="*80)


if __name__ == "__main__":
    demo_generation()
