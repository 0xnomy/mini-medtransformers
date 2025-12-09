"""
Unit Tests for MiniGPT-MedLM (Decoder-Only Transformer).

Tests:
1. Mask generation and shapes
2. Decoder layer forward pass
3. Full model forward pass
4. Text generation
5. End-to-end training step
"""

import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from masks import generate_causal_mask, generate_padding_mask, combine_masks
from decoder_layers import DecoderLayer, MultiHeadSelfAttention, FeedForward
from gpt_model import MiniGPT_MedLM, PositionalEncoding
from step2_tokenizer import Tokenizer


class TestMasks:
    """Test mask generation."""
    
    @staticmethod
    def test_causal_mask():
        print("\n[TEST] Causal Mask Generation")
        print("-" * 60)
        
        # Test shape
        seq_len = 10
        mask = generate_causal_mask(seq_len)
        assert mask.shape == (seq_len, seq_len), f"Expected shape ({seq_len}, {seq_len}), got {mask.shape}"
        print(f"✓ Shape correct: {mask.shape}")
        
        # Test values
        assert mask[0, 1] == float('-inf'), "Future position should be masked"
        assert mask[5, 2] == 0.0, "Past position should be visible"
        assert mask[5, 8] == float('-inf'), "Future position should be masked"
        print("✓ Mask values correct (0.0 for past, -inf for future)")
        
        # Test diagonal
        assert torch.all(torch.diagonal(mask) == 0.0), "Diagonal should be 0 (attend to self)"
        print("✓ Diagonal correct (can attend to current position)")
        
        print("✅ PASS: Causal mask")
    
    @staticmethod
    def test_padding_mask():
        print("\n[TEST] Padding Mask Generation")
        print("-" * 60)
        
        # Create input with padding (pad_id = 0)
        input_ids = torch.tensor([
            [1, 2, 3, 4, 0, 0],  # 2 padding tokens
            [5, 6, 0, 0, 0, 0]   # 4 padding tokens
        ])
        
        mask = generate_padding_mask(input_ids, pad_token_id=0)
        
        # Test shape
        assert mask.shape == (2, 1, 1, 6), f"Expected shape (2, 1, 1, 6), got {mask.shape}"
        print(f"✓ Shape correct: {mask.shape}")
        
        # Test values
        assert mask[0, 0, 0, 0] == 0.0, "Real token should not be masked"
        assert mask[0, 0, 0, 4] == float('-inf'), "Padding should be masked"
        assert mask[1, 0, 0, 2] == float('-inf'), "Padding should be masked"
        print("✓ Mask values correct (0.0 for real, -inf for padding)")
        
        print("✅ PASS: Padding mask")
    
    @staticmethod
    def test_combined_mask():
        print("\n[TEST] Combined Mask")
        print("-" * 60)
        
        seq_len = 5
        batch_size = 2
        
        # Generate both masks
        causal_mask = generate_causal_mask(seq_len)
        input_ids = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 8]])
        padding_mask = generate_padding_mask(input_ids, pad_token_id=0)
        
        # Combine
        combined = combine_masks(causal_mask, padding_mask)
        
        # Test shape
        assert combined.shape == (2, 1, 5, 5), f"Expected shape (2, 1, 5, 5), got {combined.shape}"
        print(f"✓ Combined shape correct: {combined.shape}")
        
        # Test that both masks are applied
        # Position [0, 0, 0, 4] should be -inf (both causal future AND padding)
        assert combined[0, 0, 0, 4] == float('-inf')
        print("✓ Both masks applied correctly")
        
        print("✅ PASS: Combined mask")


class TestDecoderComponents:
    """Test decoder layer components."""
    
    @staticmethod
    def test_multi_head_attention():
        print("\n[TEST] Multi-Head Self-Attention")
        print("-" * 60)
        
        batch_size = 2
        seq_len = 10
        d_model = 128
        num_heads = 4
        
        attn = MultiHeadSelfAttention(d_model, num_heads)
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Forward pass
        output = attn(x)
        
        assert output.shape == (batch_size, seq_len, d_model)
        print(f"✓ Output shape: {output.shape}")
        
        # Test with mask
        mask = generate_causal_mask(seq_len)
        output_masked = attn(x, mask=mask)
        
        assert output_masked.shape == (batch_size, seq_len, d_model)
        print(f"✓ Output shape with mask: {output_masked.shape}")
        
        # Outputs should be different with/without mask
        assert not torch.allclose(output, output_masked), "Mask should change output"
        print("✓ Mask affects attention output")
        
        print("✅ PASS: Multi-head attention")
    
    @staticmethod
    def test_feed_forward():
        print("\n[TEST] Feed-Forward Network")
        print("-" * 60)
        
        batch_size = 2
        seq_len = 10
        d_model = 128
        ffn_dim = 512
        
        ffn = FeedForward(d_model, ffn_dim, dropout=0.1)
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Forward pass
        output = ffn(x)
        
        assert output.shape == (batch_size, seq_len, d_model)
        print(f"✓ Output shape: {output.shape}")
        
        print("✅ PASS: Feed-forward network")
    
    @staticmethod
    def test_decoder_layer():
        print("\n[TEST] Decoder Layer")
        print("-" * 60)
        
        batch_size = 2
        seq_len = 10
        d_model = 128
        num_heads = 4
        ffn_dim = 512
        
        layer = DecoderLayer(d_model, num_heads, ffn_dim, dropout=0.1)
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Forward pass without mask
        output = layer(x)
        assert output.shape == (batch_size, seq_len, d_model)
        print(f"✓ Output shape: {output.shape}")
        
        # Forward pass with mask
        mask = generate_causal_mask(seq_len)
        output_masked = layer(x, mask=mask)
        assert output_masked.shape == (batch_size, seq_len, d_model)
        print(f"✓ Output shape with mask: {output_masked.shape}")
        
        print("✅ PASS: Decoder layer")


class TestModel:
    """Test full MiniGPT-MedLM model."""
    
    @staticmethod
    def test_positional_encoding():
        print("\n[TEST] Positional Encoding")
        print("-" * 60)
        
        d_model = 128
        max_len = 512
        batch_size = 2
        seq_len = 20
        
        pos_enc = PositionalEncoding(d_model, max_len)
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = pos_enc(x)
        
        assert output.shape == (batch_size, seq_len, d_model)
        print(f"✓ Output shape: {output.shape}")
        
        # Test that encoding is deterministic
        output2 = pos_enc(x)
        assert torch.allclose(output, output2), "Positional encoding should be deterministic"
        print("✓ Encoding is deterministic")
        
        print("✅ PASS: Positional encoding")
    
    @staticmethod
    def test_model_forward():
        print("\n[TEST] Model Forward Pass")
        print("-" * 60)
        
        vocab_size = 10006
        batch_size = 2
        seq_len = 20
        d_model = 128
        
        model = MiniGPT_MedLM(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=4,
            num_layers=2,
            ffn_dim=512,
            max_seq_len=128
        )
        
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Forward pass
        logits = model(input_ids)
        
        assert logits.shape == (batch_size, seq_len, vocab_size)
        print(f"✓ Logits shape: {logits.shape}")
        
        # Forward pass with mask
        mask = generate_causal_mask(seq_len)
        logits_masked = model(input_ids, mask=mask)
        
        assert logits_masked.shape == (batch_size, seq_len, vocab_size)
        print(f"✓ Logits shape with mask: {logits_masked.shape}")
        
        print("✅ PASS: Model forward pass")
    
    @staticmethod
    def test_weight_tying():
        print("\n[TEST] Weight Tying")
        print("-" * 60)
        
        model = MiniGPT_MedLM(vocab_size=10006, d_model=128)
        
        # Check that embedding and lm_head share weights
        assert model.lm_head.weight is model.token_embedding.weight
        print("✓ Embedding and LM head weights are tied")
        
        print("✅ PASS: Weight tying")
    
    @staticmethod
    def test_text_generation():
        print("\n[TEST] Text Generation")
        print("-" * 60)
        
        vocab_size = 10006
        model = MiniGPT_MedLM(vocab_size=vocab_size, d_model=128, num_layers=2)
        model.eval()
        
        # Generate text
        prompt = torch.randint(0, vocab_size, (1, 5))
        generated = model.generate(prompt, max_new_tokens=10, temperature=1.0, top_k=50)
        
        assert generated.shape[0] == 1
        assert generated.shape[1] == 15  # 5 + 10
        print(f"✓ Generated sequence shape: {generated.shape}")
        
        # Test with different parameters
        generated_greedy = model.generate(prompt, max_new_tokens=10, temperature=0.1, top_k=1)
        assert generated_greedy.shape[1] == 15
        print("✓ Greedy generation works (temperature=0.1)")
        
        print("✅ PASS: Text generation")


class TestTraining:
    """Test training components."""
    
    @staticmethod
    def test_loss_computation():
        print("\n[TEST] Loss Computation")
        print("-" * 60)
        
        vocab_size = 10006
        batch_size = 2
        seq_len = 20
        pad_token_id = 0
        
        model = MiniGPT_MedLM(vocab_size=vocab_size, d_model=128)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)
        
        # Create dummy data
        input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))  # Avoid pad token
        target_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
        
        # Forward pass
        logits = model(input_ids)
        
        # Compute loss
        loss = criterion(
            logits.reshape(-1, vocab_size),
            target_ids.reshape(-1)
        )
        
        assert loss.item() > 0, "Loss should be positive"
        assert not torch.isnan(loss), "Loss should not be NaN"
        print(f"✓ Loss computed: {loss.item():.4f}")
        
        print("✅ PASS: Loss computation")
    
    @staticmethod
    def test_backward_pass():
        print("\n[TEST] Backward Pass")
        print("-" * 60)
        
        vocab_size = 1000
        batch_size = 2
        seq_len = 10
        
        model = MiniGPT_MedLM(vocab_size=vocab_size, d_model=64, num_layers=1, ffn_dim=128)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Create dummy data
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        target_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Training step
        model.train()
        logits = model(input_ids)
        loss = criterion(logits.reshape(-1, vocab_size), target_ids.reshape(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check gradients exist
        has_gradients = any(p.grad is not None for p in model.parameters())
        assert has_gradients, "Model should have gradients after backward"
        print("✓ Gradients computed successfully")
        
        print("✅ PASS: Backward pass")


def run_all_tests():
    """Run all test suites."""
    print("="*80)
    print("MINIGEMPT-MEDLM UNIT TESTS")
    print("="*80)
    
    # Test masks
    print("\n" + "="*80)
    print("TEST SUITE 1: MASKS")
    print("="*80)
    TestMasks.test_causal_mask()
    TestMasks.test_padding_mask()
    TestMasks.test_combined_mask()
    
    # Test decoder components
    print("\n" + "="*80)
    print("TEST SUITE 2: DECODER COMPONENTS")
    print("="*80)
    TestDecoderComponents.test_multi_head_attention()
    TestDecoderComponents.test_feed_forward()
    TestDecoderComponents.test_decoder_layer()
    
    # Test model
    print("\n" + "="*80)
    print("TEST SUITE 3: MODEL")
    print("="*80)
    TestModel.test_positional_encoding()
    TestModel.test_model_forward()
    TestModel.test_weight_tying()
    TestModel.test_text_generation()
    
    # Test training
    print("\n" + "="*80)
    print("TEST SUITE 4: TRAINING")
    print("="*80)
    TestTraining.test_loss_computation()
    TestTraining.test_backward_pass()
    
    # Summary
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED!")
    print("="*80)
    print("\nTest Summary:")
    print("  ✓ Mask generation (causal + padding)")
    print("  ✓ Decoder components (attention, FFN, layer)")
    print("  ✓ Model architecture (forward pass, generation)")
    print("  ✓ Training (loss, gradients)")
    print("\nMiniGPT-MedLM is ready for training!")


if __name__ == "__main__":
    run_all_tests()
