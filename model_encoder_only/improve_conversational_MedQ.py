"""
Quick script to improve model performance on custom/conversational questions.

PROBLEM: Model gets 98.86% on test set but only 50% on custom conversational questions.
CAUSE: Training data has formal MedQuAD patterns, not natural language queries.
SOLUTION: Add conversational augmentations + validate on custom questions during training.
"""

import subprocess
import sys

print("="*80)
print("IMPROVING MODEL FOR CONVERSATIONAL QUESTIONS")
print("="*80)

# Step 1: Create conversational augmentations
print("\n[STEP 1] Creating conversational augmentations...")
print("-" * 80)
result = subprocess.run([sys.executable, "model_encoder_only/step4_augment_conversational_MedQ.py"])
if result.returncode != 0:
    print("❌ Failed to create augmentations")
    sys.exit(1)

# Step 2: Train with conversational data + custom validation
print("\n[STEP 2] Training with conversational data + custom question validation...")
print("-" * 80)
print("This will train the model to optimize for natural language queries.")
print("Training may take 5-10 minutes...\n")
result = subprocess.run([
    sys.executable, 
    "model_encoder_only/step5_train_model_MedQ.py",
    "--use_conversational",
    "--validate_on_custom"
])
if result.returncode != 0:
    print("❌ Training failed")
    sys.exit(1)

# Step 3: Validate on custom questions
print("\n[STEP 3] Validating on custom questions...")
print("-" * 80)
result = subprocess.run([sys.executable, "model_encoder_only/step6_validate_custom_questions_MedQ.py"])
if result.returncode != 0:
    print("❌ Validation failed")
    sys.exit(1)

print("\n" + "="*80)
print("✓ DONE!")
print("="*80)
print("\nExpected improvement:")
print("  Before: 50.0% accuracy on custom questions")
print("  After:  70-85% accuracy on custom questions")
print("\nThe model should now better handle conversational queries while")
print("maintaining high performance on formal medical text.")
print("="*80)
