"""
Augment training data with conversational/natural language question patterns.

This script addresses the distribution mismatch between formal MedQuAD questions
and real-world conversational queries by creating natural language paraphrases.
"""

import pandas as pd
import random
from pathlib import Path

# Set random seed for reproducibility
random.seed(42)

# File paths
BASE_DIR = Path(__file__).parent.parent
TRAIN_FILE = BASE_DIR / "data" / "processed" / "classification_train.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "classification_train_conversational.csv"

# Conversational transformation templates
CONVERSATIONAL_TEMPLATES = {
    "definition": [
        ("what is (are) {disease}?", "What is {disease}?"),
        ("what is (are) {disease}?", "Can you explain what {disease} is?"),
        ("what is (are) {disease}?", "What does {disease} mean?"),
        ("what is (are) {disease}?", "Tell me about {disease}."),
        ("what is (are) {disease}?", "I want to know about {disease}."),
        ("what is the outlook for {disease}?", "What's the prognosis for {disease}?"),
        ("what are the stages of {disease}?", "What stages does {disease} have?"),
        ("what are the complications of {disease}?", "What complications can {disease} cause?"),
    ],
    "treatment": [
        ("what are the treatments for {disease}?", "How can I treat {disease}?"),
        ("what are the treatments for {disease}?", "What is the best treatment for {disease}?"),
        ("what are the treatments for {disease}?", "How do I treat {disease}?"),
        ("what are the treatments for {disease}?", "What medications help with {disease}?"),
        ("what are the treatments for {disease}?", "How do doctors treat {disease}?"),
        ("what are the treatments for {disease}?", "What can be done for {disease}?"),
        ("what are the treatments for {disease}?", "How is {disease} treated?"),
        ("what are the treatments for {disease}?", "What are my options for treating {disease}?"),
    ],
    "symptom": [
        ("what are the symptoms of {disease}?", "What are the main symptoms of {disease}?"),
        ("what are the symptoms of {disease}?", "How do I know if I have {disease}?"),
        ("what are the symptoms of {disease}?", "What signs should I look for in {disease}?"),
        ("what are the symptoms of {disease}?", "What does {disease} feel like?"),
        ("what are the symptoms of {disease}?", "How can I tell if I have {disease}?"),
        ("what are the symptoms of {disease}?", "What are warning signs of {disease}?"),
    ],
    "cause": [
        ("what causes {disease}?", "Why do people get {disease}?"),
        ("what causes {disease}?", "What leads to {disease}?"),
        ("what causes {disease}?", "Why does {disease} happen?"),
        ("what causes {disease}?", "What triggers {disease}?"),
        ("what causes {disease}?", "How does someone get {disease}?"),
    ],
    "diagnosis": [
        ("how to diagnose {disease}?", "How is {disease} diagnosed?"),
        ("how to diagnose {disease}?", "What tests are needed to diagnose {disease}?"),
        ("how to diagnose {disease}?", "How do doctors test for {disease}?"),
        ("how to diagnose {disease}?", "How can I find out if I have {disease}?"),
        ("how to diagnose {disease}?", "What tests detect {disease}?"),
    ],
    "risk": [
        ("who is at risk for {disease}?", "Who is more likely to get {disease}?"),
        ("who is at risk for {disease}?", "What populations are susceptible to {disease}?"),
        ("who is at risk for {disease}?", "Who should worry about {disease}?"),
        ("who is at risk for {disease}?", "Am I at risk for {disease}?"),
        ("who is at risk for {disease}?", "Who gets {disease} most often?"),
    ],
    "genetics": [
        ("is {disease} inherited?", "Can {disease} be genetic?"),
        ("is {disease} inherited?", "Is {disease} hereditary?"),
        ("is {disease} inherited?", "Does {disease} run in families?"),
        ("is {disease} inherited?", "Can I inherit {disease}?"),
        ("what are the genetic changes related to {disease}?", "What genes cause {disease}?"),
        ("what are the genetic changes related to {disease}?", "Is {disease} caused by genetics?"),
    ],
    "general": [
        ("do you have information about {topic}", "Tell me about {topic}."),
        ("do you have information about {topic}", "What do you know about {topic}?"),
        ("do you have information about {topic}", "Information about {topic}?"),
        ("do you have information about {topic}", "I want to learn about {topic}."),
        ("how many people are affected by {disease}?", "How common is {disease}?"),
        ("how many people are affected by {disease}?", "How many people have {disease}?"),
        ("what research (or clinical trials) is being done for {disease}?", "What research exists on {disease}?"),
    ],
}


def extract_entity(question, label):
    """Extract disease/topic name from formal MedQuAD question."""
    question = question.lower().strip()
    
    # Common patterns in MedQuAD (order matters - try most specific first)
    patterns = [
        ("what are the treatments for ", "?"),
        ("what are the symptoms of ", "?"),
        ("what are the genetic changes related to ", "?"),
        ("what are the stages of ", "?"),
        ("what are the complications of ", "?"),
        ("how many people are affected by ", "?"),
        ("what research (or clinical trials) is being done for ", "?"),
        ("who is at risk for ", "?"),
        ("what is the outlook for ", "?"),
        ("what is (are) ", "?"),
        ("what causes ", "?"),
        ("how to diagnose ", "?"),
        ("do you have information about ", ""),
        ("is ", " inherited?"),
    ]
    
    for prefix, suffix in patterns:
        if question.startswith(prefix):
            # Handle cases where suffix might not match exactly
            if suffix and question.endswith(suffix):
                entity = question[len(prefix):-len(suffix)]
            elif not suffix:
                entity = question[len(prefix):]
            else:
                continue
            
            entity = entity.strip()
            if entity:  # Only return non-empty entities
                return entity
    
    return None


def create_conversational_variant(question, label):
    """Create a conversational variant of a formal question."""
    entity = extract_entity(question, label)
    
    if entity is None:
        return None
    
    # Get templates for this label
    templates = CONVERSATIONAL_TEMPLATES.get(label, [])
    if not templates:
        return None
    
    # Randomly select one template for variety
    formal_pattern, conversational_template = random.choice(templates)
    
    # Replace entity placeholder
    pattern_key = "{disease}" if "{disease}" in conversational_template else "{topic}"
    conversational = conversational_template.replace(pattern_key, entity)
    
    return conversational


def main():
    print("=" * 80)
    print("CONVERSATIONAL AUGMENTATION")
    print("=" * 80)
    
    # Load training data
    print("\n1. Loading training data...")
    df = pd.read_csv(TRAIN_FILE)
    print(f"   ✓ Loaded {len(df)} samples")
    
    # Sample questions to augment (limit to avoid data imbalance)
    # Focus on categories that failed in custom validation: treatment, diagnosis, genetics, risk, symptom
    priority_labels = ["treatment", "diagnosis", "genetics", "risk", "symptom", "cause"]
    
    print("\n2. Creating conversational variants...")
    augmentations = []
    
    for label in priority_labels:
        label_df = df[df['label'] == label].copy()
        print(f"\n   {label.upper()}")
        print(f"   - Original samples: {len(label_df)}")
        
        # Sample up to 200 questions per category
        sample_size = min(200, len(label_df))
        sampled = label_df.sample(n=sample_size, random_state=42)
        
        conversational_count = 0
        for idx, row in sampled.iterrows():
            variant = create_conversational_variant(row['text'], row['label'])
            if variant and variant.lower() != row['text'].lower():
                augmentations.append({
                    'text': variant,
                    'label': row['label'],
                    'is_augmented': 'conversational'
                })
                conversational_count += 1
        
        print(f"   - Created {conversational_count} conversational variants")
    
    # Also add some general conversational questions
    print(f"\n   GENERAL")
    general_df = df[df['label'] == 'general'].copy()
    print(f"   - Original samples: {len(general_df)}")
    sample_size = min(100, len(general_df))
    sampled = general_df.sample(n=sample_size, random_state=42)
    
    conversational_count = 0
    for idx, row in sampled.iterrows():
        variant = create_conversational_variant(row['text'], row['label'])
        if variant and variant.lower() != row['text'].lower():
            augmentations.append({
                'text': variant,
                'label': row['label'],
                'is_augmented': 'conversational'
            })
            conversational_count += 1
    
    print(f"   - Created {conversational_count} conversational variants")
    
    print(f"\n   TOTAL: Created {len(augmentations)} conversational augmentations")
    
    # Combine with original data
    print("\n3. Combining with original training data...")
    augmented_df = pd.DataFrame(augmentations)
    combined_df = pd.concat([df, augmented_df], ignore_index=True)
    print(f"   ✓ Total samples: {len(combined_df)} ({len(df)} original + {len(augmentations)} conversational)")
    
    # Save
    print(f"\n4. Saving to {OUTPUT_FILE.name}...")
    combined_df.to_csv(OUTPUT_FILE, index=False)
    print(f"   ✓ Saved successfully!")
    
    # Show distribution
    print("\n5. Class distribution:")
    for label in sorted(combined_df['label'].unique()):
        count = len(combined_df[combined_df['label'] == label])
        original = len(df[df['label'] == label])
        added = count - original
        print(f"   - {label:12s}: {count:5d} ({original:5d} original + {added:4d} conversational)")
    
    print("\n" + "=" * 80)
    print("✓ DONE! Use --train_file with this new dataset for training.")
    print("=" * 80)


if __name__ == "__main__":
    main()
