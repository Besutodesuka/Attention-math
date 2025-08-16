"""
Simple Example: Using Attention for Text Processing
=================================================

This script shows a practical example of how attention can be used
to process text sequences and understand relationships between words.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SimpleTextAttention(nn.Module):
    """
    A simple attention-based model for text processing.
    This demonstrates how attention can capture relationships between words.
    """
    
    def __init__(self, vocab_size, embedding_dim, attention_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads=2, batch_first=True)
        self.output_projection = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, x):
        # x: (batch_size, seq_len) - token indices
        batch_size, seq_len = x.shape
        
        # Step 1: Convert tokens to embeddings
        embeddings = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # Step 2: Apply self-attention
        # This allows each word to attend to all other words
        attended_output, attention_weights = self.attention(
            embeddings, embeddings, embeddings
        )
        
        # Step 3: Project to vocabulary size
        logits = self.output_projection(attended_output)
        
        return logits, attention_weights

def demonstrate_text_attention():
    """
    Demonstrate attention mechanism on a simple text example.
    """
    print("=" * 60)
    print("TEXT PROCESSING WITH ATTENTION")
    print("=" * 60)
    
    # Create a simple vocabulary
    vocab = ["<PAD>", "the", "cat", "sat", "on", "mat", "dog", "ran", "fast"]
    vocab_size = len(vocab)
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    
    print(f"Vocabulary: {vocab}")
    print(f"Vocabulary size: {vocab_size}")
    
    # Create a simple sentence
    sentence = ["the", "cat", "sat", "on", "the", "mat"]
    sentence_indices = [word_to_idx[word] for word in sentence]
    
    print(f"\nInput sentence: {sentence}")
    print(f"Sentence indices: {sentence_indices}")
    
    # Convert to tensor
    x = torch.tensor([sentence_indices], dtype=torch.long)
    print(f"Input tensor shape: {x.shape}")
    
    # Initialize model
    embedding_dim = 8
    attention_dim = 8
    model = SimpleTextAttention(vocab_size, embedding_dim, attention_dim)
    
    print(f"\nModel initialized with:")
    print(f"  Embedding dimension: {embedding_dim}")
    print(f"  Attention dimension: {attention_dim}")
    
    # Forward pass
    with torch.no_grad():
        logits, attention_weights = model(x)
    
    print(f"\nOutput logits shape: {logits.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    # Analyze attention patterns
    print(f"\n" + "="*40)
    print("ATTENTION PATTERN ANALYSIS")
    print("="*40)
    
    # Get attention weights for the first (and only) batch
    weights = attention_weights[0]  # Shape: (num_heads, seq_len, seq_len)
    
    for head in range(weights.shape[0]):
        print(f"\nAttention Head {head}:")
        print("Query → Key attention weights:")
        
        # Create a nice table
        print("      ", end="")
        for word in sentence:
            print(f"{word:>8}", end="")
        print()
        
        for i, query_word in enumerate(sentence):
            print(f"{query_word:>6}", end="")
            for j in range(len(sentence)):
                weight = weights[head, i, j].item()
                print(f"{weight:>8.3f}", end="")
            print()
    
    # Find the strongest attention for each word
    print(f"\n" + "="*40)
    print("STRONGEST ATTENTION RELATIONSHIPS")
    print("="*40)
    
    for head in range(weights.shape[0]):
        print(f"\nHead {head}:")
        for i, query_word in enumerate(sentence):
            # Find the position with highest attention
            max_attn = weights[head, i].max()
            max_pos = weights[head, i].argmax()
            print(f"  '{query_word}' (pos {i}) → '{sentence[max_pos]}' (pos {max_pos}) "
                  f"[weight: {max_attn:.3f}]")
    
    return attention_weights, sentence

def explain_attention_insights(attention_weights, sentence):
    """
    Explain what the attention patterns tell us about the text.
    """
    print(f"\n" + "="*60)
    print("INTERPRETING ATTENTION PATTERNS")
    print("="*60)
    
    print("""
What the attention mechanism learned:

1. **Self-Attention**: Each word can attend to all other words, including itself
2. **Content-Based Relationships**: Words that are semantically related get higher attention
3. **Positional Independence**: Attention is not limited by distance between words
4. **Context Awareness**: Each word's representation is influenced by its context

In our example sentence "the cat sat on the mat":
- "cat" might attend strongly to "sat" (subject-verb relationship)
- "sat" might attend to "on" (verb-preposition relationship)  
- "the" might attend to the noun it modifies
- "mat" might attend to "on" (preposition-object relationship)

The attention weights show us which relationships the model considers important!
    """)

def create_attention_visualization(attention_weights, sentence):
    """
    Create a simple text-based visualization of attention patterns.
    """
    print(f"\n" + "="*60)
    print("ATTENTION VISUALIZATION")
    print("="*60)
    
    weights = attention_weights[0]  # First batch
    
    for head in range(weights.shape[0]):
        print(f"\nAttention Head {head}:")
        print("-" * 50)
        
        for i, query_word in enumerate(sentence):
            print(f"\n'{query_word}' attends to:")
            
            # Sort attention weights for this query
            attn_scores = weights[head, i]
            sorted_indices = torch.argsort(attn_scores, descending=True)
            
            for rank, idx in enumerate(sorted_indices):
                word = sentence[idx]
                score = attn_scores[idx].item()
                bar_length = int(score * 20)  # Scale for visualization
                bar = "█" * bar_length
                print(f"  {rank+1:2d}. '{word:>6}' [{bar:<20}] {score:.3f}")

if __name__ == "__main__":
    print("Welcome to the Text Attention Example!")
    print("This demonstrates how attention mechanisms work on actual text.")
    
    # Run the demonstration
    attention_weights, sentence = demonstrate_text_attention()
    
    # Explain the insights
    explain_attention_insights(attention_weights, sentence)
    
    # Create visualization
    create_attention_visualization(attention_weights, sentence)
    
    print(f"\n" + "="*60)
    print("EXAMPLE COMPLETE!")
    print("="*60)
    print("""
Key Insights:
1. Attention allows each word to 'see' all other words
2. Attention weights show which relationships are important
3. Multi-head attention can learn different types of relationships
4. The model learns to focus on semantically related words
5. Attention patterns are interpretable and meaningful

This demonstrates why attention is so powerful for natural language processing!
    """) 