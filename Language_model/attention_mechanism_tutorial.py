"""
Attention Mechanism Tutorial: Mathematical Understanding with PyTorch
==================================================================

This script demonstrates the attention mechanism step-by-step, explaining the mathematical
concepts behind why attention works in transformer architectures.

Key Concepts Covered:
1. Query, Key, Value representation
2. Attention scores calculation
3. Softmax normalization
4. Weighted aggregation
5. Multi-head attention
6. Self-attention vs cross-attention

Mathematical Foundation:
- Attention(Q,K,V) = softmax(QK^T/√d_k)V
- Where Q, K, V are Query, Key, Value matrices
- d_k is the dimension of keys
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class AttentionMechanism:
    """
    A comprehensive implementation of attention mechanism with detailed explanations
    of each mathematical step.
    """
    
    def __init__(self, d_model: int, d_k: int, d_v: int, num_heads: int = 1):
        """
        Initialize attention mechanism.
        
        Args:
            d_model: Dimension of the model
            d_k: Dimension of keys (and queries)
            d_v: Dimension of values
            num_heads: Number of attention heads
        """
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.num_heads = num_heads
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_k * num_heads, bias=False)
        self.W_k = nn.Linear(d_model, d_k * num_heads, bias=False)
        self.W_v = nn.Linear(d_model, d_v * num_heads, bias=False)
        self.W_o = nn.Linear(d_v * num_heads, d_model, bias=False)
        
        # Scale factor for attention scores
        self.scale = d_k ** 0.5
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with detailed attention calculation.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            output: Transformed sequence
            attention_weights: Attention weights for visualization
        """
        batch_size, seq_len, _ = x.shape
        
        # Step 1: Linear projections to get Q, K, V
        Q = self.W_q(x)  # (batch_size, seq_len, d_k * num_heads)
        K = self.W_k(x)  # (batch_size, seq_len, d_k * num_heads)
        V = self.W_v(x)  # (batch_size, seq_len, d_v * num_heads)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_v).transpose(1, 2)
        
        # Step 2: Calculate attention scores
        # QK^T gives us the raw attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch_size, num_heads, seq_len, seq_len)
        
        # Step 3: Scale the scores
        # This prevents the softmax from having extremely sharp distributions
        scores = scores / self.scale
        
        # Step 4: Apply attention mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Step 5: Apply softmax to get attention weights
        # This normalizes the scores to sum to 1, creating a probability distribution
        attention_weights = F.softmax(scores, dim=-1)
        
        # Step 6: Apply attention weights to values
        # This creates the weighted sum of values based on attention scores
        context = torch.matmul(attention_weights, V)
        
        # Reshape back to original format
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.num_heads * self.d_v
        )
        
        # Final linear projection
        output = self.W_o(context)
        
        return output, attention_weights

def demonstrate_attention_step_by_step():
    """
    Demonstrate attention mechanism step-by-step with a simple example.
    """
    print("=" * 60)
    print("ATTENTION MECHANISM: STEP-BY-STEP DEMONSTRATION")
    print("=" * 60)
    
    # Create a simple example
    seq_len = 4
    d_model = 8
    d_k = 6
    d_v = 6
    
    print(f"\nInput sequence length: {seq_len}")
    print(f"Model dimension: {d_model}")
    print(f"Key/Query dimension: {d_k}")
    print(f"Value dimension: {d_v}")
    
    # Create input sequence
    x = torch.randn(1, seq_len, d_model)
    print(f"\nInput shape: {x.shape}")
    print(f"Input tensor:\n{x[0]}")
    
    # Initialize attention mechanism
    attention = AttentionMechanism(d_model, d_k, d_v)
    
    # Get Q, K, V projections
    Q = attention.W_q(x)
    K = attention.W_k(x)
    V = attention.W_v(x)
    
    print(f"\n" + "="*40)
    print("STEP 1: LINEAR PROJECTIONS")
    print("="*40)
    print(f"Query shape: {Q.shape}")
    print(f"Key shape: {K.shape}")
    print(f"Value shape: {V.shape}")
    
    # Reshape for attention calculation
    Q = Q.view(1, seq_len, 1, d_k).transpose(1, 2)
    K = K.view(1, seq_len, 1, d_k).transpose(1, 2)
    V = V.view(1, seq_len, 1, d_v).transpose(1, 2)
    
    print(f"\nReshaped Q shape: {Q.shape}")
    print(f"Reshaped K shape: {K.shape}")
    print(f"Reshaped V shape: {V.shape}")
    
    # Step 2: Calculate attention scores
    print(f"\n" + "="*40)
    print("STEP 2: ATTENTION SCORES CALCULATION")
    print("="*40)
    print("Computing QK^T...")
    
    scores = torch.matmul(Q, K.transpose(-2, -1))
    print(f"Raw attention scores shape: {scores.shape}")
    print(f"Raw attention scores:\n{scores[0, 0]}")
    
    # Step 3: Scale the scores
    print(f"\n" + "="*40)
    print("STEP 3: SCALING ATTENTION SCORES")
    print("="*40)
    scale = d_k ** 0.5
    print(f"Scale factor: √{d_k} = {scale:.3f}")
    
    scaled_scores = scores / scale
    print(f"Scaled attention scores:\n{scaled_scores[0, 0]}")
    
    # Step 4: Apply softmax
    print(f"\n" + "="*40)
    print("STEP 4: SOFTMAX NORMALIZATION")
    print("="*40)
    print("Applying softmax to get probability distribution...")
    
    attention_weights = F.softmax(scaled_scores, dim=-1)
    print(f"Attention weights (sum to 1):\n{attention_weights[0, 0]}")
    print(f"Sum of weights for each position:")
    for i in range(seq_len):
        print(f"  Position {i}: {attention_weights[0, 0, i].sum().item():.6f}")
    
    # Step 5: Apply attention to values
    print(f"\n" + "="*40)
    print("STEP 5: WEIGHTED VALUE AGGREGATION")
    print("="*40)
    print("Computing weighted sum of values...")
    
    context = torch.matmul(attention_weights, V)
    print(f"Context vector shape: {context.shape}")
    print(f"Context vectors:\n{context[0, 0]}")
    
    # Show the complete forward pass
    print(f"\n" + "="*40)
    print("COMPLETE FORWARD PASS")
    print("="*40)
    
    output, final_weights = attention.forward(x)
    print(f"Final output shape: {output.shape}")
    print(f"Final attention weights shape: {final_weights.shape}")
    
    return attention_weights[0, 0], x[0]

def visualize_attention_weights(attention_weights: torch.Tensor, input_sequence: torch.Tensor):
    """
    Visualize attention weights to understand how the model attends to different positions.
    """
    print(f"\n" + "="*40)
    print("ATTENTION WEIGHTS VISUALIZATION")
    print("="*40)
    
    # Convert to numpy for plotting
    weights_np = attention_weights.detach().numpy()
    
    # Create a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(weights_np, 
                annot=True, 
                cmap='Blues', 
                xticklabels=[f'Pos {i}' for i in range(weights_np.shape[1])],
                yticklabels=[f'Pos {i}' for i in range(weights_np.shape[0])])
    plt.title('Attention Weights Heatmap\n(How each position attends to other positions)')
    plt.xlabel('Key Positions')
    plt.ylabel('Query Positions')
    plt.tight_layout()
    plt.savefig('attention_weights_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Analyze attention patterns
    print("\nAttention Pattern Analysis:")
    for i in range(weights_np.shape[0]):
        max_attention = weights_np[i].max()
        max_pos = weights_np[i].argmax()
        print(f"  Position {i} pays most attention to position {max_pos} (weight: {max_attention:.3f})")

def demonstrate_multi_head_attention():
    """
    Demonstrate how multi-head attention works and why it's beneficial.
    """
    print(f"\n" + "="*60)
    print("MULTI-HEAD ATTENTION DEMONSTRATION")
    print("="*60)
    
    seq_len = 6
    d_model = 12
    d_k = 4
    d_v = 4
    num_heads = 3
    
    print(f"Sequence length: {seq_len}")
    print(f"Model dimension: {d_model}")
    print(f"Key/Query dimension per head: {d_k}")
    print(f"Value dimension per head: {d_v}")
    print(f"Number of heads: {num_heads}")
    
    # Create input
    x = torch.randn(1, seq_len, d_model)
    
    # Initialize multi-head attention
    multi_head_attention = AttentionMechanism(d_model, d_k, d_v, num_heads)
    
    # Forward pass
    output, attention_weights = multi_head_attention.forward(x)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    # Show attention patterns for each head
    print(f"\nAttention patterns for each head:")
    for head in range(num_heads):
        head_weights = attention_weights[0, head]
        print(f"\nHead {head}:")
        print(head_weights)
        
        # Find the strongest attention for each position
        for pos in range(seq_len):
            max_attn = head_weights[pos].max()
            max_pos = head_weights[pos].argmax()
            print(f"  Position {pos} → Position {max_pos} (weight: {max_attn:.3f})")

def explain_mathematical_intuition():
    """
    Explain the mathematical intuition behind why attention works.
    """
    print(f"\n" + "="*60)
    print("MATHEMATICAL INTUITION: WHY ATTENTION WORKS")
    print("="*60)
    
    print("""
1. QUERY-KEY SIMILARITY:
   - The dot product QK^T measures similarity between queries and keys
   - Higher similarity = higher attention score
   - This allows the model to focus on relevant information

2. SCALING FACTOR (√d_k):
   - Prevents attention scores from becoming too large
   - Large scores make softmax too sharp (approaching one-hot)
   - Scaling maintains reasonable gradients during training

3. SOFTMAX NORMALIZATION:
   - Converts scores to probability distribution
   - Ensures attention weights sum to 1
   - Creates interpretable attention patterns

4. WEIGHTED AGGREGATION:
   - Each position gets a weighted combination of all values
   - Weights are learned based on content similarity
   - Allows dynamic, content-dependent information flow

5. MULTI-HEAD BENEFITS:
   - Different heads can learn different types of relationships
   - Some heads might focus on local patterns, others on global
   - Increases model capacity and expressiveness
    """)

def create_attention_examples():
    """
    Create specific examples to illustrate different attention patterns.
    """
    print(f"\n" + "="*60)
    print("ATTENTION PATTERN EXAMPLES")
    print("="*60)
    
    # Example 1: Local attention (focusing on nearby positions)
    print("\nExample 1: Local Attention Pattern")
    print("This pattern focuses on nearby positions, useful for local dependencies.")
    
    local_weights = torch.tensor([
        [0.7, 0.2, 0.05, 0.05],
        [0.2, 0.6, 0.15, 0.05],
        [0.05, 0.15, 0.6, 0.2],
        [0.05, 0.05, 0.2, 0.7]
    ])
    
    print(f"Local attention weights:\n{local_weights}")
    
    # Example 2: Global attention (equal attention to all positions)
    print("\nExample 2: Global Attention Pattern")
    print("This pattern gives equal attention to all positions, useful for global context.")
    
    global_weights = torch.tensor([
        [0.25, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0.25]
    ])
    
    print(f"Global attention weights:\n{global_weights}")
    
    # Example 3: Causal attention (can only attend to previous positions)
    print("\nExample 3: Causal Attention Pattern")
    print("This pattern only allows attention to previous positions, used in language modeling.")
    
    causal_weights = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0],
        [0.33, 0.33, 0.34, 0.0],
        [0.25, 0.25, 0.25, 0.25]
    ])
    
    print(f"Causal attention weights:\n{causal_weights}")

if __name__ == "__main__":
    print("Welcome to the Attention Mechanism Tutorial!")
    print("This script will teach you the mathematical foundations of attention mechanisms.")
    
    # Run the demonstrations
    attention_weights, input_seq = demonstrate_attention_step_by_step()
    
    # Visualize attention weights
    visualize_attention_weights(attention_weights, input_seq)
    
    # Demonstrate multi-head attention
    demonstrate_multi_head_attention()
    
    # Explain mathematical intuition
    explain_mathematical_intuition()
    
    # Show attention pattern examples
    create_attention_examples()
    
    print(f"\n" + "="*60)
    print("TUTORIAL COMPLETE!")
    print("="*60)
    print("""
Key Takeaways:
1. Attention computes similarity between queries and keys
2. Softmax creates probability distribution over attention weights
3. Scaling prevents numerical instability
4. Multi-head attention allows learning different types of relationships
5. Attention enables dynamic, content-dependent information flow

The attention mechanism works because it allows the model to:
- Focus on relevant information dynamically
- Learn long-range dependencies
- Process variable-length sequences
- Create interpretable attention patterns
    """) 