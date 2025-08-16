"""
Tiny Simple Language Model (TSLM) - A Minimal Decoder-Only Transformer
======================================================================

This script demonstrates a complete decoder-based transformer language model
using Multi-Head Attention (MHA) with 26 English characters as tokens.

Key Features:
1. Multi-Head Self-Attention
2. Feed-Forward Networks
3. Layer Normalization
4. Positional Encoding
5. Character-level tokenization
6. Autoregressive text generation

The model is intentionally small to demonstrate the core concepts clearly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional

# Set random seed for reproducibility
torch.manual_seed(42)

class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention mechanism for decoder-only transformer."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_k ** 0.5
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply causal mask for decoder-only model
        if mask is None:
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            mask = mask.to(x.device)
        
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), -1e9)
        
        # Apply softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Reshape back
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Final linear projection
        output = self.W_o(context)
        return output

class FeedForward(nn.Module):
    """Feed-forward network with two linear transformations and ReLU activation."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class DecoderBlock(nn.Module):
    """Single decoder block with self-attention and feed-forward network."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output = self.self_attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class PositionalEncoding(nn.Module):
    """Positional encoding to give the model information about token positions."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]

class TSLM(nn.Module):
    """Tiny Simple Language Model - A minimal decoder-only transformer."""
    
    def __init__(self, vocab_size: int, d_model: int = 128, num_heads: int = 8, 
                 num_layers: int = 6, d_ff: int = 512, max_len: int = 100, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm and output projection
        self.final_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        
        # Token embeddings
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        x = self.dropout(x)
        
        # Pass through decoder blocks
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x)
            
        # Final layer norm and output projection
        x = self.final_norm(x)
        logits = self.output_projection(x)
        
        return logits

class CharacterTokenizer:
    """Simple character-level tokenizer for 26 English letters."""
    
    def __init__(self):
        # Create vocabulary with 26 English letters + special tokens
        self.chars = ['<PAD>', '<UNK>', '<START>', '<END>'] + list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        self.char_to_idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        
    def encode(self, text: str) -> List[int]:
        """Convert text to token indices."""
        return [self.char_to_idx.get(char.upper(), self.char_to_idx['<UNK>']) 
                for char in text]
    
    def decode(self, indices: List[int]) -> str:
        """Convert token indices back to text."""
        return ''.join([self.idx_to_char.get(idx, '<UNK>') for idx in indices])

def generate_text(model: TSLM, tokenizer: CharacterTokenizer, 
                 prompt: str, max_length: int = 50, temperature: float = 1.0) -> str:
    """Generate text using the trained model."""
    model.eval()
    
    # Encode the prompt
    prompt_tokens = tokenizer.encode(prompt)
    if not prompt_tokens:
        prompt_tokens = [tokenizer.char_to_idx['<START>']]
    
    # Convert to tensor
    input_tensor = torch.tensor([prompt_tokens], dtype=torch.long)
    
    generated_tokens = prompt_tokens.copy()
    
    with torch.no_grad():
        for _ in range(max_length - len(prompt_tokens)):
            # Get model predictions
            logits = model(input_tensor)
            next_token_logits = logits[0, -1, :] / temperature
            
            # Apply softmax and sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            # Add to generated sequence
            generated_tokens.append(next_token)
            input_tensor = torch.tensor([generated_tokens], dtype=torch.long)
            
            # Stop if we generate an end token
            if next_token == tokenizer.char_to_idx['<END>']:
                break
    
    # Decode and return
    return tokenizer.decode(generated_tokens)

def train_model(model: TSLM, tokenizer: CharacterTokenizer, 
                training_data: List[str], epochs: int = 10, 
                learning_rate: float = 0.001) -> None:
    """Simple training loop for the model."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    print("Training the model...")
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        for text in training_data:
            # Prepare input and target
            tokens = tokenizer.encode(text)
            if len(tokens) < 2:
                continue
                
            # Create sequences for training
            input_tokens = tokens[:-1]
            target_tokens = tokens[1:]
            
            # Convert to tensors
            input_tensor = torch.tensor([input_tokens], dtype=torch.long)
            target_tensor = torch.tensor([target_tokens], dtype=torch.long)
            
            # Forward pass
            logits = model(input_tensor)
            loss = criterion(logits.view(-1, logits.size(-1)), target_tensor.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
    
    print("Training completed!")

def count_parameters(model: TSLM) -> dict:
    """Count parameters for each component of the TSLM model."""
    param_counts = {}
    
    # Token embedding
    param_counts['token_embedding'] = sum(p.numel() for p in model.token_embedding.parameters())
    
    # Positional encoding (no learnable parameters)
    param_counts['positional_encoding'] = 0
    
    # Decoder blocks
    decoder_params = 0
    for i, block in enumerate(model.decoder_blocks):
        block_params = 0
        
        # Self-attention parameters
        attn_params = sum(p.numel() for p in block.self_attention.parameters())
        block_params += attn_params
        
        # Feed-forward parameters
        ff_params = sum(p.numel() for p in block.feed_forward.parameters())
        block_params += ff_params
        
        # Layer normalization parameters
        norm1_params = sum(p.numel() for p in block.norm1.parameters())
        norm2_params = sum(p.numel() for p in block.norm2.parameters())
        block_params += norm1_params + norm2_params
        
        param_counts[f'decoder_block_{i+1}'] = block_params
        decoder_params += block_params
    
    param_counts['decoder_blocks_total'] = decoder_params
    
    # Final layer norm
    param_counts['final_norm'] = sum(p.numel() for p in model.final_norm.parameters())
    
    # Output projection
    param_counts['output_projection'] = sum(p.numel() for p in model.output_projection.parameters())
    
    # Total parameters
    param_counts['total'] = sum(param_counts.values())
    
    return param_counts

def print_parameter_breakdown(model: TSLM):
    """Print detailed parameter breakdown for the TSLM model."""
    print("=== Detailed Parameter Count ===")
    
    # Get parameter counts
    param_counts = count_parameters(model)
    
    # Print breakdown
    print(f"Token Embedding: {param_counts['token_embedding']:,} parameters")
    print(f"Positional Encoding: {param_counts['positional_encoding']:,} parameters (fixed)")
    
    print(f"\nDecoder Blocks:")
    for i in range(len(model.decoder_blocks)):
        block_name = f'decoder_block_{i+1}'
        if block_name in param_counts:
            print(f"  Block {i+1}: {param_counts[block_name]:,} parameters")
    
    print(f"Final Layer Norm: {param_counts['final_norm']:,} parameters")
    print(f"Output Projection: {param_counts['output_projection']:,} parameters")
    
    print(f"\nTotal Parameters: {param_counts['total']:,}")
    
    # Calculate trainable vs non-trainable
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-trainable Parameters: {non_trainable_params:,}")
    
    # Memory usage estimation (assuming float32)
    memory_bytes = param_counts['total'] * 4  # 4 bytes per float32
    memory_mb = memory_bytes / (1024 * 1024)
    
    print(f"Estimated Memory Usage: {memory_mb:.2f} MB (float32)")
    print()

def get_exact_parameter_count(model: TSLM) -> int:
    """Get the exact total parameter count for the TSLM model."""
    return sum(p.numel() for p in model.parameters())

def get_parameter_summary(model: TSLM) -> dict:
    """Get a summary of parameter counts for the TSLM model."""
    param_counts = count_parameters(model)
    
    # Add trainable vs non-trainable counts
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    
    summary = {
        'total_parameters': param_counts['total'],
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': non_trainable_params,
        'token_embedding': param_counts['token_embedding'],
        'decoder_blocks_total': param_counts['decoder_blocks_total'],
        'final_norm': param_counts['final_norm'],
        'output_projection': param_counts['output_projection'],
        'memory_mb': (param_counts['total'] * 4) / (1024 * 1024)  # float32 assumption
    }
    
    return summary

def main():
    """Main function to demonstrate the TSLM."""
    print("=== Tiny Simple Language Model (TSLM) Demo ===\n")
    
    # Initialize tokenizer
    tokenizer = CharacterTokenizer()
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Characters: {tokenizer.chars}\n")
    
    # Initialize model
    model = TSLM(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        num_heads=8,
        num_layers=6,
        d_ff=512,
        max_len=100
    )
    
    # Print detailed parameter breakdown
    print_parameter_breakdown(model)
    
    print(f"Model architecture:")
    print(f"  - Embedding dimension: {model.d_model}")
    print(f"  - Number of heads: {model.decoder_blocks[0].self_attention.num_heads}")
    print(f"  - Number of layers: {len(model.decoder_blocks)}")
    print(f"  - Feed-forward dimension: {model.decoder_blocks[0].feed_forward.linear1.out_features}\n")
    
    # Sample training data (simple English words and phrases)
    training_data = [
        "HELLO WORLD",
        "PYTHON PROGRAMMING",
        "MACHINE LEARNING",
        "ARTIFICIAL INTELLIGENCE",
        "DEEP LEARNING",
        "NEURAL NETWORKS",
        "TRANSFORMER MODELS",
        "ATTENTION MECHANISM",
        "NATURAL LANGUAGE PROCESSING",
        "COMPUTER VISION",
        "DATA SCIENCE",
        "ALGORITHMS",
        "OPTIMIZATION",
        "STATISTICS",
        "MATHEMATICS"
    ]
    
    print("Training data samples:")
    for i, text in enumerate(training_data[:5]):
        print(f"  {i+1}. {text}")
    print("  ... and more\n")
    
    # Train the model
    train_model(model, tokenizer, training_data, epochs=1500, learning_rate=0.001)
    
    # Demonstrate text generation
    print("\n=== Text Generation Demo ===\n")
    
    test_prompts = ["HELLO", "PYTHON", "MACHINE", "DEEP", "NEURAL"]
    
    for prompt in test_prompts:
        generated = generate_text(model, tokenizer, prompt, max_length=20, temperature=0.8)
        print(f"Prompt: '{prompt}'")
        print(f"Generated: '{generated}'")
        print()
    
    # Interactive generation
    print("=== Interactive Generation ===")
    print("Type 'quit' to exit\n")
    
    while True:
        try:
            user_input = input("Enter a prompt (or 'quit'): ").strip().upper()
            if user_input == 'QUIT':
                break
            if user_input:
                generated = generate_text(model, tokenizer, user_input, max_length=30, temperature=0.7)
                print(f"Generated: {generated}\n")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}\n")
    
    print("Demo completed!")

# Example usage of parameter counting functions
def demo_parameter_counting():
    """Demonstrate the parameter counting functions."""
    print("=== Parameter Counting Demo ===\n")
    
    # Create a small model for demonstration
    tokenizer = CharacterTokenizer()
    model = TSLM(
        vocab_size=tokenizer.vocab_size,
        d_model=64,  # Smaller for demo
        num_heads=4,
        num_layers=2,
        d_ff=256,
        max_len=50
    )
    
    # Get exact parameter count
    exact_count = get_exact_parameter_count(model)
    print(f"Exact parameter count: {exact_count:,}")
    
    # Get detailed breakdown
    print("\nDetailed breakdown:")
    print_parameter_breakdown(model)
    
    # Get summary
    summary = get_parameter_summary(model)
    print("Parameter Summary:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value:,}")
    
    return model, summary

# Uncomment the line below to run the parameter counting demo
# demo_parameter_counting()

if __name__ == "__main__":
    demo_parameter_counting()
    main()

