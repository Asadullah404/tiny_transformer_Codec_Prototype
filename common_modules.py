"""
Common Modules and Hyperparameters

Contains the low-level building blocks (CausalConvBlock, etc.)
and all model hyperparameters, shared between the encoder and decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# === 1. HYPERPARAMETERS (from training script) ===
# These *must* match the parameters used during training.
SR = 16000
CHANNELS = 1
LATENT_DIM = 64
BLOCKS = 4
HEADS = 8
KERNEL_SIZE = 3
STRIDES = [2, 2, 2, 3]
DOWN_FACTOR = np.prod(STRIDES) # 24
NUM_CODEBOOKS = 2
CODEBOOK_SIZE = 128
COMMITMENT_COST = 0.25  # Not used in inference, but part of VQ init
TRANSFORMER_BLOCKS = 4

# Streaming-specific Hyperparameters
TRAIN_WINDOW_SAMPLES = int(0.03 * SR) # 480 samples (30ms)
STREAMING_HOP_SAMPLES = int(0.015 * SR) # 240 samples (15ms)


# === 2. MODEL BUILDING BLOCKS ===

class CausalConvBlock(nn.Module):
    """
    Causal Conv1d block from the training script.
    Uses padding on the left to ensure causality.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.padding_amount = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=0)
        self.norm = nn.GroupNorm(1, out_channels)
        self.activation = nn.GELU()
        self.stride = stride

    def forward(self, x):
        # Pad on the left (time dimension)
        x = F.pad(x, (self.padding_amount, 0), mode='constant', value=0)
        x = self.activation(self.norm(self.conv(x)))
        return x

class OptimizedTransformerBlock(nn.Module):
    """
    Optimized transformer block from the training script.
    Includes the 4-module FFN sequence.
    """
    def __init__(self, dim, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0.0) # Dropout is 0 for eval
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(0.0), # Dropout is 0 for eval
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        B, C, T = x.shape
        x_attn = x.transpose(1, 2)
        
        # Causal mask for streaming inference
        attn_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        
        # is_causal=False with attn_mask is the standard way
        attn_output, _ = self.attn(x_attn, x_attn, x_attn, attn_mask=attn_mask, is_causal=False) 
        
        x_attn = self.norm1(x_attn + attn_output)
        ffn_output = self.ffn(x_attn)
        x_attn = self.norm2(x_attn + ffn_output)
        return x_attn.transpose(1, 2)

class ImprovedVectorQuantizer(nn.Module):
    """
    Inference-only Vector Quantization layer.
    
    This module is instantiated by both the Encoder and Decoder
    to ensure the `state_dict` keys (e.g., `quantizers.0.embedding.weight`)
    match the training checkpoint perfectly.
    
    The Encoder uses the `forward` pass to get indices.
    The Decoder *only* uses the `self.embedding` layer.
    """
    def __init__(self, num_embeddings=CODEBOOK_SIZE, embedding_dim=LATENT_DIM, commitment_cost=COMMITMENT_COST):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        
        # This is the only part that stores weights
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)

        # We don't need the EMA buffers from training, as they are not
        # saved in the state_dict (they are buffers, not parameters).
        # The `embedding.weight` is what's trained and saved.

    @torch.no_grad()
    def forward(self, inputs):
        """
        Inference-only forward pass.
        Takes (B, C, T) float tensor, returns (B, T) integer indices.
        """
        # (B, C, T) -> (B, T, C)
        inputs_transposed = inputs.transpose(1, 2)
        B, T, C = inputs_transposed.shape
        
        # (B, T, C) -> (B*T, C)
        flat_input = inputs_transposed.contiguous().view(-1, self.embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embedding.weight.t()))

        # Find the nearest codebook vector index
        # (B*T, 1)
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        # Reshape to (B, T)
        return encoding_indices.view(B, T)
