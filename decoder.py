"""
TinyTransformerCodec Decoder Module

Converts a list of integer indices back into a raw audio chunk.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from common_modules import (
    OptimizedTransformerBlock, ImprovedVectorQuantizer,
    CHANNELS, BLOCKS, LATENT_DIM, HEADS, KERNEL_SIZE, STRIDES,
    NUM_CODEBOOKS, CODEBOOK_SIZE, TRANSFORMER_BLOCKS
)

class DecoderModule(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, blocks=BLOCKS, heads=HEADS):
        super().__init__()
        self.num_codebooks = NUM_CODEBOOKS
        
        # --- 1. Embedding Layers ---
        # We instantiate the VQ modules, just like the encoder,
        # to ensure the state_dict keys match the checkpoint.
        # We will only use the `.embedding` submodule.
        self.quantizers = nn.ModuleList([
            ImprovedVectorQuantizer(CODEBOOK_SIZE, LATENT_DIM)
            for _ in range(NUM_CODEBOOKS)
        ])

        # --- 2. Latent Transformer ---
        self.transformer = nn.Sequential(*[
            OptimizedTransformerBlock(latent_dim * NUM_CODEBOOKS, heads)
            for _ in range(TRANSFORMER_BLOCKS)
        ])
        
        # --- 3. Post-Transformer Convolution ---
        self.post_transformer = nn.Conv1d(latent_dim * NUM_CODEBOOKS, latent_dim * NUM_CODEBOOKS, 1)

        # --- 4. Decoder Transposed Convolutions ---
        self.decoder_tconvs = nn.ModuleList()
        
        in_c = latent_dim * NUM_CODEBOOKS
        
        # Pre-calculate encoder channels for decoder output shapes
        encoder_channels = []
        for i in range(blocks):
            encoder_channels.append(min(latent_dim, 16 * (2**i)))

        for i in range(blocks):
            idx = blocks - 1 - i
            stride = STRIDES[idx]
            
            # Determine output channels for this transpose conv
            if idx > 0:
                out_c = encoder_channels[idx - 1]
            else:
                out_c = 16 # Final conv block output channel
                
            self.decoder_tconvs.append(
                nn.ConvTranspose1d(in_c, out_c, KERNEL_SIZE, stride, padding=KERNEL_SIZE//2)
            )
            in_c = out_c

        # --- 5. Final Output Convolution ---
        self.post_decoder_final = nn.Conv1d(in_c, CHANNELS, 1)

    @torch.no_grad()
    def forward(self, indices_list, input_length=None):
        """
        Decodes integer indices back into an audio chunk.
        
        Input:
            indices_list (list[torch.Tensor]): 
                A list of integer index tensors.
                Shape: [ (B, T_latent), (B, T_latent) ]
                Example: [ (1, 20), (1, 20) ]
            
            input_length (int, optional):
                The target length of the output audio (e.g., 480).
                Used to trim or pad the final output.
        
        Output:
            torch.Tensor: Reconstructed audio chunk of shape (B, 1, L)
        """
        
        # --- 1. Get Embeddings from Indices ---
        z_q_list = []
        for i, indices in enumerate(indices_list):
            # (B, T_latent) -> (B, T_latent, C)
            quantized = self.quantizers[i].embedding(indices)
            # (B, T_latent, C) -> (B, C, T_latent)
            quantized = quantized.transpose(1, 2)
            z_q_list.append(quantized)

        # (B, C*num_codebooks, T_latent)
        x = torch.cat(z_q_list, dim=1)

        # --- 2. Latent Transformer ---
        x = self.transformer(x)
        x = self.post_transformer(x)

        # --- 3. Decoder ---
        for i, tconv in enumerate(self.decoder_tconvs):
            x = F.gelu(tconv(x))
            # Skip connections are intentionally omitted, matching
            # the training script's validation/streaming logic.

        # --- 4. Final Output ---
        x = torch.tanh(self.post_decoder_final(x))

        # Match input length
        if input_length is not None:
            if x.shape[-1] > input_length:
                x = x[..., :input_length]
            elif x.shape[-1] < input_length:
                x = F.pad(x, (0, input_length - x.shape[-1]))

        return x.view(x.size(0), CHANNELS, -1)
