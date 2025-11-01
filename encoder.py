"""
TinyTransformerCodec Encoder Module

Converts a raw audio chunk into a list of integer indices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from common_modules import (
    CausalConvBlock, ImprovedVectorQuantizer,
    CHANNELS, BLOCKS, LATENT_DIM, KERNEL_SIZE, STRIDES,
    NUM_CODEBOOKS, CODEBOOK_SIZE
)

class EncoderModule(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, blocks=BLOCKS):
        super().__init__()
        self.num_codebooks = NUM_CODEBOOKS

        # --- 1. Encoder Convolutions ---
        self.encoder_convs = nn.ModuleList()
        in_c = CHANNELS
        for i in range(blocks):
            out_c = min(latent_dim, 16 * (2**i))
            stride = STRIDES[i]
            self.encoder_convs.append(CausalConvBlock(in_c, out_c, KERNEL_SIZE, stride))
            in_c = out_c

        # --- 2. Pre-Quantization Convolution ---
        self.pre_quant = CausalConvBlock(in_c, LATENT_DIM * NUM_CODEBOOKS, KERNEL_SIZE, 1)

        # --- 3. Vector Quantization ---
        # We build the VQ modules here so the state_dict keys
        # (e.g., `quantizers.0.embedding.weight`) match the checkpoint.
        self.quantizers = nn.ModuleList([
            ImprovedVectorQuantizer(CODEBOOK_SIZE, LATENT_DIM)
            for _ in range(NUM_CODEBOOKS)
        ])
        
        # The Transformer is NOT part of the encoder-only module.
        # It's part of the decoder, which takes indices as input.

    @torch.no_grad()
    def forward(self, x):
        """
        Encodes an audio chunk into integer indices.
        
        Input:
            x (torch.Tensor): Audio chunk of shape (B, 1, L) or (B, L).
                             Example: (1, 1, 480)
        
        Output:
            list[torch.Tensor]: A list of integer index tensors.
                                Shape: [ (B, T_latent), (B, T_latent) ]
                                Example: [ (1, 20), (1, 20) ]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1) # (B, L) -> (B, 1, L)
        
        x = x.view(x.size(0), CHANNELS, -1) # Ensure correct shape

        # --- 1. Encoder ---
        for layer in self.encoder_convs:
            x = layer(x)

        # --- 2. Pre-Quant ---
        z_e = self.pre_quant(x) # (B, C*num_codebooks, T_latent)

        # --- 3. Vector Quantization ---
        indices_list = []
        z_e_split = z_e.chunk(self.num_codebooks, dim=1)

        for i in range(self.num_codebooks):
            # The VQ forward pass returns integer indices (B, T_latent)
            indices = self.quantizers[i](z_e_split[i])
            indices_list.append(indices)

        return indices_list
