"""
Main User-Facing API for TinyTransformerCodec

This file provides the `TinyCodec` class, which wraps the
encoder and decoder modules for simple, high-level access.
"""

import torch
import torch.nn as nn
import numpy as np
import os

# Import the inference modules
from encoder import EncoderModule
from decoder import DecoderModule

class TinyCodec:
    """
    A simple API wrapper for the TinyTransformerCodec.
    
    This class loads the pre-trained weights into the separate
    Encoder and Decoder modules and provides simple `encode_audio`
    and `decode_audio` methods.
    """
    def __init__(self, model_path="tiny_transformer_best.pt", device=None):
        """
        Initializes the codec and loads the model weights.
        
        Args:
            model_path (str): Path to the `.pt` checkpoint file.
            device (torch.device, optional): Device to run on.
                                            Auto-detects CUDA if None.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"Loading TinyCodec models to device: {self.device}")

        # 1. Initialize Encoder and Decoder
        self.encoder = EncoderModule().to(self.device).eval()
        self.decoder = DecoderModule().to(self.device).eval()

        # 2. Load the checkpoint
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model checkpoint not found at {model_path}. "
                "Please place 'tiny_transformer_best.pt' in this directory."
            )
            
        # We load the full checkpoint from training
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 3. Load state_dict into *both* modules
        # `strict=False` is crucial here.
        # The encoder will only load encoder weights (e.g., `encoder_convs.*`)
        # and VQ weights (`quantizers.*`). It will ignore decoder weights.
        self.encoder.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        # The decoder will only load decoder weights (e.g., `decoder_tconvs.*`),
        # VQ weights (`quantizers.*`), and transformer weights.
        self.decoder.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        print("Codec models loaded successfully.")

    @torch.no_grad()
    def encode_audio(self, audio_tensor):
        """
        Encodes a chunk of audio.
        
        Args:
            audio_tensor (torch.Tensor): Audio data of shape (L,) or (B, L).
                                         Must be on the same device as the model.
        
        Returns:
            list[np.ndarray]: A list of NumPy arrays containing the integer
                              indices. This is the compressed format.
                              Example shape for B=1: [ (1, 20), (1, 20) ]
        """
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0) # (L,) -> (1, L)
            
        # Ensure tensor is on the correct device
        if audio_tensor.device != self.device:
            audio_tensor = audio_tensor.to(self.device)
            
        # The encoder returns a list of torch tensors
        indices_list_torch = self.encoder(audio_tensor)
        
        # Convert to numpy for serialization/transmission
        indices_list_numpy = [indices.cpu().numpy() for indices in indices_list_torch]
        
        return indices_list_numpy

    @torch.no_grad()
    def decode_audio(self, indices_list, original_length):
        """
        Decodes a list of integer indices back into an audio chunk.
        
        Args:
            indices_list (list[np.ndarray] or list[list[int]]): 
                The compressed indices from `encode_audio`.
            
            original_length (int): 
                The target length of the output audio (e.g., 480).
                This is used to trim padding from the decoder output.
        
        Returns:
            torch.Tensor: The reconstructed audio, shape (L,)
        """
        # Convert numpy arrays or lists back to torch tensors on the correct device
        indices_list_torch = [
            torch.tensor(indices, dtype=torch.long, device=self.device) 
            for indices in indices_list
        ]
        
        # Decode
        reconstructed_audio = self.decoder(indices_list_torch, original_length)
        
        # Squeeze to (L,) for a single audio chunk
        return reconstructed_audio.squeeze()
