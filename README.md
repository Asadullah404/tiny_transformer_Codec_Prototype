TinyTransformerCodec Inference Package

This package contains the inference modules for the TinyTransformerCodec, a low-latency audio codec designed for streaming. It is based on a Causal Convolutional VQ-VAE with a Transformer in the latent space.

The system is split into an encoder and a decoder, which communicate using integer "tokens" (indices), making it ideal for transmission over a network.

Files in this Package

codec_api.py: The main, high-level API. This is the only file you should need to import directly.

encoder.py: The neural encoder module. It converts audio chunks into integer indices.

decoder.py: The neural decoder module. It converts integer indices back into audio chunks.

common_modules.py: Shared components (like model architecture blocks) and all hyperparameters, imported by the encoder and decoder.

streaming_example.py: A complete, runnable example of how to use the API for real-time Overlap-Save (OaS) streaming.

tiny_transformer_best.pt: (You must provide this). This file should contain the pre-trained model weights from your training script.

Quickstart

Install Dependencies:

pip install torch numpy librosa soundfile


(Note: einops is not required for this inference package).

Add Weights:
Place your trained tiny_transformer_best.pt file in the same directory as these scripts.
 OR 
EPOCH 131 Best Weights will be downaloded its self

Run the Streaming Example:

python streaming_example.py


This will:

Load a test audio file.

Simulate streaming it chunk-by-chunk through the encoder and decoder.

Save the output as original_stream.wav and reconstructed_stream.wav.

API Usage (codec_api.py)

The TinyCodec class in codec_api.py is the simplest way to use the model.

from codec_api import TinyCodec
import torch

# 1. Initialize the codec
# This loads both the encoder and decoder and maps them to the device.
codec = TinyCodec(model_path="tiny_transformer_best.pt")

# 2. Prepare audio
# The codec expects audio chunks of 480 samples (30ms)
# Input can be (L,) or (B, L)
audio_chunk = torch.randn(480).to(codec.device)

# 3. Encode Audio
# Encodes the 480-sample chunk into integer indices.
# The output is a list of NumPy arrays, one for each codebook.
# For a batch size of 1, output shape is: [ (1, 20), (1, 20) ]
indices_list = codec.encode_audio(audio_chunk)

# (At this point, you would send `indices_list` over the network)

# 4. Decode Audio
# Decodes the integer indices back into a 480-sample audio chunk.
# The `original_length` is important for trimming padding.
reconstructed_audio = codec.decode_audio(indices_list, original_length=480)

print(reconstructed_audio.shape)
# Output: torch.Size([480])


How It Works: Overlap-Save (OaS) Streaming

This codec is causal, meaning its output for a given sample depends only on past samples. This is what allows for real-time streaming.

However, to get a clean output, we must use an Overlap-Save (OaS) strategy. The model was trained on 480-sample (30ms) windows, but the streaming hop size is 240 samples (15ms).

Here is the process, as implemented in streaming_example.py:

Buffer: We buffer 480 samples (30ms) from the input stream.

Encode: We pass this entire 480-sample chunk to codec.encode_audio().

Decode: We pass the resulting indices to codec.decode_audio().

Save: The model returns a 480-sample chunk. The first 240 samples are "warm-up" (for the causal convolutions and transformer) and are discarded. We keep only the last 240 samples.

Hop: We advance our input stream by 240 samples (15ms) and repeat the process.


This ensures that the model always has 240 samples of "past context" (the overlap) to correctly predict the 240 samples of "new audio" (the save).
