"""
Example: Overlap-Save (OaS) Streaming

This script demonstrates how to use the `TinyCodec` API
to perform real-time, low-latency streaming on a full audio file
OR from a live microphone.
"""

# === 1. AUTO-INSTALLER ===
# This block checks for and installs missing packages.
import subprocess
import sys
import os

def install_required_packages():
    """
    Checks for and installs all required packages for this script.
    """
    print("Checking for required packages...")
    # List all dependencies for the entire app
    packages = {
        "torch": "torch",
        "librosa": "librosa",
        "soundfile": "SoundFile", # import soundfile, pip install SoundFile
        "tqdm": "tqdm",
        "gdown": "gdown",
        "numpy": "numpy",
        "sounddevice": "sounddevice" # <-- ADDED for mic test
    }
    
    installed_packages = {}

    for import_name, install_name in packages.items():
        try:
            __import__(import_name)
            print(f"  [✓] {import_name} is already installed.")
            installed_packages[import_name] = True
        except ImportError:
            print(f"  [!] {import_name} not found. Attempting to install '{install_name}'...")
            try:
                # Suppress output for a cleaner install log
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", install_name],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                print(f"      ... Successfully installed '{install_name}'.")
                installed_packages[import_name] = True
            except (subprocess.CalledProcessError, Exception) as e:
                print(f"      ... WARNING: Failed to install '{install_name}'.")
                print(f"      Please try installing it manually: pip install {install_name}")
                print(f"      Error details: {e}")
                installed_packages[import_name] = False
    
    print("Package check complete.\n")
    return installed_packages

# Run the installer *before* other imports
INSTALLED_PACKAGES = install_required_packages()
# === END AUTO-INSTALLER ===


# --- Main Imports ---
# These will now succeed if the installer worked
try:
    import torch
    import torch.nn.functional as F
    import librosa
    import numpy as np
    import soundfile as sf
    import time
    from tqdm import tqdm
    
    # We still need to import gdown to use it
    if INSTALLED_PACKAGES.get("gdown", False):
        import gdown
        GDOWN_AVAILABLE = True
    else:
        GDOWN_AVAILABLE = False
    
    # Check for sounddevice
    if INSTALLED_PACKAGES.get("sounddevice", False):
        import sounddevice as sd
        SOUNDDEVICE_AVAILABLE = True
    else:
        SOUNDDEVICE_AVAILABLE = False
        
except ImportError as e:
    print(f"FATAL ERROR: A required package is missing and could not be installed: {e}")
    print("Please install the packages listed above and try again.")
    sys.exit(1) # Exit if critical components are missing


# --- Tkinter Check (File Dialog) ---
try:
    import tkinter as tk
    from tkinter import filedialog
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
    print("Tkinter not found (this is common on servers).")
    print("Will fall back to a default audio file for file mode.")
# --- End of Imports ---


# Import the main API
from codec_api import TinyCodec

# Import hyperparameters
from common_modules import SR, TRAIN_WINDOW_SAMPLES, STREAMING_HOP_SAMPLES

def download_models():
    """
    Checks for all model files from the user's links
    and downloads them if they are missing.
    """
    models_to_download = {
        "best_model.pth": "https://drive.google.com/uc?export=download&id=1IOggsjQ-AtmBGUhXVrrmgerb45zGHxFk",
        "latest_model.pth": "https://drive.google.com/uc?export=download&id=1ru7ixDdkZiEDgQy5ss8s8gSoPupAxZj-",
        "tiny_transformer_best.pt": "https://drive.google.com/uc?export=download&id=16ChGzs6MR8PcGHYmFKkW_WsaWEUjegsn"
    }

    if not GDOWN_AVAILABLE:
        print("ERROR: 'gdown' is not installed or failed to install. Cannot auto-download models.")
        print("Please run: pip install gdown")
        print("Or manually download the following files:")
        for name, url in models_to_download.items():
            # Show the user-friendly link
            print(f"  - {name} from {url.replace('uc?export=download&id=', 'file/d/')}")
        return False # Indicate failure

    all_present_or_downloaded = True
    for name, url in models_to_download.items():
        if not os.path.exists(name):
            print(f"Model file '{name}' not found. Attempting to download...")
            try:
                gdown.download(url, name, quiet=False)
                print(f"'{name}' downloaded successfully.")
            except Exception as e:
                print(f"ERROR: Failed to download {name}: {e}")
                print(f"Please manually download the file from: {url.replace('uc?export=download&id=', 'file/d/')}")
                all_present_or_downloaded = False # Mark as failed
        else:
            print(f"Model file '{name}' already present.")
            
    return all_present_or_downloaded

def run_file_processing(codec):
    """
    Runs the demo in mode (1): processing a local audio file.
    """
    print("\n--- Mode 1: Process Audio File ---")
    
    # --- 2. Load Test Audio ---
    print(f"Loading test audio... (Target SR: {SR}Hz)")
    
    audio_file_path = None
    if TKINTER_AVAILABLE:
        try:
            print("Opening file explorer to select an audio file...")
            root = tk.Tk()
            root.withdraw() # Hide the main window
            audio_file_path = filedialog.askopenfilename(
                title="Select an audio file for processing",
                filetypes=[
                    ("Audio Files", "*.wav *.mp3 *.flac"),
                    ("All Files", "*.*")
                ]
            )
            root.destroy()
        except Exception as e:
            print(f"Could not open file dialog: {e}. Will use fallback.")
            
    if not audio_file_path:
        if TKINTER_AVAILABLE:
            print("No file selected. Using librosa example audio as a fallback.")
        try:
            # Fallback to librosa example if dialog is cancelled or fails
            original_wav, sr = librosa.load(librosa.example('libri1'), sr=SR, mono=True)
            original_wav_torch = torch.tensor(original_wav, device=codec.device)
            print(f"Loaded librosa example audio: {original_wav.shape[0] / SR:.2f} seconds")
        except Exception as e:
            print(f"FATAL: Failed to load librosa example file: {e}")
            print("Please ensure you have an internet connection for the fallback,")
            print("or select a local file.")
            return
    else:
        try:
            # Load the file selected by the user
            print(f"Loading user-selected file: {audio_file_path}")
            original_wav, sr = librosa.load(audio_file_path, sr=SR, mono=True)
            original_wav_torch = torch.tensor(original_wav, device=codec.device)
            print(f"Loaded audio: {original_wav.shape[0] / SR:.2f} seconds")
        except Exception as e:
            print(f"Failed to load selected file: {e}")
            return

    # --- 3. Run Overlap-Save (OaS) Streaming ---
    print(f"Running streaming simulation...")
    print(f"  Window size: {TRAIN_WINDOW_SAMPLES} samples (30.0 ms)")
    print(f"  Hop size:    {STREAMING_HOP_SAMPLES} samples (15.0 ms)")
    
    reconstructed_chunks = []
    
    start_time = time.time()
    
    # Iterate over the audio with a hop of 240 samples
    for i in tqdm(range(0, original_wav_torch.shape[0], STREAMING_HOP_SAMPLES)):
        
        # a. Get a window of 480 samples
        chunk = original_wav_torch[i : i + TRAIN_WINDOW_SAMPLES]

        # b. Pad the final chunk if it's too short
        current_length = chunk.shape[0]
        if current_length < TRAIN_WINDOW_SAMPLES:
            chunk = F.pad(chunk, (0, TRAIN_WINDOW_SAMPLES - current_length), 'constant', 0)
        
        # c. ENCODE (Audio -> Indices)
        # Input shape: (480,)
        indices_list = codec.encode_audio(chunk)
        
        # (Simulate network transmission: `indices_list` is what you'd send)
        
        # d. DECODE (Indices -> Audio)
        # We tell the decoder the *target* length is 480
        reconstructed_chunk = codec.decode_audio(indices_list, TRAIN_WINDOW_SAMPLES)
        
        # e. OVERLAP-SAVE
        # We *discard* the first 240 samples (the "overlap")
        # and *keep* only the last 240 samples (the "save").
        new_audio = reconstructed_chunk[-STREAMING_HOP_SAMPLES:]
        
        # Store the valid, new audio chunk
        reconstructed_chunks.append(new_audio.cpu().numpy())

    end_time = time.time()
    
    # --- 4. Save Results ---
    print("\nStreaming complete.")
    
    # Stitch all the 240-sample chunks back together
    reconstructed_wav = np.concatenate(reconstructed_chunks)
    
    # Trim to match original length (streaming can add extra padding at the end)
    reconstructed_wav = reconstructed_wav[:original_wav.shape[0]]

    # Save to disk
    sf.write("original_stream.wav", original_wav, SR)
    sf.write("reconstructed_stream.wav", reconstructed_wav, SR)

    # --- 5. Report Metrics ---
    duration = original_wav.shape[0] / SR
    processing_time = end_time - start_time
    rtf = processing_time / duration
    
    print(f"Saved 'original_stream.wav' and 'reconstructed_stream.wav'")
    print(f" - Audio Duration:   {duration:.2f} seconds")
    # --- THIS IS THE FIX ---
    print(f" - Processing Time: {processing_time:.2f} seconds") 
    # --- END OF FIX ---
    print(f" - Real-Time Factor (RTF): {rtf:.4f}")
    if rtf < 1.0:
        print("   (This is faster than real-time!)")
    else:
        print("   (This is slower than real-time. A GPU is recommended.)")

def run_mic_test(codec):
    """
    Runs the demo in mode (2): live streaming from mic to speakers.
    """
    print("\n--- Mode 2: Live Microphone Test ---")
    print("Press ENTER to stop the stream.")
    
    # The "overlap" is the part of the *previous* window we need
    # to build the *next* window.
    # Window (480) = Overlap (240) + New Chunk (240)
    overlap_samples = TRAIN_WINDOW_SAMPLES - STREAMING_HOP_SAMPLES
    overlap_buffer = torch.zeros(overlap_samples, device=codec.device)

    def audio_callback(indata, outdata, frames, time, status):
        """
        This is the core of the real-time processing.
        It's called by the sounddevice library every time
        it has a new block of audio (indata) from the mic
        and needs a block to send to the speaker (outdata).
        """
        nonlocal overlap_buffer
        if status:
            print(status, file=sys.stderr)
        
        # 1. Get audio from mic
        # indata is (240, 1) numpy array, convert to (240,) tensor
        current_chunk = torch.tensor(indata[:, 0], dtype=torch.float32, device=codec.device)
        
        # --- OaS Logic ---
        # 2. Create the 480-sample window
        #    (240 samples from last time, 240 new samples)
        window = torch.cat((overlap_buffer, current_chunk)) # (480,)
        
        # 3. Update overlap_buffer *for the next call*
        # The new chunk we just got becomes the overlap for the next window
        overlap_buffer = current_chunk
        
        # 4. Encode/Decode (The Codec Brain)
        indices_list = codec.encode_audio(window)
        reconstructed_chunk = codec.decode_audio(indices_list, TRAIN_WINDOW_SAMPLES)
        
        # 5. Get the "save" part (the valid 240 samples)
        new_audio = reconstructed_chunk[-STREAMING_HOP_SAMPLES:] # (240,)
        
        # 6. Send to speaker
        # outdata needs shape (240, 1)
        outdata[:] = new_audio.cpu().numpy().reshape(-1, 1)

    try:
        # We set the blocksize to our HOP size. This is the key
        # to making the OaS logic simple and efficient.
        with sd.Stream(
            samplerate=SR,
            blocksize=STREAMING_HOP_SAMPLES, # Process 240 samples at a time
            device=None, # Default device
            channels=1, # Mono
            dtype='float32',
            callback=audio_callback
        ):
            print(f"Stream active. (Blocksize: {STREAMING_HOP_SAMPLES} samples)")
            print("Say something into your mic... (you should hear it back)")
            input() # Wait for user to press Enter

    except Exception as e:
        print(f"\nFATAL: Error starting audio stream: {e}")
        print("Do you have a microphone and speaker connected?")
        print("On some systems (like Linux), you may need to install 'portaudio'.")
        return
        
    print("Stream stopped.")


def main():
    print("--- TinyTransformerCodec Streaming Example ---")
    
    # --- 1. Download and Load Codec ---
    print("Checking for model files...")
    if not download_models():
        if not GDOWN_AVAILABLE:
            print("\nExiting. Please install 'gdown' to proceed.")
            return
        else:
            print("\nWarning: One or more models failed to download.")
            print("The script will try to continue, but may fail if 'best_model.pth' is missing.")
            
    
    model_path = "best_model.pth" # We will still use the 'best' model by default
    
    print(f"\nLoading codec using '{model_path}'...")
    try:
        codec = TinyCodec(model_path=model_path) # <-- Use this one
        
    except FileNotFoundError as e:
        # This will catch if the download failed for best_model.pth
        print(f"\nERROR: {e}")
        print(f"Please make sure '{model_path}' is in this directory.")
        return
    
    # --- 2. Mode Selection ---
    print("\nSelect operation mode:")
    print("  (1) Process an audio file (default)")
    print("  (2) Start live microphone test")
    
    choice = input("Enter your choice (1 or 2): ").strip()
    
    if choice == '2':
        if not SOUNDDEVICE_AVAILABLE:
            print("\nERROR: 'sounddevice' package is required for mic test.")
            print("The installer tried but failed.")
            print("Please install it manually: pip install sounddevice")
            return
        try:
            run_mic_test(codec)
        except Exception as e:
            print(f"\nAn error occurred during the mic test: {e}")
            print("Please check your microphone and speaker settings.")
    else:
        if choice != '1':
            print("Invalid choice. Defaulting to (1) Process audio file.")
        run_file_processing(codec)


if __name__ == "__main__":
    main()

