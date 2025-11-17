"""
Example: Overlap-Save (OaS) Streaming with Networking

This script demonstrates how to use the `TinyCodec` API
to perform real-time, low-latency streaming:
1. On a full audio file.
2. From a live microphone (local loopback).
3. From a microphone to a network client (Server Mode).
4. From a network server to a speaker (Client Mode).
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
        "sounddevice": "sounddevice" # Required for mic/speaker modes
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
    
    # --- New Imports for Networking ---
    import socket
    import threading
    import queue
    import pickle
    # --- End New Imports ---
    
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
# --- MOCKUP for testing without the real files ---
# These files are assumed to exist:
# - codec_api.py (with class TinyCodec)
# - common_modules.py (with SR, TRAIN_WINDOW_SAMPLES, STREAMING_HOP_SAMPLES)

if not os.path.exists("codec_api.py") or not os.path.exists("common_modules.py"):
    print("WARNING: 'codec_api.py' or 'common_modules.py' not found.")
    print("Creating mock objects for testing...")
    
    # Mock common_modules.py
    SR = 16000
    TRAIN_WINDOW_SAMPLES = 480
    STREAMING_HOP_SAMPLES = 240
    
    # Mock codec_api.py
    class TinyCodec:
        def __init__(self, model_path=None):
            print(f"MockCodec: Initialized (model: {model_path})")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"MockCodec: Using device: {self.device}")
        
        def encode_audio(self, chunk):
            # Mock: return a list of dummy indices
            # Real codec would do GPU/CPU processing
            time.sleep(0.001) # Simulate some work
            return [torch.randint(0, 10, (5,), device=self.device),
                    torch.randint(0, 10, (5,), device=self.device)]
        
        def decode_audio(self, indices_list, target_length):
            # Mock: return a chunk of audio (e.g., sine wave)
            # Real codec would do GPU/CPU processing
            time.sleep(0.001) # Simulate some work
            # In a real codec, this would be the decoded audio.
            # Here, we'll just pass through a sine wave for testing.
            # We'll actually just return a silent chunk
            return torch.zeros(target_length, device=self.device)
            
            # Note: A pass-through (return chunk) would be better for
            # testing, but the original chunk isn't available here.
            # So we just return silence.
else:
    # Import the main API
    from codec_api import TinyCodec
    
    # Import hyperparameters
    from common_modules import SR, TRAIN_WINDOW_SAMPLES, STREAMING_HOP_SAMPLES
# --- End of Mockup ---


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
    print(f"  Window size: {TRAIN_WINDOW_SAMPLES} samples ({(TRAIN_WINDOW_SAMPLES/SR)*1000:.1f} ms)")
    print(f"  Hop size:    {STREAMING_HOP_SAMPLES} samples ({(STREAMING_HOP_SAMPLES/SR)*1000:.1f} ms)")
    
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
    print(f" - Audio Duration:    {duration:.2f} seconds")
    print(f" - Processing Time: {processing_time:.2f} seconds") 
    print(f" - Real-Time Factor (RTF): {rtf:.4f}")
    if rtf < 1.0:
        print("   (This is faster than real-time!)")
    else:
        print("   (This is slower than real-time. A GPU is recommended.)")

def run_mic_test(codec):
    """
    Runs the demo in mode (2): live streaming from mic to speakers.
    """
    print("\n--- Mode 2: Live Microphone Test (Local Loopback) ---")
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


# ===================================================================
# === NEW NETWORKING FUNCTIONS ======================================
# ===================================================================

# Define a simple message-framing protocol
# We send a 4-byte header (big-endian) with the length
# of the pickled data that follows.
HEADER_LENGTH = 4

def recv_all(sock, n):
    """Helper function to reliably receive n bytes from a socket."""
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def stream_audio_to_client(codec, conn):
    """
    Called by the server.
    Handles the audio-to-network loop for a single client.
    """
    print(f"Streaming audio to client...")
    
    # Queue to pass encoded data from audio thread to network thread
    data_queue = queue.Queue(maxsize=10) # Buffer max 10 packets
    
    # OaS overlap buffer
    overlap_samples = TRAIN_WINDOW_SAMPLES - STREAMING_HOP_SAMPLES
    overlap_buffer = torch.zeros(overlap_samples, device=codec.device)

    def server_audio_callback(indata, frames, time, status):
        """
        Audio callback for the server.
        Reads mic, encodes, and puts data on the queue.
        """
        nonlocal overlap_buffer
        if status:
            print(status, file=sys.stderr)
        
        try:
            # 1. Get audio from mic
            current_chunk = torch.tensor(indata[:, 0], dtype=torch.float32, device=codec.device)
            
            # 2. Create OaS window
            window = torch.cat((overlap_buffer, current_chunk))
            
            # 3. Update overlap buffer
            overlap_buffer = current_chunk
            
            # 4. Encode
            indices_list = codec.encode_audio(window)
            
            # 5. Put encoded data on the queue for the network thread
            # We move to CPU and pickle *before* putting on the queue
            # to offload work from the network thread.
            # But, pickling tensors might be slow.
            # Let's try pickling the GPU tensors directly.
            # Note: The client must be able to unpickle them (e.g., have torch)
            
            # Alternative: move to CPU first
            # indices_list_cpu = [idx.cpu() for idx in indices_list]
            # data_queue.put(indices_list_cpu, block=False)
            
            # Let's assume receiver can handle tensors from any device
            data_queue.put(indices_list, block=False) # Non-blocking
            
        except queue.Full:
            print("Server network queue is full. Dropping packet.", file=sys.stderr)
        except Exception as e:
            print(f"Error in server audio callback: {e}", file=sys.stderr)

    try:
        # Start the microphone input stream
        with sd.InputStream(
            samplerate=SR,
            blocksize=STREAMING_HOP_SAMPLES,
            device=None,
            channels=1,
            dtype='float32',
            callback=server_audio_callback
        ):
            print("Microphone is live. Sending data...")
            while True:
                # 1. Get encoded data from the audio thread
                indices_list = data_queue.get() # Blocks until data is ready
                
                # 2. Serialize the data
                data_bytes = pickle.dumps(indices_list)
                
                # 3. Create header and send
                header = len(data_bytes).to_bytes(HEADER_LENGTH, 'big')
                
                # 4. Send all data
                conn.sendall(header + data_bytes)

    except (socket.error, BrokenPipeError, ConnectionResetError) as e:
        print(f"Client disconnected: {e}")
    except Exception as e:
        print(f"Error in server streaming loop: {e}")
    finally:
        print("Stopping server stream.")

def run_server(codec):
    """
    Runs the demo in mode (3): Start a server (Mic -> Network)
    """
    print("\n--- Mode 3: Start a Server (Mic -> Network) ---")
    
    host = input(f"Enter host IP to bind (default: 0.0.0.0): ").strip() or "0.0.0.0"
    port_str = input(f"Enter port to listen on (default: 12345): ").strip() or "12345"
    
    try:
        port = int(port_str)
        if not 1024 <= port <= 65535:
            raise ValueError
    except ValueError:
        print(f"Invalid port: {port_str}. Must be an integer 1024-65535.")
        return

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            s.listen()
            print(f"Server listening on {host}:{port}...")
            print("Press Ctrl+C to stop the server.")
            
            while True:
                try:
                    conn, addr = s.accept()
                    with conn:
                        print(f"Connected by {addr}")
                        # Handle the full streaming logic for this client
                        stream_audio_to_client(codec, conn)
                    print(f"Connection with {addr} closed.")
                
                except KeyboardInterrupt:
                    print("\nShutting down server...")
                    break
                except Exception as e:
                    print(f"Error accepting connection: {e}")
        
        except KeyboardInterrupt:
            print("\nShutting down server...")
        except OSError as e:
            print(f"FATAL: Could not bind to {host}:{port}. Error: {e}")
            print("Is the port already in use?")

def receive_audio_from_server(codec, sock):
    """
    Called by the client.
    Handles the network-to-audio loop.
    """
    print("Receiving audio from server...")
    
    # Queue to pass decoded audio from network thread to audio thread
    audio_queue = queue.Queue(maxsize=10) # Buffer 10 audio chunks

    # --- Network Receiver Thread ---
    def network_receiver_loop():
        """
        Runs in a separate thread.
        Receives data, unpickles, decodes, and puts audio on the queue.
        """
        try:
            while True:
                # 1. Receive header
                header_bytes = recv_all(sock, HEADER_LENGTH)
                if not header_bytes:
                    print("Server closed connection (header).")
                    break
                
                data_len = int.from_bytes(header_bytes, 'big')
                
                # 2. Receive data payload
                data_bytes = recv_all(sock, data_len)
                if not data_bytes:
                    print("Server closed connection (payload).")
                    break
                    
                # 3. Deserialize
                indices_list = pickle.loads(data_bytes)
                
                # 4. Decode
                reconstructed_chunk = codec.decode_audio(indices_list, TRAIN_WINDOW_SAMPLES)
                
                # 5. Get OaS "save" part
                new_audio = reconstructed_chunk[-STREAMING_HOP_SAMPLES:] # (240,)
                
                # 6. Put audio chunk on the queue for the speaker
                audio_queue.put(new_audio.cpu().numpy())
                
        except (ConnectionResetError, BrokenPipeError, EOFError):
            print("Server disconnected.")
        except pickle.UnpicklingError:
            print("Received corrupted data from server.")
        except Exception as e:
            print(f"Error in network receiver thread: {e}")
        finally:
            # Put a sentinel value to signal the audio thread to stop
            audio_queue.put(None)
            print("Network receiver thread stopped.")

    # --- Audio Player Callback ---
    def client_audio_callback(outdata, frames, time, status):
        """
        Audio callback for the client.
        Pulls decoded audio from the queue and plays it.
        """
        if status:
            print(status, file=sys.stderr)
        
        try:
            # Get audio chunk from the queue
            audio_chunk = audio_queue.get_nowait()
            
            if audio_chunk is None:
                # Sentinel value received
                print("Stopping audio stream (sentinel).")
                raise sd.CallbackStop
            
            # Play the audio
            outdata[:] = audio_chunk.reshape(-1, 1)
            
        except queue.Empty:
            # Buffer underrun
            print("Client buffer underrun. Playing silence.", file=sys.stderr)
            outdata.fill(0) # Play silence

    # --- Start streaming ---
    try:
        # Start the network receiver thread
        net_thread = threading.Thread(target=network_receiver_loop, daemon=True)
        net_thread.start()
        
        # Start the audio output stream
        with sd.OutputStream(
            samplerate=SR,
            blocksize=STREAMING_HOP_SAMPLES,
            device=None,
            channels=1,
            dtype='float32',
            callback=client_audio_callback
        ):
            print("Audio output stream started. Waiting for data...")
            # Keep the main thread alive while the other threads run
            net_thread.join() 
            
    except sd.CallbackStop:
        print("Audio stream stopped.")
    except Exception as e:
        print(f"Error starting client audio stream: {e}")
    finally:
        print("Client streaming finished.")

def run_client(codec):
    """
    Runs the demo in mode (4): Connect to a server (Network -> Speaker)
    """
    print("\n--- Mode 4: Connect to a Server (Network -> Speaker) ---")
    
    host = input(f"Enter server IP to connect to (default: 127.0.0.1): ").strip() or "127.0.0.1"
    port_str = input(f"Enter server port (default: 12345): ").strip() or "12345"
    
    try:
        port = int(port_str)
    except ValueError:
        print(f"Invalid port: {port_str}.")
        return

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            print(f"Connecting to {host}:{port}...")
            s.connect((host, port))
            print("Connected to server!")
            # Handle the full receiving and playing logic
            receive_audio_from_server(codec, s)
            
    except socket.timeout:
        print("Connection timed out.")
    except ConnectionRefusedError:
        print(f"Connection refused. Is the server running at {host}:{port}?")
    except KeyboardInterrupt:
        print("\nDisconnecting...")
    except Exception as e:
        print(f"Error connecting to server: {e}")
    finally:
        print("Disconnected from server.")


# ===================================================================
# === MAIN FUNCTION (Updated) =======================================
# ===================================================================

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
        codec = TinyCodec(model_path=model_path)
        
    except FileNotFoundError as e:
        # This will catch if the download failed for best_model.pth
        print(f"\nERROR: {e}")
        print(f"Please make sure '{model_path}' is in this directory.")
        return
    except Exception as e:
        print(f"\nFATAL: Failed to load codec: {e}")
        print("This might be due to a missing file or an error in the codec/model files.")
        return

    
    # --- 2. Mode Selection ---
    while True:
        print("\nSelect operation mode:")
        print("  (1) Process an audio file")
        print("  (2) Live microphone test (local loopback)")
        print("  (3) Start a server (Mic -> Network)")
        print("  (4) Connect to a server (Network -> Speaker)")
        print("  (Q) Quit")
        
        choice = input("Enter your choice (1, 2, 3, 4, or Q): ").strip().lower()
        
        if choice == '1':
            run_file_processing(codec)
            
        elif choice == '2':
            if not SOUNDDEVICE_AVAILABLE:
                print("\nERROR: 'sounddevice' package is required for this mode.")
                print("Please install it manually: pip install sounddevice")
                continue
            try:
                run_mic_test(codec)
            except Exception as e:
                print(f"\nAn error occurred during the mic test: {e}")
                print("Please check your microphone and speaker settings.")
        
        elif choice == '3':
            if not SOUNDDEVICE_AVAILABLE:
                print("\nERROR: 'sounddevice' package is required for this mode.")
                print("Please install it manually: pip install sounddevice")
                continue
            run_server(codec)
        
        elif choice == '4':
            if not SOUNDDEVICE_AVAILABLE:
                print("\nERROR: 'sounddevice' package is required for this mode.")
                print("Please install it manually: pip install sounddevice")
                continue
            run_client(codec)
        
        elif choice in ('q', 'quit'):
            print("Exiting.")
            break
            
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()