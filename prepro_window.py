"""Hamming windowing module for speech processing.

Author: Valerie Hofmann
Date: 11.11.2024
"""

import os
import warnings

import librosa
import numpy as np
from scipy import signal
import soundfile as sf

warnings.filterwarnings("ignore")


def process_with_ola(audio, window_size, hop_length, window_type='hamming'):
    """Process audio using overlap-add method.

    Args:
        audio (np.array): Input audio signal
        window_size (int): Size of window in samples
        hop_length (int): Number of samples between windows
        window_type (str): Type of window ('hamming', 'hann', or 'triang')

    Returns:
        np.array: Processed audio signal
    """
    # Initialize output array
    output = np.zeros(len(audio))
    # Initialize normalization array
    norm = np.zeros(len(audio))

    # Get window function
    if window_type == 'hann':
        window = signal.windows.hann(window_size)
    elif window_type == 'hamming':
        window = signal.windows.hamming(window_size)
    else:
        window = signal.windows.triang(window_size)

    # Process each frame
    for i in range(0, len(audio) - window_size, hop_length):
        # Extract frame
        frame = audio[i:i+window_size]
        if len(frame) < window_size:
            break

        # Apply window
        windowed_frame = frame * window

        # Add to output (overlap-add)
        output[i:i+window_size] += windowed_frame
        # Add window weights for normalization
        norm[i:i+window_size] += window

    # Normalize
    mask = norm > 1e-10
    output[mask] /= norm[mask]

    return output


def apply_windowing(input_file):
    """Apply Hamming windowing with fixed window size of 512.

    Args:
        input_file (str): Path to input audio file
        
    Returns:
        tuple: Processed audio signal and sample rate
    """
    # Load full audio
    audio, sr = librosa.load(input_file)

    # Fixed configuration
    window_size = 512
    hop_length = window_size // 4  # 75% overlap
    window_type = 'hamming'

    # Process audio
    processed = process_with_ola(
        audio,
        window_size=window_size,
        hop_length=hop_length,
        window_type=window_type
    )

    return processed, sr


if __name__ == "__main__":
    # Setup paths
    input_file = "/mnt/nfs/kcltrauma/01_complete_trauma/cleaned/IS030_trauma_cleaned.wav"
    output_file = "/mnt/nfs/kcltrauma/01_complete_trauma/cleaned/IS030_trauma_cleaned.wav"

    # Process and save
    processed, sr = apply_windowing(input_file)
    sf.write(output_file, processed, sr)
    print(f"\nProcessed file saved to: {output_file}")
