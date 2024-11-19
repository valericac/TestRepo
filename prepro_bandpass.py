"""Bandpass Filter module for speech processing.

Author: Valerie Hofmann
Date: 11.11.2024
"""

import os
import sys
import librosa
import soundfile as sf
import numpy as np
from scipy import signal


def bandpass(input_file):
    """Apply bandpass filter to speech audio file.
    
    Args:
        input_file (str): Path to input audio file
        
    Returns:
        tuple: Filtered audio signal and sample rate
    """
    # Load audio
    audio, sr = librosa.load(input_file)

    # Define speech filter parameters
    nyquist = sr // 2
    filter_order = 4
    speech_range = [80, 8000]  # Speech frequency range in Hz, speech bannana 
    
    # Apply bandpass filter
    b, a = signal.butter(
        filter_order,
        [f/nyquist for f in speech_range],
        btype='band'
    )
    filtered_audio = signal.filtfilt(b, a, audio)
    
    return filtered_audio, sr


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python prepro_bandpass.py <input_file>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    filtered_audio, sr = bandpass(input_file)
    
    # Save filtered audio back to same file
    sf.write(input_file, filtered_audio, sr)
    print(f"Filtered audio saved to: {input_file}")