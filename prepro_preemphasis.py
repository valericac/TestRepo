"""Preemphasis module for speech processing.

Author: Valerie Hofmann
Date: 11.11.2024
"""

import os
import librosa
import soundfile as sf
import numpy as np


def preemphasis(input_file):
    """Apply pre-emphasis filter to speech audio file.
    
    Args:
        input_file: Path to input audio file
        
    Returns:
        tuple: Pre-emphasized audio signal and sample rate
    """
    # Load audio
    audio, sr = librosa.load(input_file)

    # Apply pre-emphasis with coefficient 0.60
    pre_emphasized = librosa.effects.preemphasis(audio, coef=0.60)
    
    return pre_emphasized, sr


if __name__ == "__main__":
    input_file = "/mnt/nfs/kcltrauma/01_complete_trauma/cleaned/IS030_trauma_cleaned.wav"
    pre_emphasized, sr = preemphasis(input_file)
    
    # Save audio
    output_dir = "/mnt/nfs/kcltrauma/01_complete_trauma/cleaned"
    os.makedirs(output_dir, exist_ok=True)
    base_name = '_'.join(os.path.basename(input_file).split('_')[:2])
    output_file = os.path.join(output_dir, f'{base_name}_cleaned.wav')
    sf.write(output_file, pre_emphasized, sr)
    print(f"Pre-emphasized audio saved to: {output_file}")