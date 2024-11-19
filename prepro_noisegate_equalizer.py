"""Dynamic noise reduction module for speech processing.

Author: Valerie Hofmann
Date: 15.11.2024
"""

import numpy as np
import librosa
import soundfile as sf
import scipy.signal as signal
import copy
from numpy import (array, argmin, argmax, arange, floor, mod, zeros,
                  median, log10, int16)


def dynamic_noise_reduction(audio, sr, chunk_duration=0.025,
                          overlap_ratio=0.5, threshold_ratio=2.0):
    """Apply dynamic noise reduction based on RMS energy and spectral analysis.

    Parameters:
        audio: np.array
            Audio signal
        sr: int
            Sample rate
        chunk_duration: float
            Analysis window size in seconds (default 25ms)
        overlap_ratio: float
            Overlap between windows (0-1)
        threshold_ratio: float
            How many times above median RMS to set threshold
            
    Returns:
        np.array: Processed audio with noise reduction applied
    """
    # Calculate chunk sizes
    chunk_size = int(chunk_duration * sr)
    hop_size = int(chunk_size * (1 - overlap_ratio))

    # Frame the signal
    frames = librosa.util.frame(audio, frame_length=chunk_size,
                               hop_length=hop_size)

    # Initialize output
    processed_audio = np.zeros_like(audio)
    gain_envelope = np.zeros_like(audio)

    for i in range(frames.shape[1]):
        chunk = frames[:, i]

        # Calculate spectral features
        faxis, ps = signal.periodogram(chunk, fs=sr, window=('kaiser', 38))

        # Find fundamental frequency
        fund_bin = np.argmax(ps)
        fund_indices = get_indices_around_peak(ps, fund_bin)

        # Calculate noise profile
        noise_prepared = copy.copy(ps)
        noise_prepared[fund_indices] = 0
        noise_mean = np.median(noise_prepared[noise_prepared != 0])

        # Calculate RMS for this chunk
        chunk_rms = np.sqrt(np.mean(chunk**2))

        # Dynamic threshold based on noise floor
        noise_floor = np.sqrt(noise_mean)
        threshold = noise_floor * threshold_ratio

        # Calculate gain
        if chunk_rms > threshold:
            gain = 1.0
        else:
            gain = (chunk_rms / threshold) ** 2

        # Apply gain to chunk
        processed_chunk = chunk * gain

        # Add to output with overlap-add
        start_idx = i * hop_size
        end_idx = start_idx + chunk_size
        processed_audio[start_idx:end_idx] += processed_chunk * np.hanning(chunk_size)
        gain_envelope[start_idx:end_idx] += np.hanning(chunk_size)

    # Normalize for overlap-add
    processed_audio = processed_audio / np.maximum(gain_envelope, 1e-8)

    return processed_audio


def get_indices_around_peak(arr, peak_index, search_width=1000):
    """Get indices around a peak in an array.

    Args:
        arr: array to search
        peak_index: index of peak
        search_width: how far to search around peak

    Returns:
        array of indices around peak
    """
    peak_bins = []
    mag_max = arr[peak_index]
    cur_val = mag_max

    for i in range(search_width):
        new_bin = peak_index + i
        if new_bin >= len(arr):
            break
        new_val = arr[new_bin]
        if new_val > cur_val:
            break
        else:
            peak_bins.append(int(new_bin))
            cur_val = new_val

    cur_val = mag_max
    for i in range(search_width):
        new_bin = peak_index - i
        if new_bin < 0:
            break
        new_val = arr[new_bin]
        if new_val > cur_val:
            break
        else:
            peak_bins.append(int(new_bin))
            cur_val = new_val

    return array(list(set(peak_bins)))


def process_file(input_file):
    """Process an audio file with dynamic noise reduction.

    Args:
        input_file: path to input audio file

    Returns:
        tuple: (processed_audio, sample_rate)
    """
    # Load audio
    audio, sr = librosa.load(input_file, sr=None)

    # Apply noise reduction
    processed_audio = dynamic_noise_reduction(
        audio,
        sr,
        chunk_duration=0.025,  # 25ms windows
        overlap_ratio=0.5,     # 50% overlap
        threshold_ratio=2.0    # Adjust based on your needs
    )

    return processed_audio, sr


if __name__ == "__main__":
    input_file = "/mnt/nfs/kcltrauma/01_complete_trauma/cleaned/IS030_trauma_cleaned.wav"
    output_file = input_file.replace('.wav', '_denoised.wav')
    
    processed_audio, sr = process_file(input_file)
    sf.write(output_file, processed_audio, sr)
    print("Processing complete!")