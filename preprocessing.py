"""Main preprocessing script for audio files.

This script orchestrates the preprocessing pipeline for audio files,
running them through multiple processing stages in sequence.

Author: Valerie Hofmann
Date: 11.11.2024
"""

import os
import glob
import librosa
import soundfile as sf
import logging
from pathlib import Path
from typing import Tuple, List

# Import all preprocessing components
from .prepro_noisegate_equalizer import dynamic_noise_reduction
from .prepro_bandpass import apply_bandpass_filter 
from .prepro_preemphasis import apply_preemphasis
from .prepro_window import apply_windowing
from .prepro_audioseparator import separate_audio


def setup_logging() -> None:
    """Configure logging for the preprocessing pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('preprocessing.log')
        ]
    )


def process_single_file(wav_file: Path, output_path: Path) -> bool:
    """Process a single audio file through the complete pipeline.
    
    Args:
        wav_file: Path to input wav file
        output_path: Path to save processed file
        
    Returns:
        bool: True if processing successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Load audio
        audio, sr = librosa.load(str(wav_file), sr=None)
        
        # 1. Separate vocals from background
        vocals, _ = separate_audio(audio, sr)
        
        # 2. Apply noise reduction
        audio_denoised = dynamic_noise_reduction(vocals, sr)
        
        # 3. Apply bandpass filter
        audio_filtered = apply_bandpass_filter(audio_denoised, sr)
        
        # 4. Apply pre-emphasis
        audio_emphasized = apply_preemphasis(audio_filtered)
        
        # 5. Apply windowing
        audio_windowed = apply_windowing(audio_emphasized)
        
        # Save processed audio
        sf.write(str(output_path), audio_windowed, sr)
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing {wav_file}: {str(e)}")
        return False


def process_audio_files(input_dir: str, output_dir: str) -> None:
    """Process multiple audio files through preprocessing pipeline.
    
    Args:
        input_dir: Directory containing input .wav files
        output_dir: Directory to save processed files
        
    Raises:
        FileNotFoundError: If input directory doesn't exist
        PermissionError: If lacking write permissions for output directory
    """
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Validate directories
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory {input_dir} does not exist")
        
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get list of first 5 IS*.wav files
    wav_files = sorted(list(input_path.glob("IS0*.wav")))[:5]
    
    for wav_file in wav_files:
        logger.info(f"Processing {wav_file}...")
        
        # Get filename without path and extension
        base_name = '_'.join(wav_file.stem.split('_')[:2])  # Take first two parts of name
        cleaned_name = f"{base_name}_cleaned.wav"
        output_file = output_path / cleaned_name
        
        if process_single_file(wav_file, output_file):
            logger.info(f"Successfully processed {wav_file}")
            logger.info(f"Saved cleaned file to: {output_file}")
        else:
            logger.error(f"Failed to process {wav_file}")


def main() -> None:
    """Run the main preprocessing pipeline."""
    input_dir = "/mnt/nfs/kcltrauma/01_complete_trauma"
    output_dir = "/mnt/nfs/kcltrauma/01_complete_trauma/cleaned"
    
    try:
        process_audio_files(input_dir, output_dir)
        logging.info("All processing complete!")
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")


if __name__ == "__main__":
    main()
