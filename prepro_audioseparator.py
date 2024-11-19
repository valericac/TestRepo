"""Audio Separator module to separate audio files into vocals and instrumentals.

Author: Valerie Hofmann
Date: 11.11.2024
"""

import os
import sys
import subprocess
import librosa
import pandas as pd


def separate_audio(input_file, output_dir):
    """Separate audio using audio-separator.
    
    Args:
        input_file (str): Path to input audio file
        output_dir (str): Directory for separated audio files
        
    Returns:
        tuple: Paths to the created vocals and instrumental files as tuples
    """
    try:
        # Debug prints
        print(f"\nDEBUG: Input file path: {input_file}")
        print(f"DEBUG: Output directory: {output_dir}")
        
        # Check if input_file is valid
        if not input_file or input_file == '/':
            raise ValueError(f"Invalid input file path: {input_file}")
            
        if not os.path.isfile(input_file):
            raise FileNotFoundError(f"Input file does not exist: {input_file}")

        # Get base filename without extension
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Build the command
        cmd = [
            "audio-separator",
            input_file,
            "--model_filename", "UVR-MDX-NET-Inst_HQ_3.onnx",
            "--output_dir", output_dir,
            "--output_format", "flac"
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        
        # Execute command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=os.environ.copy()
        )
        
        if result.returncode != 0:
            print(f"Error output: {result.stderr}")
            print(f"Standard output: {result.stdout}")
            raise Exception("audio-separator command failed")
            
        # Define output file paths
        vocals_file = os.path.join(
            output_dir,
            f"{base_name}_(Vocals)_UVR-MDX-NET-Inst_HQ_3.flac"
        )
        instrumental_file = os.path.join(
            output_dir,
            f"{base_name}_(Instrumental)_UVR-MDX-NET-Inst_HQ_3.flac"
        )
        
        # Verify files were created
        if not os.path.exists(vocals_file):
            raise FileNotFoundError(f"Vocals file not created: {vocals_file}")
            
        print(f"Audio-separator processing complete. Files created:")
        print(f"- {vocals_file}")
        print(f"- {instrumental_file}")
        
        # Return paths as tuples to match expected format
        return (vocals_file,), (instrumental_file,)
        
    except Exception as e:
        print(f"Separation error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    # This is only for direct testing
    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    separate_audio(input_file, output_dir)