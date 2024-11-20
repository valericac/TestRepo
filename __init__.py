"""Audio preprocessing package for speech analysis.

This __init__.py file marks this directory as a Python package, allowing its
modules to be imported. It provides:

1. Package Documentation: Overview of what this package does
2. Version Information: Current package version
3. Public Interface: What functions/classes are available to users
4. Imports: Makes key functionality available at package level

The package provides tools for processing audio files with focus on speech analysis,
including noise reduction, filtering, and feature extraction.

Author: Valerie
Date: 13.11.2024
"""

# Version info
__version__ = '0.1.0'

# Import main functionality to make available at package level
from .preprocessing import process_audio_files
from .prepro_noisegate_equalizer import dynamic_noise_reduction

# Define public API
__all__ = [
    'process_audio_files',
    'dynamic_noise_reduction'
]



#Creating an error
