#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example script to generate overlap enrollment data for ESPnet target speaker ASR training

Usage:
    python run_overlap_enrollment_espnet.py
    
This script demonstrates how to create overlap enrollment data that can be directly
used for ESPnet target speaker ASR training.

Copyright 2024  Northwestern Polytechnical University (author: Pengcheng Guo)
Apache 2.0
"""

import os
import sys
import logging
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from create_overlap_enrollment import LibriSpeechDataLoader, OverlapEnrollmentGenerator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    # Configuration
    config = {
        # Input data directories
        'librispeech_data_dir': '/path/to/librispeech/train-clean-100',  # Change this
        'enrollment_data_dir': '/path/to/librispeech/dev-clean',        # Change this
        
        # Output directory
        'output_dir': './data/overlap_enrollment_train',
        
        # Generation parameters
        'num_mixtures': 10000,
        'sir_min': -5.0,
        'sir_max': 5.0,
        'enrollment_prefix': 'xvector',
        'seed': 42
    }
    
    # Check if input directories exist
    if not Path(config['librispeech_data_dir']).exists():
        logger.error(f"LibriSpeech data directory not found: {config['librispeech_data_dir']}")
        logger.info("Please update the 'librispeech_data_dir' path in the script")
        return False
    
    if config['enrollment_data_dir'] and not Path(config['enrollment_data_dir']).exists():
        logger.warning(f"Enrollment data directory not found: {config['enrollment_data_dir']}")
        logger.info("Will use training mode pattern for enrollment")
        config['enrollment_data_dir'] = None
    
    try:
        # Load LibriSpeech data
        logger.info(f"Loading LibriSpeech data from {config['librispeech_data_dir']}")
        data_loader = LibriSpeechDataLoader(config['librispeech_data_dir'])
        
        # Create overlap enrollment generator
        logger.info("Creating overlap enrollment generator")
        generator = OverlapEnrollmentGenerator(
            data_loader=data_loader,
            output_dir=config['output_dir'],
            enrollment_data_dir=config['enrollment_data_dir']
        )
        
        # Generate overlapped mixtures
        logger.info(f"Generating {config['num_mixtures']} overlap enrollments")
        sir_range = (config['sir_min'], config['sir_max'])
        generated_count = generator.generate_mixtures(config['num_mixtures'], sir_range)
        
        # Save all data files in ESPnet format
        logger.info("Saving data files in ESPnet format")
        generator.save_data_files()
        
        # Print results
        logger.info("=" * 60)
        logger.info("GENERATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Generated: {generated_count} overlap enrollments")
        logger.info(f"Output directory: {config['output_dir']}")
        logger.info(f"SIR range: [{config['sir_min']}, {config['sir_max']}] dB")
        
        # Show generated files
        output_path = Path(config['output_dir'])
        generated_files = []
        for file_pattern in ['wav.scp', 'utt2spk', 'spk2utt', 'text', 'spk2gender', 
                           f"{config['enrollment_prefix']}.scp", 'spk2enroll.json']:
            file_path = output_path / file_pattern
            if file_path.exists():
                generated_files.append(f"✓ {file_pattern}")
            else:
                generated_files.append(f"✗ {file_pattern} (not created)")
        
        logger.info("\nGenerated files:")
        for file_info in generated_files:
            logger.info(f"  {file_info}")
        
        # Show training instructions
        logger.info("\n" + "=" * 60)
        logger.info("ESPNET TRAINING INSTRUCTIONS")
        logger.info("=" * 60)
        logger.info("To use this data for target speaker ASR training:")
        logger.info(f"1. Copy/link the data directory to your recipe:")
        logger.info(f"   ln -s {os.path.abspath(config['output_dir'])} /path/to/espnet/egs2/recipe/data/")
        logger.info(f"2. Set the following parameters in your run script:")
        logger.info(f"   --train_set {Path(config['output_dir']).name}")
        logger.info(f"   --tgtspk_asr true")
        logger.info(f"   --enroll_prefix {config['enrollment_prefix']}")
        logger.info(f"   --enroll_type kaldi_ark")
        logger.info(f"3. Add target speaker ASR configuration to your config file")
        logger.info(f"4. Run training: ./asr_my.sh --stage 10")
        
        # Example data format
        logger.info("\n" + "=" * 40)
        logger.info("EXAMPLE DATA FORMAT")
        logger.info("=" * 40)
        
        wav_scp_file = output_path / "wav.scp"
        if wav_scp_file.exists():
            logger.info("wav.scp (first 3 lines):")
            with open(wav_scp_file, 'r') as f:
                for i, line in enumerate(f):
                    if i >= 3:
                        break
                    logger.info(f"  {line.strip()}")
        
        utt2spk_file = output_path / "utt2spk"
        if utt2spk_file.exists():
            logger.info("\nutt2spk (first 3 lines):")
            with open(utt2spk_file, 'r') as f:
                for i, line in enumerate(f):
                    if i >= 3:
                        break
                    logger.info(f"  {line.strip()}")
        
        enrollment_file = output_path / f"{config['enrollment_prefix']}.scp"
        if enrollment_file.exists():
            logger.info(f"\n{config['enrollment_prefix']}.scp (first 3 lines):")
            with open(enrollment_file, 'r') as f:
                for i, line in enumerate(f):
                    if i >= 3:
                        break
                    logger.info(f"  {line.strip()}")
        
        logger.info("\n🎉 Ready for ESPnet training!")
        return True
        
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
