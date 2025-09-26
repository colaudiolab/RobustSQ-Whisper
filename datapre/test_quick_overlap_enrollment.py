#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quick test script for the modified overlap enrollment generator

This script creates a minimal test dataset to verify that the modified
create_overlap_enrollment.py works correctly for ESPnet training.

Copyright 2024  Northwestern Polytechnical University (author: Pengcheng Guo)
Apache 2.0
"""

import os
import sys
import tempfile
import shutil
import numpy as np
import soundfile as sf
from pathlib import Path
import logging

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from create_overlap_enrollment import LibriSpeechDataLoader, OverlapEnrollmentGenerator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_librispeech_data(temp_dir):
    """Create minimal test LibriSpeech data"""
    test_data_dir = Path(temp_dir) / "test_librispeech"
    test_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create test audio files
    audio_dir = test_data_dir / "audio"
    audio_dir.mkdir(exist_ok=True)
    
    sample_rate = 16000
    duration = 2.0  # 2 seconds
    samples = int(sample_rate * duration)
    
    # Create test audio files for 4 speakers, 2 utterances each
    speakers = ["1001", "1002", "2001", "2002"]
    utterances = {}
    
    for spk_id in speakers:
        for utt_idx in range(2):
            utt_id = f"{spk_id}-001-000{utt_idx}"
            
            # Generate simple sine wave with different frequencies for each speaker
            freq = 440 + int(spk_id) % 1000  # Different frequency per speaker
            t = np.linspace(0, duration, samples, False)
            audio = 0.1 * np.sin(2 * np.pi * freq * t)
            
            # Add some noise to make it more realistic
            noise = 0.01 * np.random.randn(samples)
            audio = audio + noise
            
            # Save audio file
            audio_path = audio_dir / f"{utt_id}.wav"
            sf.write(str(audio_path), audio, sample_rate)
            
            utterances[utt_id] = {
                'path': str(audio_path),
                'speaker': spk_id,
                'text': f"Test utterance {utt_idx} from speaker {spk_id}"
            }
    
    # Create wav.scp
    with open(test_data_dir / "wav.scp", 'w') as f:
        for utt_id, info in utterances.items():
            f.write(f"{utt_id} {info['path']}\n")
    
    # Create utt2spk  
    with open(test_data_dir / "utt2spk", 'w') as f:
        for utt_id, info in utterances.items():
            f.write(f"{utt_id} {info['speaker']}\n")
    
    # Create spk2utt
    spk2utt = {}
    for utt_id, info in utterances.items():
        spk_id = info['speaker']
        if spk_id not in spk2utt:
            spk2utt[spk_id] = []
        spk2utt[spk_id].append(utt_id)
    
    with open(test_data_dir / "spk2utt", 'w') as f:
        for spk_id, utts in spk2utt.items():
            f.write(f"{spk_id} {' '.join(utts)}\n")
    
    # Create text
    with open(test_data_dir / "text", 'w') as f:
        for utt_id, info in utterances.items():
            f.write(f"{utt_id} {info['text']}\n")
    
    # Create spk2gender (optional)
    with open(test_data_dir / "spk2gender", 'w') as f:
        for spk_id in speakers:
            gender = "m" if spk_id.startswith("1") else "f"
            f.write(f"{spk_id} {gender}\n")
    
    logger.info(f"Created test LibriSpeech data with {len(utterances)} utterances")
    logger.info(f"Speakers: {speakers}")
    
    return test_data_dir


def create_test_enrollment_data(temp_dir, librispeech_data_dir):
    """Create minimal test enrollment data"""
    enrollment_dir = Path(temp_dir) / "test_enrollment"  
    enrollment_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy some utterances from LibriSpeech data for enrollment
    with open(librispeech_data_dir / "wav.scp", 'r') as f:
        wav_entries = [line.strip().split(None, 1) for line in f if line.strip()]
    
    with open(librispeech_data_dir / "utt2spk", 'r') as f:
        utt2spk_entries = [line.strip().split() for line in f if line.strip()]
    
    # Take first utterance from each speaker for enrollment
    used_speakers = set()
    enrollment_utts = []
    
    for utt_id, spk_id in utt2spk_entries:
        if spk_id not in used_speakers:
            used_speakers.add(spk_id)
            enrollment_utts.append((utt_id, spk_id))
    
    # Create enrollment wav.scp
    with open(enrollment_dir / "wav.scp", 'w') as f:
        for utt_id, spk_id in enrollment_utts:
            # Find corresponding wav path
            for wav_utt, wav_path in wav_entries:
                if wav_utt == utt_id:
                    f.write(f"{utt_id} {wav_path}\n")
                    break
    
    # Create enrollment utt2spk
    with open(enrollment_dir / "utt2spk", 'w') as f:
        for utt_id, spk_id in enrollment_utts:
            f.write(f"{utt_id} {spk_id}\n")
    
    logger.info(f"Created test enrollment data with {len(enrollment_utts)} enrollments")
    
    return enrollment_dir


def test_overlap_enrollment_generation():
    """Test the overlap enrollment generation"""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info(f"Using temporary directory: {temp_dir}")
        
        try:
            # Create test data
            logger.info("Creating test LibriSpeech data...")
            librispeech_dir = create_test_librispeech_data(temp_dir)
            
            logger.info("Creating test enrollment data...")
            enrollment_dir = create_test_enrollment_data(temp_dir, librispeech_dir)
            
            # Test data loading
            logger.info("Testing data loading...")
            data_loader = LibriSpeechDataLoader(str(librispeech_dir))
            
            # Verify data was loaded correctly
            assert len(data_loader.wav_scp) > 0, "No wav.scp entries loaded"
            assert len(data_loader.utt2spk) > 0, "No utt2spk entries loaded"
            assert len(data_loader.spk2utt) > 0, "No speakers found"
            logger.info(f"✓ Loaded {len(data_loader.wav_scp)} utterances from {len(data_loader.spk2utt)} speakers")
            
            # Test overlap enrollment generation
            logger.info("Testing overlap enrollment generation...")
            output_dir = Path(temp_dir) / "overlap_output"
            
            generator = OverlapEnrollmentGenerator(
                data_loader=data_loader,
                output_dir=str(output_dir),
                enrollment_data_dir=str(enrollment_dir)
            )
            
            # Generate small number of mixtures for testing
            num_mixtures = 2
            generated_count = generator.generate_mixtures(num_mixtures, (-5.0, 5.0))
            
            assert generated_count > 0, "No mixtures generated"
            logger.info(f"✓ Generated {generated_count} mixtures")
            
            # Save data files
            logger.info("Testing data file saving...")
            generator.save_data_files()
            
            # Verify output files exist
            required_files = ["wav.scp", "utt2spk", "spk2utt"]
            optional_files = ["text", "spk2gender", "xvector.scp", "spk2enroll.json"]
            
            for filename in required_files:
                file_path = output_dir / filename
                assert file_path.exists(), f"Required file missing: {filename}"
                assert file_path.stat().st_size > 0, f"File is empty: {filename}"
                logger.info(f"✓ {filename} created and non-empty")
            
            for filename in optional_files:
                file_path = output_dir / filename
                if file_path.exists():
                    logger.info(f"✓ {filename} created")
                else:
                    logger.info(f"- {filename} not created (optional)")
            
            # Test format validation
            logger.info("Testing format validation...")
            
            # Check wav.scp format
            with open(output_dir / "wav.scp", 'r') as f:
                wav_lines = [line.strip() for line in f if line.strip()]
            
            # Check utt2spk format  
            with open(output_dir / "utt2spk", 'r') as f:
                utt2spk_lines = [line.strip() for line in f if line.strip()]
            
            # Verify target speaker format
            target_speaker_count = 0
            for line in utt2spk_lines:
                utt_id, spk_id = line.split()
                if "_spk1" in utt_id or "_spk2" in utt_id:
                    target_speaker_count += 1
            
            assert target_speaker_count > 0, "No target speaker format utterances found"
            logger.info(f"✓ Found {target_speaker_count} target speaker format utterances")
            
            # Check consistency
            wav_utts = {line.split()[0] for line in wav_lines}
            utt2spk_utts = {line.split()[0] for line in utt2spk_lines}
            assert wav_utts == utt2spk_utts, "Utterance ID mismatch between wav.scp and utt2spk"
            logger.info("✓ wav.scp and utt2spk are consistent")
            
            # Test audio files exist
            for line in wav_lines[:2]:  # Check first 2 files
                utt_id, wav_path = line.split(None, 1)
                assert Path(wav_path).exists(), f"Audio file missing: {wav_path}"
                
                # Check audio can be read
                info = sf.info(wav_path)
                assert info.samplerate == 16000, f"Wrong sample rate: {info.samplerate}"
                assert info.duration > 0, f"Empty audio file: {wav_path}"
            
            logger.info("✓ Audio files exist and are valid")
            
            logger.info("\n" + "="*50)
            logger.info("🎉 ALL TESTS PASSED!")
            logger.info("="*50)
            logger.info("The modified overlap enrollment generator works correctly")
            logger.info("Generated data is ready for ESPnet target speaker ASR training")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    logger.info("Testing overlap enrollment generator for ESPnet compatibility")
    logger.info("="*60)
    
    success = test_overlap_enrollment_generation()
    
    if success:
        logger.info("\n✅ Test completed successfully!")
        logger.info("The tool is ready for production use.")
    else:
        logger.error("\n❌ Test failed!")
        logger.error("Please check the errors above.")
    
    sys.exit(0 if success else 1)
