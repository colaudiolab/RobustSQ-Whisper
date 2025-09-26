#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script to validate the format of overlap enrollment data for ESPnet training

This script checks if the generated data follows the correct ESPnet format and
can be used for target speaker ASR training.

Copyright 2024  Northwestern Polytechnical University (author: Pengcheng Guo)
Apache 2.0
"""

import os
import sys
import json
import logging
from pathlib import Path
import soundfile as sf

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_file_exists(file_path, required=True):
    """Check if file exists and log result"""
    if file_path.exists():
        logger.info(f"✓ Found: {file_path}")
        return True
    else:
        level = logging.ERROR if required else logging.WARNING
        logger.log(level, f"✗ Missing: {file_path}")
        return False


def validate_wav_scp(data_dir):
    """Validate wav.scp file"""
    wav_scp_file = data_dir / "wav.scp"
    if not check_file_exists(wav_scp_file):
        return False
    
    logger.info("Validating wav.scp format...")
    valid_entries = 0
    invalid_entries = 0
    
    with open(wav_scp_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(None, 1)
            if len(parts) != 2:
                logger.error(f"Invalid wav.scp line {line_num}: {line}")
                invalid_entries += 1
                continue
            
            utt_id, wav_path = parts
            
            # Check utterance ID format for target speaker ASR
            if "_spk" not in utt_id:
                logger.warning(f"Utterance ID may not follow target speaker format: {utt_id}")
            
            # Check if audio file exists
            if not Path(wav_path).exists():
                logger.error(f"Audio file not found: {wav_path}")
                invalid_entries += 1
                continue
            
            # Check audio file format
            try:
                info = sf.info(wav_path)
                if info.samplerate != 16000:
                    logger.warning(f"Non-standard sample rate {info.samplerate} for {utt_id}")
            except Exception as e:
                logger.error(f"Error reading audio file {wav_path}: {e}")
                invalid_entries += 1
                continue
            
            valid_entries += 1
            
            # Only check first few entries for speed
            if line_num >= 5:
                break
    
    logger.info(f"wav.scp validation: {valid_entries} valid, {invalid_entries} invalid entries")
    return invalid_entries == 0


def validate_utt2spk(data_dir):
    """Validate utt2spk file"""
    utt2spk_file = data_dir / "utt2spk"
    if not check_file_exists(utt2spk_file):
        return False
    
    logger.info("Validating utt2spk format...")
    valid_entries = 0
    speakers = set()
    
    with open(utt2spk_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) != 2:
                logger.error(f"Invalid utt2spk line {line_num}: {line}")
                continue
            
            utt_id, spk_id = parts
            speakers.add(spk_id)
            valid_entries += 1
            
            # Only check first few entries for speed
            if line_num >= 5:
                break
    
    logger.info(f"utt2spk validation: {valid_entries} entries, {len(speakers)} speakers")
    return True


def validate_enrollment_scp(data_dir, prefix="xvector"):
    """Validate enrollment scp file"""
    enrollment_file = data_dir / f"{prefix}.scp"
    if not check_file_exists(enrollment_file, required=False):
        return True  # Optional file
    
    logger.info(f"Validating {prefix}.scp format...")
    training_pattern_count = 0
    file_path_count = 0
    
    with open(enrollment_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(None, 1)
            if len(parts) != 2:
                logger.error(f"Invalid {prefix}.scp line {line_num}: {line}")
                continue
            
            utt_id, enrollment_info = parts
            
            if enrollment_info.startswith("*"):
                # Training pattern format
                training_pattern_count += 1
                logger.debug(f"Training pattern: {line}")
            else:
                # File path format
                file_path_count += 1
                if not Path(enrollment_info).exists():
                    logger.warning(f"Enrollment file not found: {enrollment_info}")
            
            # Only check first few entries for speed
            if line_num >= 5:
                break
    
    logger.info(f"{prefix}.scp validation: {training_pattern_count} training patterns, "
                f"{file_path_count} file paths")
    return True


def validate_spk2enroll_json(data_dir):
    """Validate spk2enroll.json file"""
    json_file = data_dir / "spk2enroll.json"
    if not check_file_exists(json_file, required=False):
        return True  # Optional file
    
    logger.info("Validating spk2enroll.json format...")
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            spk2enroll = json.load(f)
        
        if not isinstance(spk2enroll, dict):
            logger.error("spk2enroll.json should contain a dictionary")
            return False
        
        speaker_count = len(spk2enroll)
        total_enrollments = sum(len(enrollments) for enrollments in spk2enroll.values())
        
        # Check format for a few speakers
        for spk_id, enrollments in list(spk2enroll.items())[:3]:
            if not isinstance(enrollments, list):
                logger.error(f"Enrollments for speaker {spk_id} should be a list")
                continue
            
            for enrollment in enrollments:
                if not isinstance(enrollment, list) or len(enrollment) != 2:
                    logger.error(f"Each enrollment should be [utt_id, path]: {enrollment}")
                    continue
                
                utt_id, path = enrollment
                if not isinstance(utt_id, str) or not isinstance(path, str):
                    logger.error(f"Enrollment format error: {enrollment}")
        
        logger.info(f"spk2enroll.json validation: {speaker_count} speakers, "
                    f"{total_enrollments} total enrollments")
        return True
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in spk2enroll.json: {e}")
        return False
    except Exception as e:
        logger.error(f"Error validating spk2enroll.json: {e}")
        return False


def validate_consistency(data_dir):
    """Validate consistency between files"""
    logger.info("Validating consistency between files...")
    
    # Load wav.scp utterances
    wav_scp_file = data_dir / "wav.scp"
    wav_utts = set()
    if wav_scp_file.exists():
        with open(wav_scp_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    wav_utts.add(line.split()[0])
    
    # Load utt2spk utterances
    utt2spk_file = data_dir / "utt2spk"
    utt2spk_utts = set()
    if utt2spk_file.exists():
        with open(utt2spk_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    utt2spk_utts.add(line.split()[0])
    
    # Check consistency
    if wav_utts != utt2spk_utts:
        logger.error("Utterance IDs in wav.scp and utt2spk don't match")
        logger.error(f"wav.scp only: {wav_utts - utt2spk_utts}")
        logger.error(f"utt2spk only: {utt2spk_utts - wav_utts}")
        return False
    
    logger.info(f"✓ Consistency check passed: {len(wav_utts)} utterances")
    return True


def validate_target_speaker_format(data_dir):
    """Validate target speaker ASR format"""
    logger.info("Validating target speaker ASR format...")
    
    utt2spk_file = data_dir / "utt2spk"
    if not utt2spk_file.exists():
        return False
    
    target_speaker_format_count = 0
    total_count = 0
    
    with open(utt2spk_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) != 2:
                continue
            
            utt_id, spk_id = parts
            total_count += 1
            
            # Check if utterance ID follows target speaker format
            if "_spk1" in utt_id or "_spk2" in utt_id:
                target_speaker_format_count += 1
    
    if target_speaker_format_count == 0:
        logger.warning("No utterances follow target speaker format (*_spk1, *_spk2)")
    else:
        logger.info(f"✓ Target speaker format: {target_speaker_format_count}/{total_count} utterances")
    
    return True


def main():
    if len(sys.argv) != 2:
        print("Usage: python test_overlap_enrollment_format.py <data_directory>")
        sys.exit(1)
    
    data_dir = Path(sys.argv[1])
    
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)
    
    logger.info(f"Validating overlap enrollment data in: {data_dir}")
    logger.info("=" * 60)
    
    # Run all validations
    validations = [
        ("Required files", lambda: all([
            check_file_exists(data_dir / "wav.scp"),
            check_file_exists(data_dir / "utt2spk"),
            check_file_exists(data_dir / "spk2utt"),
        ])),
        ("wav.scp format", lambda: validate_wav_scp(data_dir)),
        ("utt2spk format", lambda: validate_utt2spk(data_dir)),
        ("enrollment scp", lambda: validate_enrollment_scp(data_dir)),
        ("spk2enroll.json", lambda: validate_spk2enroll_json(data_dir)),
        ("file consistency", lambda: validate_consistency(data_dir)),
        ("target speaker format", lambda: validate_target_speaker_format(data_dir)),
    ]
    
    all_passed = True
    for name, validation_func in validations:
        logger.info(f"\n--- {name} ---")
        try:
            result = validation_func()
            if not result:
                all_passed = False
                logger.error(f"❌ {name} validation failed")
            else:
                logger.info(f"✅ {name} validation passed")
        except Exception as e:
            logger.error(f"❌ {name} validation error: {e}")
            all_passed = False
    
    # Summary
    logger.info("\n" + "=" * 60)
    if all_passed:
        logger.info("🎉 ALL VALIDATIONS PASSED")
        logger.info("Data is ready for ESPnet target speaker ASR training!")
        logger.info("\nTo use this data:")
        logger.info(f"1. Set --train_set {data_dir.name}")
        logger.info("2. Set --tgtspk_asr true")
        logger.info("3. Set --enroll_prefix xvector")
        logger.info("4. Use appropriate ASR config with target speaker support")
    else:
        logger.error("❌ SOME VALIDATIONS FAILED")
        logger.error("Please fix the issues before using for training")
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
