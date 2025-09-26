#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create overlapped enrollment by mixing two speakers with random SIR ∈ [-5, 5] dB

Copyright 2024  Northwestern Polytechnical University (author: Pengcheng Guo)
Apache 2.0
"""

import argparse
import os
import random
import numpy as np
import soundfile as sf
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LibriSpeechDataLoader:
    """Load LibriSpeech data from Kaldi format files"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.wav_scp = {}
        self.utt2spk = {}
        self.spk2utt = {}
        self.text = {}
        self.spk2gender = {}
        
        self._load_data()
    
    def _load_data(self):
        """Load all necessary data files"""
        # Load wav.scp
        wav_scp_file = self.data_dir / "wav.scp"
        if wav_scp_file.exists():
            with open(wav_scp_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(None, 1)
                    if len(parts) == 2:
                        self.wav_scp[parts[0]] = parts[1]
        
        # Load utt2spk
        utt2spk_file = self.data_dir / "utt2spk"
        if utt2spk_file.exists():
            with open(utt2spk_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        self.utt2spk[parts[0]] = parts[1]
        
        # Generate spk2utt from utt2spk
        for utt, spk in self.utt2spk.items():
            if spk not in self.spk2utt:
                self.spk2utt[spk] = []
            self.spk2utt[spk].append(utt)
        
        # Load text (optional)
        text_file = self.data_dir / "text"
        if text_file.exists():
            with open(text_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(None, 1)
                    if len(parts) == 2:
                        self.text[parts[0]] = parts[1]
        
        # Load spk2gender (optional)
        spk2gender_file = self.data_dir / "spk2gender"
        if spk2gender_file.exists():
            with open(spk2gender_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        self.spk2gender[parts[0]] = parts[1]
        
        logger.info(f"Loaded {len(self.wav_scp)} utterances from {len(self.spk2utt)} speakers")


class AudioMixer:
    """Mix two audio signals with specified SIR"""
    
    @staticmethod
    def load_audio(file_path: str) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """Load audio file and return waveform and sample rate"""
        try:
            waveform, sr = sf.read(file_path)
            return waveform, sr
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None, None
    
    @staticmethod
    def mix_audio_with_sir(audio1: np.ndarray, audio2: np.ndarray, sir_db: float) -> np.ndarray:
        """Mix two audio signals with specified SIR in dB"""
        # Convert SIR from dB to linear scale
        sir_linear = 10 ** (sir_db / 10.0)
        
        # Calculate power of both signals
        power1 = np.mean(audio1 ** 2)
        power2 = np.mean(audio2 ** 2)
        
        # Avoid division by zero
        if power2 == 0:
            logger.warning("Second audio signal has zero power, using original first signal")
            return audio1
        
        # Adjust the power of audio2 according to SIR
        # SIR = Power1 / Power2, so Power2_new = Power1 / SIR
        target_power2 = power1 / sir_linear
        scale_factor = np.sqrt(target_power2 / power2)
        
        audio2_scaled = audio2 * scale_factor
        
        # Make sure both audios have the same length
        min_len = min(len(audio1), len(audio2_scaled))
        audio1 = audio1[:min_len]
        audio2_scaled = audio2_scaled[:min_len]
        
        # Mix the signals
        mixed_audio = audio1 + audio2_scaled
        
        return mixed_audio


class OverlapEnrollmentGenerator:
    """Generate overlapped enrollment data"""
    
    def __init__(self, data_loader: LibriSpeechDataLoader, output_dir: str, 
                 enrollment_data_dir: str = None):
        self.data_loader = data_loader
        self.output_dir = Path(output_dir)
        self.enrollment_data_dir = Path(enrollment_data_dir) if enrollment_data_dir else None
        self.mixer = AudioMixer()
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mixed_audio_dir = self.output_dir / "mixed_audio"
        self.mixed_audio_dir.mkdir(exist_ok=True)
        
        # Output data structures
        self.output_wav_scp = {}
        self.output_utt2spk = {}
        self.output_text = {}
        self.output_spk2gender = {}
        
        # For enrollment generation
        self.enrollment_scp = {}
        self.spk2enroll = {}
        
        # Load enrollment data if provided
        if self.enrollment_data_dir:
            self._load_enrollment_data()
    
    def _load_enrollment_data(self):
        """Load enrollment data for target speaker training"""
        try:
            # Load enrollment wav.scp
            enrollment_wav_scp = self.enrollment_data_dir / "wav.scp"
            if enrollment_wav_scp.exists():
                with open(enrollment_wav_scp, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split(None, 1)
                        if len(parts) == 2:
                            self.enrollment_scp[parts[0]] = parts[1]
            
            # Load enrollment utt2spk
            enrollment_utt2spk_file = self.enrollment_data_dir / "utt2spk"
            enrollment_utt2spk = {}
            if enrollment_utt2spk_file.exists():
                with open(enrollment_utt2spk_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            enrollment_utt2spk[parts[0]] = parts[1]
            
            # Build spk2enroll mapping
            for utt_id, spk_id in enrollment_utt2spk.items():
                if utt_id in self.enrollment_scp:
                    if spk_id not in self.spk2enroll:
                        self.spk2enroll[spk_id] = []
                    self.spk2enroll[spk_id].append([utt_id, self.enrollment_scp[utt_id]])
            
            logger.info(f"Loaded enrollment data for {len(self.spk2enroll)} speakers")
            
        except Exception as e:
            logger.warning(f"Failed to load enrollment data: {e}")
            self.spk2enroll = {}
    
    def _generate_mixed_utterance_id(self, spk1: str, spk2: str, index: int, 
                                   target_spk: int = 1) -> str:
        """Generate mixed utterance ID in ESPnet format
        
        Args:
            spk1: First speaker ID
            spk2: Second speaker ID  
            index: Index of the mixture
            target_spk: Target speaker (1 or 2)
            
        Returns:
            Mixed utterance ID in format: {spk1}_{spk1}_{spk2}_spk{target_spk}
        """
        return f"{spk1}_{spk1}_{spk2}_spk{target_spk}"
    
    def generate_mixtures(self, num_mixtures: int, sir_range: Tuple[float, float] = (-5.0, 5.0)):
        """Generate specified number of mixed audio files"""
        speakers = list(self.data_loader.spk2utt.keys())
        
        if len(speakers) < 2:
            raise ValueError("Need at least 2 speakers to create mixtures")
        
        logger.info(f"Generating {num_mixtures} mixtures from {len(speakers)} speakers")
        logger.info(f"SIR range: [{sir_range[0]}, {sir_range[1]}] dB")
        
        count = 0
        attempts = 0
        max_attempts = num_mixtures * 10  # Prevent infinite loop
        
        while count < num_mixtures and attempts < max_attempts:
            attempts += 1
            
            # Randomly select two different speakers
            spk1, spk2 = random.sample(speakers, 2)
            
            # Get utterances for both speakers
            spk1_utts = self.data_loader.spk2utt[spk1]
            spk2_utts = self.data_loader.spk2utt[spk2]
            
            if not spk1_utts or not spk2_utts:
                continue
            
            # Randomly select one utterance from each speaker
            utt1 = random.choice(spk1_utts)
            utt2 = random.choice(spk2_utts)
            
            # Get audio file paths
            if utt1 not in self.data_loader.wav_scp or utt2 not in self.data_loader.wav_scp:
                continue
            
            audio1_path = self.data_loader.wav_scp[utt1]
            audio2_path = self.data_loader.wav_scp[utt2]
            
            # Load audio files
            audio1, sr1 = self.mixer.load_audio(audio1_path)
            audio2, sr2 = self.mixer.load_audio(audio2_path)
            
            if audio1 is None or audio2 is None:
                continue
            
            if sr1 != sr2:
                logger.warning(f"Sample rates don't match: {sr1} vs {sr2}, skipping")
                continue
            
            # Generate random SIR
            sir_db = random.uniform(sir_range[0], sir_range[1])
            
            # Mix audio
            try:
                mixed_audio = self.mixer.mix_audio_with_sir(audio1, audio2, sir_db)
                
                # Create mixed utterance IDs for both target speakers
                for target_spk in [1, 2]:
                    mixed_utt_id = self._generate_mixed_utterance_id(spk1, spk2, count, target_spk)
                    mixed_audio_path = self.mixed_audio_dir / f"{mixed_utt_id}.wav"
                    
                    # Save mixed audio (same audio for both targets)
                    sf.write(str(mixed_audio_path), mixed_audio, sr1)
                    
                    # Record information
                    self.output_wav_scp[mixed_utt_id] = str(mixed_audio_path)
                    
                    # Set target speaker for utt2spk
                    target_spk_id = spk1 if target_spk == 1 else spk2
                    self.output_utt2spk[mixed_utt_id] = target_spk_id
                    
                    # Combine text if available (use target speaker's text)
                    target_utt = utt1 if target_spk == 1 else utt2
                    if target_utt in self.data_loader.text:
                        self.output_text[mixed_utt_id] = self.data_loader.text[target_utt]
                    
                    # Set gender info if available
                    if target_spk_id in self.data_loader.spk2gender:
                        self.output_spk2gender[target_spk_id] = self.data_loader.spk2gender[target_spk_id]
                
                count += 1
                
                if count % 100 == 0:
                    logger.info(f"Generated {count}/{num_mixtures} mixtures...")
                
                if count % 1000 == 0:
                    logger.info(f"Last mixture: mix_{count:06d}, SIR: {sir_db:.2f} dB")
                
            except Exception as e:
                logger.error(f"Failed to mix {utt1} and {utt2}: {e}")
                continue
        
        if count < num_mixtures:
            logger.warning(f"Only generated {count}/{num_mixtures} mixtures after {attempts} attempts")
        
        return count
    
    def save_data_files(self):
        """Save all data files in Kaldi format"""
        # Save wav.scp
        wav_scp_file = self.output_dir / "wav.scp"
        with open(wav_scp_file, 'w', encoding='utf-8') as f:
            for utt_id in sorted(self.output_wav_scp.keys()):
                f.write(f"{utt_id} {self.output_wav_scp[utt_id]}\n")
        
        # Save utt2spk
        utt2spk_file = self.output_dir / "utt2spk"
        with open(utt2spk_file, 'w', encoding='utf-8') as f:
            for utt_id in sorted(self.output_utt2spk.keys()):
                f.write(f"{utt_id} {self.output_utt2spk[utt_id]}\n")
        
        # Generate and save spk2utt
        spk2utt = {}
        for utt_id, spk_id in self.output_utt2spk.items():
            if spk_id not in spk2utt:
                spk2utt[spk_id] = []
            spk2utt[spk_id].append(utt_id)
        
        spk2utt_file = self.output_dir / "spk2utt"
        with open(spk2utt_file, 'w', encoding='utf-8') as f:
            for spk_id in sorted(spk2utt.keys()):
                utts = ' '.join(sorted(spk2utt[spk_id]))
                f.write(f"{spk_id} {utts}\n")
        
        # Save text if available
        if self.output_text:
            text_file = self.output_dir / "text"
            with open(text_file, 'w', encoding='utf-8') as f:
                for utt_id in sorted(self.output_text.keys()):
                    f.write(f"{utt_id} {self.output_text[utt_id]}\n")
        
        # Save spk2gender if available
        if self.output_spk2gender:
            spk2gender_file = self.output_dir / "spk2gender"
            with open(spk2gender_file, 'w', encoding='utf-8') as f:
                for spk_id in sorted(self.output_spk2gender.keys()):
                    f.write(f"{spk_id} {self.output_spk2gender[spk_id]}\n")
        
        # Save enrollment files
        self.save_enrollment_files()
        
        logger.info(f"Saved data files to {self.output_dir}")
        logger.info(f"Generated files:")
        logger.info(f"  - wav.scp: {len(self.output_wav_scp)} entries")
        logger.info(f"  - utt2spk: {len(self.output_utt2spk)} entries")
        logger.info(f"  - spk2utt: {len(spk2utt)} speakers")
        if self.output_text:
            logger.info(f"  - text: {len(self.output_text)} entries")
        if self.output_spk2gender:
            logger.info(f"  - spk2gender: {len(self.output_spk2gender)} entries")
        
        # Show enrollment info
        enrollment_scp = self.generate_enrollment_scp()
        if enrollment_scp:
            logger.info(f"  - enrollment scp: {len(enrollment_scp)} entries")
        if self.spk2enroll:
            logger.info(f"  - spk2enroll.json: {len(self.spk2enroll)} speakers")
    
    def generate_enrollment_scp(self, prefix: str = "xvector"):
        """Generate enrollment scp files for target speaker ASR training"""
        enrollment_scp = {}
        
        for utt_id, spk_id in self.output_utt2spk.items():
            if spk_id in self.spk2enroll and self.spk2enroll[spk_id]:
                # Use the first enrollment for each speaker
                enroll_utt, enroll_path = self.spk2enroll[spk_id][0]
                enrollment_scp[utt_id] = enroll_path
            else:
                # For training mode, use pattern format
                enrollment_scp[utt_id] = f"*{utt_id} {spk_id}"
        
        return enrollment_scp
    
    def save_enrollment_files(self, prefix: str = "xvector"):
        """Save enrollment related files"""
        # Save enrollment scp
        enrollment_scp = self.generate_enrollment_scp(prefix)
        if enrollment_scp:
            enrollment_scp_file = self.output_dir / f"{prefix}.scp"
            with open(enrollment_scp_file, 'w', encoding='utf-8') as f:
                for utt_id in sorted(enrollment_scp.keys()):
                    f.write(f"{utt_id} {enrollment_scp[utt_id]}\n")
            logger.info(f"Saved enrollment scp: {enrollment_scp_file}")
        
        # Save spk2enroll.json
        if self.spk2enroll:
            spk2enroll_file = self.output_dir / "spk2enroll.json"
            with open(spk2enroll_file, 'w', encoding='utf-8') as f:
                import json
                json.dump(self.spk2enroll, f, indent=2)
            logger.info(f"Saved spk2enroll.json: {spk2enroll_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Create overlapped enrollment by mixing two speakers with random SIR'
    )
    parser.add_argument(
        'data_dir',
        help='LibriSpeech data directory (containing wav.scp, utt2spk, etc.)'
    )
    parser.add_argument(
        'output_dir',
        help='Output directory for mixed enrollment data'
    )
    parser.add_argument(
        'num_mixtures',
        type=int,
        help='Number of mixed utterances to generate'
    )
    parser.add_argument(
        '--sir-min',
        type=float,
        default=-5.0,
        help='Minimum SIR in dB (default: -5.0)'
    )
    parser.add_argument(
        '--sir-max',
        type=float,
        default=5.0,
        help='Maximum SIR in dB (default: 5.0)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--enrollment-data-dir',
        type=str,
        default=None,
        help='Directory containing enrollment data (wav.scp, utt2spk) for target speaker training'
    )
    parser.add_argument(
        '--enrollment-prefix',
        type=str,
        default='xvector',
        help='Prefix for enrollment scp file (default: xvector)'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    try:
        # Load data
        logger.info(f"Loading data from {args.data_dir}")
        data_loader = LibriSpeechDataLoader(args.data_dir)
        
        # Create generator
        generator = OverlapEnrollmentGenerator(
            data_loader, 
            args.output_dir,
            args.enrollment_data_dir
        )
        
        # Generate mixtures
        sir_range = (args.sir_min, args.sir_max)
        generated_count = generator.generate_mixtures(args.num_mixtures, sir_range)
        
        # Save data files
        generator.save_data_files()
        
        logger.info(f"Successfully generated {generated_count} overlapped enrollments")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Mixed audio files: {args.output_dir}/mixed_audio/")
        
        # Show training instructions
        logger.info("\n=== ESPnet Training Instructions ===")
        logger.info("To use this data for target speaker ASR training:")
        logger.info(f"1. Set train_set={Path(args.output_dir).name} in your run script")
        logger.info("2. Add --tgtspk_asr true flag")
        logger.info(f"3. Add --enroll_prefix {args.enrollment_prefix} flag") 
        logger.info("4. Make sure the data directory is accessible from your recipe")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
