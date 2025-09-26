#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Add WHAM! environmental noise to overlapped enrollment with controllable SNR

Based on Libri2Mix-noisy but with customizable SNR control instead of fixed LUFS range.

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
import glob

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WHAMNoiseLoader:
    """Load WHAM! environmental noise data"""
    
    def __init__(self, wham_noise_dir: str):
        self.wham_noise_dir = Path(wham_noise_dir)
        self.noise_files = []
        self._load_noise_files()
    
    def _load_noise_files(self):
        """Load all noise files from WHAM! directory"""
        # WHAM! noise files are typically in .wav format
        noise_patterns = [
            "*.wav", "*.WAV", "*.flac", "*.FLAC"
        ]
        
        for pattern in noise_patterns:
            noise_files = list(self.wham_noise_dir.rglob(pattern))
            self.noise_files.extend(noise_files)
        
        if not self.noise_files:
            raise ValueError(f"No noise files found in {self.wham_noise_dir}")
        
        logger.info(f"Loaded {len(self.noise_files)} noise files from WHAM!")
    
    def get_random_noise_segment(self, duration: float, sr: int = 16000) -> Optional[np.ndarray]:
        """Get a random noise segment of specified duration"""
        # Randomly select a noise file
        noise_file = random.choice(self.noise_files)
        
        try:
            noise_audio, noise_sr = sf.read(str(noise_file))
            
            # Resample if necessary (simple nearest neighbor for now)
            if noise_sr != sr:
                # Simple resampling - for production use librosa.resample
                ratio = sr / noise_sr
                new_length = int(len(noise_audio) * ratio)
                indices = np.linspace(0, len(noise_audio) - 1, new_length).astype(int)
                noise_audio = noise_audio[indices]
            
            # Calculate required samples
            required_samples = int(duration * sr)
            
            if len(noise_audio) < required_samples:
                # Repeat noise if too short
                repeats = (required_samples // len(noise_audio)) + 1
                noise_audio = np.tile(noise_audio, repeats)
            
            # Randomly select a segment
            if len(noise_audio) > required_samples:
                start_idx = random.randint(0, len(noise_audio) - required_samples)
                noise_audio = noise_audio[start_idx:start_idx + required_samples]
            else:
                noise_audio = noise_audio[:required_samples]
            
            return noise_audio
            
        except Exception as e:
            logger.error(f"Error loading noise from {noise_file}: {e}")
            return None


class AudioProcessor:
    """Process audio with noise addition and SNR control"""
    
    @staticmethod
    def calculate_rms(audio: np.ndarray) -> float:
        """Calculate RMS (Root Mean Square) of audio signal"""
        return np.sqrt(np.mean(audio ** 2))
    
    @staticmethod
    def calculate_lufs(audio: np.ndarray, sr: int = 16000) -> float:
        """
        Simplified LUFS calculation (approximation)
        Real LUFS requires complex filtering, this is a simplified version
        """
        # Convert to dB relative to full scale
        rms = AudioProcessor.calculate_rms(audio)
        if rms == 0:
            return -float('inf')
        
        # Simplified LUFS approximation
        lufs = 20 * np.log10(rms) - 0.691  # Rough conversion factor
        return lufs
    
    @staticmethod
    def add_noise_with_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
        """Add noise to speech with specified SNR in dB"""
        # Calculate power of speech and noise
        speech_power = np.mean(speech ** 2)
        noise_power = np.mean(noise ** 2)
        
        if noise_power == 0:
            logger.warning("Noise signal has zero power, returning original speech")
            return speech
        
        # Convert SNR from dB to linear scale
        snr_linear = 10 ** (snr_db / 10.0)
        
        # Calculate required noise power: SNR = P_speech / P_noise
        target_noise_power = speech_power / snr_linear
        noise_scale = np.sqrt(target_noise_power / noise_power)
        
        # Scale noise and add to speech
        scaled_noise = noise * noise_scale
        noisy_speech = speech + scaled_noise
        
        return noisy_speech
    
    @staticmethod
    def add_noise_with_lufs(speech: np.ndarray, noise: np.ndarray, 
                           target_lufs: float, sr: int = 16000) -> np.ndarray:
        """Add noise to speech with specified LUFS level"""
        current_lufs = AudioProcessor.calculate_lufs(noise, sr)
        
        if current_lufs == -float('inf'):
            logger.warning("Cannot calculate LUFS for zero-power noise")
            return speech
        
        # Calculate scaling factor to achieve target LUFS
        lufs_diff = target_lufs - current_lufs
        scale_factor = 10 ** (lufs_diff / 20.0)
        
        # Scale noise and add to speech
        scaled_noise = noise * scale_factor
        noisy_speech = speech + scaled_noise
        
        return noisy_speech
    
    @staticmethod
    def clip_to_prevent_clipping(audio: np.ndarray, max_value: float = 0.9) -> np.ndarray:
        """Clip audio to prevent clipping, maintaining relative dynamics"""
        max_abs = np.max(np.abs(audio))
        
        if max_abs > max_value:
            scale_factor = max_value / max_abs
            audio = audio * scale_factor
            logger.debug(f"Clipped audio by factor {scale_factor:.3f} to prevent clipping")
        
        return audio


class NoisyEnrollmentGenerator:
    """Generate noisy enrollment data from clean overlapped enrollment"""
    
    def __init__(self, clean_data_dir: str, wham_noise_dir: str, output_dir: str):
        self.clean_data_dir = Path(clean_data_dir)
        self.output_dir = Path(output_dir)
        self.noise_loader = WHAMNoiseLoader(wham_noise_dir)
        self.processor = AudioProcessor()
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.noisy_audio_dir = self.output_dir / "noisy_audio"
        self.noisy_audio_dir.mkdir(exist_ok=True)
        
        # Load clean data
        self.clean_wav_scp = {}
        self.clean_utt2spk = {}
        self.clean_text = {}
        self.clean_spk2gender = {}
        
        self._load_clean_data()
    
    def _load_clean_data(self):
        """Load clean overlapped enrollment data"""
        # Load wav.scp
        wav_scp_file = self.clean_data_dir / "wav.scp"
        if wav_scp_file.exists():
            with open(wav_scp_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(None, 1)
                    if len(parts) == 2:
                        self.clean_wav_scp[parts[0]] = parts[1]
        
        # Load utt2spk
        utt2spk_file = self.clean_data_dir / "utt2spk"
        if utt2spk_file.exists():
            with open(utt2spk_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        self.clean_utt2spk[parts[0]] = parts[1]
        
        # Load text (optional)
        text_file = self.clean_data_dir / "text"
        if text_file.exists():
            with open(text_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(None, 1)
                    if len(parts) == 2:
                        self.clean_text[parts[0]] = parts[1]
        
        # Load spk2gender (optional)
        spk2gender_file = self.clean_data_dir / "spk2gender"
        if spk2gender_file.exists():
            with open(spk2gender_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        self.clean_spk2gender[parts[0]] = parts[1]
        
        logger.info(f"Loaded {len(self.clean_wav_scp)} clean utterances")
    
    def generate_noisy_data(self, snr_range: Tuple[float, float] = (10.0, 20.0), 
                           use_lufs: bool = False, lufs_range: Tuple[float, float] = (-38.0, -30.0)):
        """Generate noisy versions of clean overlapped enrollment"""
        
        output_wav_scp = {}
        output_utt2spk = {}
        output_text = {}
        output_spk2gender = {}
        
        total_utterances = len(self.clean_wav_scp)
        processed_count = 0
        
        logger.info(f"Processing {total_utterances} utterances...")
        if use_lufs:
            logger.info(f"Using LUFS range: [{lufs_range[0]}, {lufs_range[1]}] LUFS")
        else:
            logger.info(f"Using SNR range: [{snr_range[0]}, {snr_range[1]}] dB")
        
        for utt_id, clean_audio_path in self.clean_wav_scp.items():
            try:
                # Load clean audio
                clean_audio, sr = sf.read(clean_audio_path)
                
                if len(clean_audio) == 0:
                    logger.warning(f"Empty audio file: {clean_audio_path}")
                    continue
                
                # Get noise segment of same duration
                duration = len(clean_audio) / sr
                noise_segment = self.noise_loader.get_random_noise_segment(duration, sr)
                
                if noise_segment is None:
                    logger.warning(f"Failed to get noise segment for {utt_id}")
                    continue
                
                # Ensure same length
                min_len = min(len(clean_audio), len(noise_segment))
                clean_audio = clean_audio[:min_len]
                noise_segment = noise_segment[:min_len]
                
                # Add noise
                if use_lufs:
                    # Use LUFS-based noise addition (original Libri2Mix-noisy approach)
                    target_lufs = random.uniform(lufs_range[0], lufs_range[1])
                    noisy_audio = self.processor.add_noise_with_lufs(
                        clean_audio, noise_segment, target_lufs, sr
                    )
                else:
                    # Use SNR-based noise addition (controllable approach)
                    target_snr = random.uniform(snr_range[0], snr_range[1])
                    noisy_audio = self.processor.add_noise_with_snr(
                        clean_audio, noise_segment, target_snr
                    )
                
                # Clip to prevent clipping
                noisy_audio = self.processor.clip_to_prevent_clipping(noisy_audio, 0.9)
                
                # Create output filename
                noisy_utt_id = f"noisy_{utt_id}"
                noisy_audio_path = self.noisy_audio_dir / f"{noisy_utt_id}.wav"
                
                # Save noisy audio
                sf.write(str(noisy_audio_path), noisy_audio, sr)
                
                # Record information
                output_wav_scp[noisy_utt_id] = str(noisy_audio_path)
                
                if utt_id in self.clean_utt2spk:
                    output_utt2spk[noisy_utt_id] = self.clean_utt2spk[utt_id]
                
                if utt_id in self.clean_text:
                    output_text[noisy_utt_id] = self.clean_text[utt_id]
                
                # Copy speaker gender info
                if utt_id in self.clean_utt2spk:
                    spk_id = self.clean_utt2spk[utt_id]
                    if spk_id in self.clean_spk2gender:
                        output_spk2gender[spk_id] = self.clean_spk2gender[spk_id]
                
                processed_count += 1
                
                if processed_count % 100 == 0:
                    logger.info(f"Processed {processed_count}/{total_utterances} utterances...")
                
            except Exception as e:
                logger.error(f"Failed to process {utt_id}: {e}")
                continue
        
        # Save output data files
        self._save_data_files(output_wav_scp, output_utt2spk, output_text, output_spk2gender)
        
        logger.info(f"Successfully processed {processed_count}/{total_utterances} utterances")
        return processed_count
    
    def _save_data_files(self, wav_scp: Dict, utt2spk: Dict, text: Dict, spk2gender: Dict):
        """Save all data files in Kaldi format"""
        # Save wav.scp
        wav_scp_file = self.output_dir / "wav.scp"
        with open(wav_scp_file, 'w', encoding='utf-8') as f:
            for utt_id in sorted(wav_scp.keys()):
                f.write(f"{utt_id} {wav_scp[utt_id]}\n")
        
        # Save utt2spk
        utt2spk_file = self.output_dir / "utt2spk"
        with open(utt2spk_file, 'w', encoding='utf-8') as f:
            for utt_id in sorted(utt2spk.keys()):
                f.write(f"{utt_id} {utt2spk[utt_id]}\n")
        
        # Generate and save spk2utt
        spk2utt = {}
        for utt_id, spk_id in utt2spk.items():
            if spk_id not in spk2utt:
                spk2utt[spk_id] = []
            spk2utt[spk_id].append(utt_id)
        
        spk2utt_file = self.output_dir / "spk2utt"
        with open(spk2utt_file, 'w', encoding='utf-8') as f:
            for spk_id in sorted(spk2utt.keys()):
                utts = ' '.join(sorted(spk2utt[spk_id]))
                f.write(f"{spk_id} {utts}\n")
        
        # Save text if available
        if text:
            text_file = self.output_dir / "text"
            with open(text_file, 'w', encoding='utf-8') as f:
                for utt_id in sorted(text.keys()):
                    f.write(f"{utt_id} {text[utt_id]}\n")
        
        # Save spk2gender if available
        if spk2gender:
            spk2gender_file = self.output_dir / "spk2gender"
            with open(spk2gender_file, 'w', encoding='utf-8') as f:
                for spk_id in sorted(spk2gender.keys()):
                    f.write(f"{spk_id} {spk2gender[spk_id]}\n")
        
        logger.info(f"Saved noisy data files to {self.output_dir}")
        logger.info(f"Generated files:")
        logger.info(f"  - wav.scp: {len(wav_scp)} entries")
        logger.info(f"  - utt2spk: {len(utt2spk)} entries")
        logger.info(f"  - spk2utt: {len(spk2utt)} speakers")
        if text:
            logger.info(f"  - text: {len(text)} entries")
        if spk2gender:
            logger.info(f"  - spk2gender: {len(spk2gender)} entries")


def main():
    parser = argparse.ArgumentParser(
        description='Add WHAM! environmental noise to overlapped enrollment with controllable SNR'
    )
    parser.add_argument(
        'clean_data_dir',
        help='Clean overlapped enrollment data directory'
    )
    parser.add_argument(
        'wham_noise_dir', 
        help='WHAM! noise directory'
    )
    parser.add_argument(
        'output_dir',
        help='Output directory for noisy enrollment data'
    )
    parser.add_argument(
        '--snr-min',
        type=float,
        default=10.0,
        help='Minimum SNR in dB (default: 10.0)'
    )
    parser.add_argument(
        '--snr-max',
        type=float,
        default=20.0,
        help='Maximum SNR in dB (default: 20.0)'
    )
    parser.add_argument(
        '--use-lufs',
        action='store_true',
        help='Use LUFS-based noise addition instead of SNR-based'
    )
    parser.add_argument(
        '--lufs-min',
        type=float,
        default=-38.0,
        help='Minimum noise LUFS (default: -38.0, only used with --use-lufs)'
    )
    parser.add_argument(
        '--lufs-max',
        type=float,
        default=-30.0,
        help='Maximum noise LUFS (default: -30.0, only used with --use-lufs)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    try:
        # Create generator
        generator = NoisyEnrollmentGenerator(
            args.clean_data_dir,
            args.wham_noise_dir,
            args.output_dir
        )
        
        # Generate noisy data
        if args.use_lufs:
            processed_count = generator.generate_noisy_data(
                use_lufs=True,
                lufs_range=(args.lufs_min, args.lufs_max)
            )
            logger.info(f"Used LUFS-based noise addition with range [{args.lufs_min}, {args.lufs_max}] LUFS")
        else:
            processed_count = generator.generate_noisy_data(
                snr_range=(args.snr_min, args.snr_max),
                use_lufs=False
            )
            logger.info(f"Used SNR-based noise addition with range [{args.snr_min}, {args.snr_max}] dB")
        
        logger.info(f"Successfully generated {processed_count} noisy enrollment utterances")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Noisy audio files: {args.output_dir}/noisy_audio/")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
