#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example usage script for add_wham_noise.py

Copyright 2024  Northwestern Polytechnical University (author: Pengcheng Guo)
Apache 2.0
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_noise_addition(
    clean_data_dir: str,
    wham_noise_dir: str,
    output_dir: str,
    snr_min: float = 10.0,
    snr_max: float = 20.0,
    use_lufs: bool = False,
    lufs_min: float = -38.0,
    lufs_max: float = -30.0,
    seed: int = 42
):
    """Run the noise addition script"""
    
    script_path = Path(__file__).parent / "add_wham_noise.py"
    
    if not script_path.exists():
        print(f"Error: Script {script_path} not found!")
        return False
    
    cmd = [
        sys.executable,
        str(script_path),
        clean_data_dir,
        wham_noise_dir,
        output_dir,
        "--seed", str(seed)
    ]
    
    if use_lufs:
        cmd.extend([
            "--use-lufs",
            "--lufs-min", str(lufs_min),
            "--lufs-max", str(lufs_max)
        ])
    else:
        cmd.extend([
            "--snr-min", str(snr_min),
            "--snr-max", str(snr_max)
        ])
    
    print("Adding WHAM! noise to overlapped enrollment...")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, check=True)
        print("-" * 60)
        print("Successfully added noise to enrollment data!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: Script failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def validate_output(output_dir: str):
    """Validate the generated output"""
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"Error: Output directory {output_dir} does not exist!")
        return False
    
    required_files = ["wav.scp", "utt2spk", "spk2utt"]
    optional_files = ["text", "spk2gender"]
    
    print(f"\nValidating output in {output_dir}:")
    
    for file_name in required_files:
        file_path = output_path / file_name
        if file_path.exists():
            with open(file_path, 'r') as f:
                lines = f.readlines()
            print(f"  ✓ {file_name}: {len(lines)} entries")
        else:
            print(f"  ✗ {file_name}: Missing!")
            return False
    
    for file_name in optional_files:
        file_path = output_path / file_name
        if file_path.exists():
            with open(file_path, 'r') as f:
                lines = f.readlines()
            print(f"  ✓ {file_name}: {len(lines)} entries")
        else:
            print(f"  - {file_name}: Not available")
    
    # Check noisy audio directory
    noisy_audio_dir = output_path / "noisy_audio"
    if noisy_audio_dir.exists():
        audio_files = list(noisy_audio_dir.glob("*.wav"))
        print(f"  ✓ noisy_audio/: {len(audio_files)} audio files")
    else:
        print(f"  ✗ noisy_audio/: Missing!")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Example script for adding WHAM! noise to overlapped enrollment'
    )
    parser.add_argument(
        '--clean-data-dir',
        default='data/train_enrollment_overlap',
        help='Clean overlapped enrollment directory (default: data/train_enrollment_overlap)'
    )
    parser.add_argument(
        '--wham-noise-dir',
        required=True,
        help='WHAM! noise directory (required)'
    )
    parser.add_argument(
        '--output-dir',
        default='data/train_enrollment_overlap_noisy',
        help='Output directory (default: data/train_enrollment_overlap_noisy)'
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
        help='Use original LUFS-based approach instead of SNR'
    )
    parser.add_argument(
        '--lufs-min',
        type=float,
        default=-38.0,
        help='Minimum noise LUFS (default: -38.0)'
    )
    parser.add_argument(
        '--lufs-max',
        type=float,
        default=-30.0,
        help='Maximum noise LUFS (default: -30.0)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate existing output directory'
    )
    
    args = parser.parse_args()
    
    if args.validate_only:
        success = validate_output(args.output_dir)
        sys.exit(0 if success else 1)
    
    print("WHAM! Noise Addition to Overlapped Enrollment")
    print("=" * 50)
    print(f"Clean data directory: {args.clean_data_dir}")
    print(f"WHAM! noise directory: {args.wham_noise_dir}")
    print(f"Output directory: {args.output_dir}")
    
    if args.use_lufs:
        print(f"Noise method: LUFS-based (original Libri2Mix-noisy)")
        print(f"LUFS range: [{args.lufs_min}, {args.lufs_max}] LUFS")
    else:
        print(f"Noise method: SNR-based (controllable)")
        print(f"SNR range: [{args.snr_min}, {args.snr_max}] dB")
    
    print(f"Random seed: {args.seed}")
    print()
    
    # Check if input directories exist
    if not Path(args.clean_data_dir).exists():
        print(f"Error: Clean data directory {args.clean_data_dir} does not exist!")
        print("Please run create_overlap_enrollment.py first.")
        sys.exit(1)
    
    if not Path(args.wham_noise_dir).exists():
        print(f"Error: WHAM! noise directory {args.wham_noise_dir} does not exist!")
        print("Please download and extract WHAM! noise dataset.")
        sys.exit(1)
    
    # Run the noise addition
    success = run_noise_addition(
        args.clean_data_dir,
        args.wham_noise_dir,
        args.output_dir,
        args.snr_min,
        args.snr_max,
        args.use_lufs,
        args.lufs_min,
        args.lufs_max,
        args.seed
    )
    
    if success:
        # Validate output
        print("\nValidating generated data...")
        validate_success = validate_output(args.output_dir)
        
        if validate_success:
            print("\n🎉 All done! Noisy overlapped enrollment data created successfully!")
            print(f"\nYou can find the results in: {args.output_dir}")
            print("\nGenerated files:")
            print("  - wav.scp: List of noisy audio files")
            print("  - utt2spk: Utterance to speaker mapping")
            print("  - spk2utt: Speaker to utterance mapping")
            print("  - text: Text transcriptions (if available)")
            print("  - spk2gender: Speaker gender information (if available)")
            print("  - noisy_audio/: Directory containing noisy audio files")
            
            if args.use_lufs:
                print(f"\nNoise was added using LUFS range: [{args.lufs_min}, {args.lufs_max}] LUFS")
            else:
                print(f"\nNoise was added using SNR range: [{args.snr_min}, {args.snr_max}] dB")
        else:
            print("\n❌ Validation failed!")
            sys.exit(1)
    else:
        print("\n❌ Failed to add noise to enrollment data!")
        sys.exit(1)


if __name__ == "__main__":
    main()
