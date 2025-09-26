#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example usage script for create_overlap_enrollment.py

Copyright 2024  Northwestern Polytechnical University (author: Pengcheng Guo)
Apache 2.0
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_overlap_enrollment_creation(
    data_dir: str,
    output_dir: str,
    num_mixtures: int = 5000,
    sir_min: float = -5.0,
    sir_max: float = 5.0,
    seed: int = 42
):
    """Run the overlap enrollment creation script"""
    
    script_path = Path(__file__).parent / "create_overlap_enrollment.py"
    
    if not script_path.exists():
        print(f"Error: Script {script_path} not found!")
        return False
    
    cmd = [
        sys.executable,
        str(script_path),
        data_dir,
        output_dir,
        str(num_mixtures),
        "--sir-min", str(sir_min),
        "--sir-max", str(sir_max),
        "--seed", str(seed)
    ]
    
    print("Creating overlapped enrollment data...")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, check=True)
        print("-" * 60)
        print("Successfully created overlapped enrollment data!")
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
    
    # Check mixed audio directory
    mixed_audio_dir = output_path / "mixed_audio"
    if mixed_audio_dir.exists():
        audio_files = list(mixed_audio_dir.glob("*.wav"))
        print(f"  ✓ mixed_audio/: {len(audio_files)} audio files")
    else:
        print(f"  ✗ mixed_audio/: Missing!")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Example script for creating overlapped enrollment data'
    )
    parser.add_argument(
        '--data-dir',
        default='data/train_clean_100',
        help='LibriSpeech data directory (default: data/train_clean_100)'
    )
    parser.add_argument(
        '--output-dir',
        default='data/train_enrollment_overlap',
        help='Output directory (default: data/train_enrollment_overlap)'
    )
    parser.add_argument(
        '--num-mixtures',
        type=int,
        default=5000,
        help='Number of mixtures to generate (default: 5000)'
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
    
    print("Overlapped Enrollment Generation")
    print("=" * 40)
    print(f"Input data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of mixtures: {args.num_mixtures}")
    print(f"SIR range: [{args.sir_min}, {args.sir_max}] dB")
    print(f"Random seed: {args.seed}")
    print()
    
    # Check if input directory exists
    if not Path(args.data_dir).exists():
        print(f"Error: Input directory {args.data_dir} does not exist!")
        print("Please prepare LibriSpeech data first or specify correct path.")
        sys.exit(1)
    
    # Run the overlap enrollment creation
    success = run_overlap_enrollment_creation(
        args.data_dir,
        args.output_dir,
        args.num_mixtures,
        args.sir_min,
        args.sir_max,
        args.seed
    )
    
    if success:
        # Validate output
        print("\nValidating generated data...")
        validate_success = validate_output(args.output_dir)
        
        if validate_success:
            print("\n🎉 All done! Overlapped enrollment data created successfully!")
            print(f"\nYou can find the results in: {args.output_dir}")
            print("\nGenerated files:")
            print("  - wav.scp: List of mixed audio files")
            print("  - utt2spk: Utterance to speaker mapping (format: spk1_spk2)")
            print("  - spk2utt: Speaker to utterance mapping")
            print("  - text: Combined text transcriptions (if available)")
            print("  - spk2gender: Speaker gender information (if available)")
            print("  - mixed_audio/: Directory containing mixed audio files")
            print(f"\nEach mixed utterance has random SIR ∈ [{args.sir_min}, {args.sir_max}] dB")
        else:
            print("\n❌ Validation failed!")
            sys.exit(1)
    else:
        print("\n❌ Failed to create overlapped enrollment data!")
        sys.exit(1)


if __name__ == "__main__":
    main()
