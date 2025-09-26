#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for WHAM! noise addition

Copyright 2024  Northwestern Polytechnical University (author: Pengcheng Guo)
Apache 2.0
"""

import numpy as np
import soundfile as sf
import tempfile
import os
from pathlib import Path
import sys

# Add the local directory to path to import our modules
sys.path.insert(0, str(Path(__file__).parent))

from add_wham_noise import AudioProcessor, WHAMNoiseLoader, NoisyEnrollmentGenerator


def create_test_audio(duration: float = 2.0, sr: int = 16000, freq: float = 440.0) -> np.ndarray:
    """Create a test sine wave audio signal"""
    t = np.linspace(0, duration, int(duration * sr), False)
    audio = 0.3 * np.sin(2 * np.pi * freq * t)
    return audio


def create_test_noise(duration: float = 2.0, sr: int = 16000, freq: float = 100.0) -> np.ndarray:
    """Create a test noise signal (low frequency sine wave)"""
    t = np.linspace(0, duration, int(duration * sr), False)
    noise = 0.1 * np.sin(2 * np.pi * freq * t)
    return noise


def test_audio_processor():
    """Test the AudioProcessor class"""
    print("Testing AudioProcessor...")
    
    # Create test signals
    speech = create_test_audio(freq=440.0)  # A4 note
    noise = create_test_noise(freq=100.0)   # Low frequency noise
    
    processor = AudioProcessor()
    
    # Test RMS calculation
    rms_speech = processor.calculate_rms(speech)
    rms_noise = processor.calculate_rms(noise)
    
    print(f"  ✓ RMS calculation - Speech: {rms_speech:.4f}, Noise: {rms_noise:.4f}")
    
    # Test SNR-based noise addition
    snr_values = [5.0, 10.0, 15.0, 20.0]
    
    for target_snr in snr_values:
        noisy_speech = processor.add_noise_with_snr(speech, noise, target_snr)
        
        # Calculate actual SNR
        speech_power = np.mean(speech ** 2)
        added_noise_power = np.mean((noisy_speech - speech) ** 2)
        actual_snr = 10 * np.log10(speech_power / added_noise_power)
        
        print(f"  ✓ SNR {target_snr:2.0f} dB: actual = {actual_snr:5.2f} dB")
        
        # Check if SNR is close to target (allow 0.1 dB tolerance)
        assert abs(actual_snr - target_snr) < 0.1, f"SNR error too large: {actual_snr:.3f} vs {target_snr}"
    
    # Test clipping prevention
    large_signal = np.ones(1000) * 2.0  # Signal that exceeds [-1, 1]
    clipped_signal = processor.clip_to_prevent_clipping(large_signal, 0.9)
    
    assert np.max(np.abs(clipped_signal)) <= 0.9, "Clipping failed"
    print(f"  ✓ Clipping prevention: max value = {np.max(np.abs(clipped_signal)):.3f}")
    
    print("  ✓ AudioProcessor tests passed!")


def test_lufs_calculation():
    """Test LUFS calculation"""
    print("Testing LUFS calculation...")
    
    processor = AudioProcessor()
    
    # Test with different amplitude signals
    amplitudes = [0.1, 0.3, 0.5, 0.7]
    
    for amp in amplitudes:
        test_signal = create_test_audio() * amp
        lufs = processor.calculate_lufs(test_signal)
        
        # LUFS should be more negative for quieter signals
        print(f"  ✓ Amplitude {amp:.1f}: LUFS = {lufs:6.2f}")
    
    # Test zero signal
    zero_signal = np.zeros(1000)
    lufs_zero = processor.calculate_lufs(zero_signal)
    assert lufs_zero == -float('inf'), "Zero signal should have -inf LUFS"
    
    print("  ✓ LUFS calculation tests passed!")


def create_test_wham_directory():
    """Create a test directory with fake WHAM! noise files"""
    test_dir = Path(tempfile.mkdtemp(prefix='test_wham_'))
    
    # Create some test noise files
    for i in range(3):
        noise_data = create_test_noise(duration=5.0, freq=50.0 + i * 25)  # Different frequencies
        noise_file = test_dir / f"noise_{i:02d}.wav"
        sf.write(str(noise_file), noise_data, 16000)
    
    return test_dir


def test_wham_noise_loader():
    """Test the WHAMNoiseLoader class"""
    print("Testing WHAMNoiseLoader...")
    
    # Create test WHAM directory
    test_dir = create_test_wham_directory()
    
    try:
        # Load noise files
        loader = WHAMNoiseLoader(str(test_dir))
        
        assert len(loader.noise_files) == 3, f"Expected 3 noise files, got {len(loader.noise_files)}"
        
        # Test getting random noise segments
        for duration in [1.0, 2.0, 3.0]:
            noise_segment = loader.get_random_noise_segment(duration)
            
            assert noise_segment is not None, f"Failed to get noise segment of {duration}s"
            expected_samples = int(duration * 16000)
            assert len(noise_segment) == expected_samples, f"Wrong segment length: {len(noise_segment)} vs {expected_samples}"
            
            print(f"  ✓ Noise segment {duration}s: {len(noise_segment)} samples")
        
    finally:
        # Clean up
        import shutil
        shutil.rmtree(test_dir)
    
    print("  ✓ WHAMNoiseLoader tests passed!")


def create_test_clean_data():
    """Create test clean overlapped enrollment data"""
    test_dir = Path(tempfile.mkdtemp(prefix='test_clean_'))
    
    # Create test audio directory
    audio_dir = test_dir / "mixed_audio"
    audio_dir.mkdir()
    
    # Create test audio files
    wav_scp_content = []
    utt2spk_content = []
    text_content = []
    
    for i in range(2):
        # Create test audio
        audio_data = create_test_audio(duration=2.0, freq=440.0 + i * 100)
        audio_file = audio_dir / f"mix_{i:06d}_spk001_spk002.wav"
        sf.write(str(audio_file), audio_data, 16000)
        
        utt_id = f"mix_{i:06d}_spk001_spk002"
        wav_scp_content.append(f"{utt_id} {audio_file}")
        utt2spk_content.append(f"{utt_id} spk001_spk002")
        text_content.append(f"{utt_id} test utterance {i}")
    
    # Write data files
    with open(test_dir / "wav.scp", 'w') as f:
        f.write('\n'.join(wav_scp_content) + '\n')
    
    with open(test_dir / "utt2spk", 'w') as f:
        f.write('\n'.join(utt2spk_content) + '\n')
    
    with open(test_dir / "text", 'w') as f:
        f.write('\n'.join(text_content) + '\n')
    
    return test_dir


def test_noisy_enrollment_generator():
    """Test the NoisyEnrollmentGenerator class"""
    print("Testing NoisyEnrollmentGenerator...")
    
    # Create test clean data
    clean_dir = create_test_clean_data()
    wham_dir = create_test_wham_directory()
    output_dir = Path(tempfile.mkdtemp(prefix='test_noisy_'))
    
    try:
        # Create generator
        generator = NoisyEnrollmentGenerator(
            str(clean_dir),
            str(wham_dir),
            str(output_dir)
        )
        
        # Test loading clean data
        assert len(generator.clean_wav_scp) == 2, f"Expected 2 clean utterances, got {len(generator.clean_wav_scp)}"
        
        # Generate noisy data with SNR
        processed_count = generator.generate_noisy_data(
            snr_range=(15.0, 20.0),
            use_lufs=False
        )
        
        assert processed_count == 2, f"Expected to process 2 utterances, got {processed_count}"
        
        # Check output files
        assert (output_dir / "wav.scp").exists(), "wav.scp not created"
        assert (output_dir / "utt2spk").exists(), "utt2spk not created"
        assert (output_dir / "noisy_audio").exists(), "noisy_audio directory not created"
        
        # Check noisy audio files
        noisy_files = list((output_dir / "noisy_audio").glob("*.wav"))
        assert len(noisy_files) == 2, f"Expected 2 noisy audio files, got {len(noisy_files)}"
        
        print(f"  ✓ Generated {processed_count} noisy utterances")
        print(f"  ✓ Created {len(noisy_files)} noisy audio files")
        
    finally:
        # Clean up
        import shutil
        shutil.rmtree(clean_dir)
        shutil.rmtree(wham_dir)
        shutil.rmtree(output_dir)
    
    print("  ✓ NoisyEnrollmentGenerator tests passed!")


def main():
    """Run all tests"""
    print("Running WHAM! noise addition tests...")
    print("=" * 40)
    
    try:
        test_audio_processor()
        print()
        
        test_lufs_calculation()
        print()
        
        test_wham_noise_loader()
        print()
        
        test_noisy_enrollment_generator()
        print()
        
        print("🎉 All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
