#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for overlap enrollment generation

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

from create_overlap_enrollment import AudioMixer, LibriSpeechDataLoader, OverlapEnrollmentGenerator


def create_test_audio(duration: float = 2.0, sr: int = 16000, freq: float = 440.0) -> np.ndarray:
    """Create a test sine wave audio signal"""
    t = np.linspace(0, duration, int(duration * sr), False)
    audio = 0.3 * np.sin(2 * np.pi * freq * t)
    return audio


def test_audio_mixer():
    """Test the AudioMixer class"""
    print("Testing AudioMixer...")
    
    # Create test audio signals
    audio1 = create_test_audio(freq=440.0)  # A4 note
    audio2 = create_test_audio(freq=880.0)  # A5 note (octave higher)
    
    mixer = AudioMixer()
    
    # Test different SIR values
    sir_values = [-5.0, 0.0, 5.0]
    
    for sir_db in sir_values:
        mixed_audio = mixer.mix_audio_with_sir(audio1, audio2, sir_db)
        
        # Check that mixed audio has reasonable properties
        assert len(mixed_audio) == min(len(audio1), len(audio2))
        assert not np.any(np.isnan(mixed_audio))
        assert not np.any(np.isinf(mixed_audio))
        
        print(f"  ✓ SIR {sir_db:+.1f} dB: mixed audio length = {len(mixed_audio)}")
    
    print("  ✓ AudioMixer tests passed!")


def test_audio_loading():
    """Test audio loading and saving"""
    print("Testing audio loading/saving...")
    
    # Create a temporary audio file
    test_audio = create_test_audio()
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        # Save test audio
        sf.write(tmp_path, test_audio, 16000)
        
        # Load audio using our mixer
        mixer = AudioMixer()
        loaded_audio, sr = mixer.load_audio(tmp_path)
        
        assert loaded_audio is not None
        assert sr == 16000
        assert len(loaded_audio) == len(test_audio)
        
        # Check if audio content is similar (allowing for small numerical differences)
        np.testing.assert_allclose(loaded_audio, test_audio, rtol=1e-5)
        
        print(f"  ✓ Audio loading/saving: {len(loaded_audio)} samples at {sr} Hz")
        
    finally:
        # Clean up
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    
    print("  ✓ Audio loading/saving tests passed!")


def create_test_data_dir():
    """Create a minimal test data directory with Kaldi format files"""
    test_dir = Path(tempfile.mkdtemp(prefix='test_librispeech_'))
    
    # Create test wav.scp
    wav_scp_content = [
        "utt_001 /path/to/audio1.wav",
        "utt_002 /path/to/audio2.wav",
        "utt_003 /path/to/audio3.wav",
        "utt_004 /path/to/audio4.wav",
    ]
    with open(test_dir / "wav.scp", 'w') as f:
        f.write('\n'.join(wav_scp_content) + '\n')
    
    # Create test utt2spk
    utt2spk_content = [
        "utt_001 spk_001",
        "utt_002 spk_002", 
        "utt_003 spk_001",
        "utt_004 spk_002",
    ]
    with open(test_dir / "utt2spk", 'w') as f:
        f.write('\n'.join(utt2spk_content) + '\n')
    
    # Create test text
    text_content = [
        "utt_001 hello world",
        "utt_002 test audio file",
        "utt_003 another utterance",
        "utt_004 final test utterance",
    ]
    with open(test_dir / "text", 'w') as f:
        f.write('\n'.join(text_content) + '\n')
    
    # Create test spk2gender
    spk2gender_content = [
        "spk_001 m",
        "spk_002 f",
    ]
    with open(test_dir / "spk2gender", 'w') as f:
        f.write('\n'.join(spk2gender_content) + '\n')
    
    return test_dir


def test_data_loader():
    """Test the LibriSpeechDataLoader class"""
    print("Testing LibriSpeechDataLoader...")
    
    # Create test data directory
    test_dir = create_test_data_dir()
    
    try:
        # Load data
        loader = LibriSpeechDataLoader(str(test_dir))
        
        # Check loaded data
        assert len(loader.wav_scp) == 4
        assert len(loader.utt2spk) == 4
        assert len(loader.spk2utt) == 2
        assert len(loader.text) == 4
        assert len(loader.spk2gender) == 2
        
        # Check specific mappings
        assert loader.utt2spk["utt_001"] == "spk_001"
        assert loader.utt2spk["utt_002"] == "spk_002"
        assert "utt_001" in loader.spk2utt["spk_001"]
        assert "utt_003" in loader.spk2utt["spk_001"]
        assert "utt_002" in loader.spk2utt["spk_002"]
        assert "utt_004" in loader.spk2utt["spk_002"]
        
        print(f"  ✓ Loaded {len(loader.wav_scp)} utterances from {len(loader.spk2utt)} speakers")
        print(f"  ✓ Speaker 'spk_001' has {len(loader.spk2utt['spk_001'])} utterances")
        print(f"  ✓ Speaker 'spk_002' has {len(loader.spk2utt['spk_002'])} utterances")
        
    finally:
        # Clean up
        import shutil
        shutil.rmtree(test_dir)
    
    print("  ✓ LibriSpeechDataLoader tests passed!")


def test_sir_calculation():
    """Test SIR calculation accuracy"""
    print("Testing SIR calculation accuracy...")
    
    # Create test signals with known power ratio
    signal1 = np.ones(1000) * 0.5  # Power = 0.25
    signal2 = np.ones(1000) * 0.25  # Power = 0.0625
    
    # Expected SIR = 10 * log10(0.25 / 0.0625) = 10 * log10(4) ≈ 6.02 dB
    expected_sir = 10 * np.log10(0.25 / 0.0625)
    
    mixer = AudioMixer()
    
    # Test with target SIR = 0 dB (equal power)
    mixed = mixer.mix_audio_with_sir(signal1, signal2, 0.0)
    
    # Calculate actual power ratio
    power1 = np.mean(signal1 ** 2)
    power2_after_scaling = np.mean((mixed - signal1) ** 2)
    actual_sir = 10 * np.log10(power1 / power2_after_scaling)
    
    print(f"  ✓ Original SIR: {expected_sir:.2f} dB")
    print(f"  ✓ Target SIR: 0.00 dB")
    print(f"  ✓ Actual SIR: {actual_sir:.2f} dB")
    
    # Allow for small numerical errors
    assert abs(actual_sir) < 0.1, f"SIR calculation error too large: {actual_sir:.3f} dB"
    
    print("  ✓ SIR calculation tests passed!")


def main():
    """Run all tests"""
    print("Running overlap enrollment generation tests...")
    print("=" * 50)
    
    try:
        test_audio_mixer()
        print()
        
        test_audio_loading()
        print()
        
        test_data_loader()
        print()
        
        test_sir_calculation()
        print()
        
        print("🎉 All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
