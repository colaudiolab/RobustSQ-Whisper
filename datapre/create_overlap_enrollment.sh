#!/usr/bin/env bash

# Copyright 2024  Northwestern Polytechnical University (author: Pengcheng Guo)
# Apache 2.0

# Script to create overlapped enrollment by mixing two speakers with random SIR ∈ [-5, 5] dB

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <librispeech-data-dir> <output-dir> <num-mixtures>"
  echo "e.g.: $0 data/train_clean_100 data/train_enrollment_overlap 10000"
  exit 1
fi

data_dir=$1
output_dir=$2
num_mixtures=$3

# Check if input directory exists
[ ! -d $data_dir ] && echo "$0: no such directory $data_dir" && exit 1

# Create output directory
mkdir -p $output_dir || exit 1

# Required files
wav_scp=$data_dir/wav.scp
utt2spk=$data_dir/utt2spk
spk2utt=$data_dir/spk2utt

[ ! -f $wav_scp ] && echo "$0: expected file $wav_scp to exist" && exit 1
[ ! -f $utt2spk ] && echo "$0: expected file $utt2spk to exist" && exit 1
[ ! -f $spk2utt ] && echo "$0: expected file $spk2utt to exist" && exit 1

# Output files
output_wav_scp=$output_dir/wav.scp
output_utt2spk=$output_dir/utt2spk
output_text=$output_dir/text
output_spk2gender=$output_dir/spk2gender

# Remove existing output files
[[ -f "$output_wav_scp" ]] && rm $output_wav_scp
[[ -f "$output_utt2spk" ]] && rm $output_utt2spk
[[ -f "$output_text" ]] && rm $output_text
[[ -f "$output_spk2gender" ]] && rm $output_spk2gender

# Get all speakers
speakers=($(cut -d' ' -f1 $spk2utt))
num_speakers=${#speakers[@]}

echo "$0: Found $num_speakers speakers in $data_dir"
echo "$0: Generating $num_mixtures overlapped enrollments..."

# Create Python script for audio mixing
python_script=$output_dir/mix_enrollment.py
cat > $python_script << 'EOF'
#!/usr/bin/env python3

import numpy as np
import soundfile as sf
import argparse
import random
import os

def load_audio(file_path):
    """Load audio file and return waveform and sample rate"""
    try:
        waveform, sr = sf.read(file_path)
        return waveform, sr
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def mix_audio_with_sir(audio1, audio2, sir_db):
    """Mix two audio signals with specified SIR in dB"""
    # Convert SIR from dB to linear scale
    sir_linear = 10 ** (sir_db / 10.0)
    
    # Calculate power of both signals
    power1 = np.mean(audio1 ** 2)
    power2 = np.mean(audio2 ** 2)
    
    # Adjust the power of audio2 according to SIR
    # SIR = Power1 / Power2, so Power2_new = Power1 / SIR
    target_power2 = power1 / sir_linear
    scale_factor = np.sqrt(target_power2 / power2) if power2 > 0 else 0
    
    audio2_scaled = audio2 * scale_factor
    
    # Make sure both audios have the same length
    min_len = min(len(audio1), len(audio2_scaled))
    audio1 = audio1[:min_len]
    audio2_scaled = audio2_scaled[:min_len]
    
    # Mix the signals
    mixed_audio = audio1 + audio2_scaled
    
    return mixed_audio

def main():
    parser = argparse.ArgumentParser(description='Mix two audio files with random SIR')
    parser.add_argument('audio1_path', help='Path to first audio file')
    parser.add_argument('audio2_path', help='Path to second audio file')
    parser.add_argument('output_path', help='Path to output mixed audio file')
    parser.add_argument('--sir-min', type=float, default=-5.0, help='Minimum SIR in dB')
    parser.add_argument('--sir-max', type=float, default=5.0, help='Maximum SIR in dB')
    
    args = parser.parse_args()
    
    # Load audio files
    audio1, sr1 = load_audio(args.audio1_path)
    audio2, sr2 = load_audio(args.audio2_path)
    
    if audio1 is None or audio2 is None:
        print("Failed to load one or both audio files")
        return 1
    
    if sr1 != sr2:
        print(f"Sample rates don't match: {sr1} vs {sr2}")
        return 1
    
    # Generate random SIR
    sir_db = random.uniform(args.sir_min, args.sir_max)
    print(f"Using SIR: {sir_db:.2f} dB")
    
    # Mix audio
    mixed_audio = mix_audio_with_sir(audio1, audio2, sir_db)
    
    # Save mixed audio
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    sf.write(args.output_path, mixed_audio, sr1)
    
    return 0

if __name__ == "__main__":
    exit(main())
EOF

chmod +x $python_script

# Create output directory for mixed audio
mkdir -p $output_dir/mixed_audio

# Generate mixtures
count=0
while [ $count -lt $num_mixtures ]; do
    # Randomly select two different speakers
    spk1_idx=$((RANDOM % num_speakers))
    spk2_idx=$((RANDOM % num_speakers))
    
    # Make sure we don't select the same speaker
    while [ $spk1_idx -eq $spk2_idx ]; do
        spk2_idx=$((RANDOM % num_speakers))
    done
    
    spk1=${speakers[$spk1_idx]}
    spk2=${speakers[$spk2_idx]}
    
    # Get utterances for both speakers
    spk1_utts=($(grep "^$spk1 " $spk2utt | cut -d' ' -f2-))
    spk2_utts=($(grep "^$spk2 " $spk2utt | cut -d' ' -f2-))
    
    # Skip if either speaker has no utterances
    if [ ${#spk1_utts[@]} -eq 0 ] || [ ${#spk2_utts[@]} -eq 0 ]; then
        continue
    fi
    
    # Randomly select one utterance from each speaker
    utt1=${spk1_utts[$((RANDOM % ${#spk1_utts[@]}))]}
    utt2=${spk2_utts[$((RANDOM % ${#spk2_utts[@]}))]}
    
    # Get audio file paths
    audio1_path=$(grep "^$utt1 " $wav_scp | cut -d' ' -f2-)
    audio2_path=$(grep "^$utt2 " $wav_scp | cut -d' ' -f2-)
    
    # Skip if we can't find the audio files
    if [ -z "$audio1_path" ] || [ -z "$audio2_path" ]; then
        continue
    fi
    
    # Create mixed utterance ID
    mixed_utt_id="mix_${count}_${spk1}_${spk2}"
    mixed_audio_path="$output_dir/mixed_audio/${mixed_utt_id}.wav"
    
    # Mix the audio files
    if python3 $python_script "$audio1_path" "$audio2_path" "$mixed_audio_path"; then
        # Write to output files
        echo "$mixed_utt_id $mixed_audio_path" >> $output_wav_scp
        echo "$mixed_utt_id ${spk1}_${spk2}" >> $output_utt2spk
        
        # Create dummy text (mix of both speakers' texts if available)
        if [ -f $data_dir/text ]; then
            text1=$(grep "^$utt1 " $data_dir/text | cut -d' ' -f2- || echo "")
            text2=$(grep "^$utt2 " $data_dir/text | cut -d' ' -f2- || echo "")
            echo "$mixed_utt_id $text1 $text2" >> $output_text
        fi
        
        # Create speaker gender info if available
        if [ -f $data_dir/spk2gender ]; then
            gender1=$(grep "^$spk1 " $data_dir/spk2gender | cut -d' ' -f2 || echo "u")
            gender2=$(grep "^$spk2 " $data_dir/spk2gender | cut -d' ' -f2 || echo "u")
            echo "${spk1}_${spk2} ${gender1}_${gender2}" >> $output_spk2gender
        fi
        
        count=$((count + 1))
        
        if [ $((count % 100)) -eq 0 ]; then
            echo "Generated $count mixtures..."
        fi
    else
        echo "Failed to mix $utt1 and $utt2"
    fi
done

# Generate spk2utt
output_spk2utt=$output_dir/spk2utt
utils/utt2spk_to_spk2utt.pl <$output_utt2spk >$output_spk2utt || exit 1

# Validate data directory
utils/validate_data_dir.sh --no-feats $output_dir || exit 1

# Clean up
rm $python_script

echo "$0: Successfully generated $count overlapped enrollments in $output_dir"
echo "   - Random SIR range: [-5, 5] dB"
echo "   - Mixed audio files: $output_dir/mixed_audio/"
echo "   - Data files: wav.scp, utt2spk, spk2utt"

exit 0
