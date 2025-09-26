#!/usr/bin/env bash

# Example usage script for create_overlap_enrollment.sh
# This shows how to use the overlap enrollment creation script

# Set paths (modify these according to your setup)
LIBRISPEECH_DATA="data/train_clean_100"  # Original LibriSpeech data directory
OUTPUT_DIR="data/train_enrollment_overlap"  # Output directory for overlapped enrollment
NUM_MIXTURES=5000  # Number of overlapped enrollment utterances to generate

echo "Creating overlapped enrollment data..."
echo "Input data: $LIBRISPEECH_DATA"
echo "Output directory: $OUTPUT_DIR"
echo "Number of mixtures: $NUM_MIXTURES"

# Run the overlap enrollment creation script
./local/create_overlap_enrollment.sh \
    $LIBRISPEECH_DATA \
    $OUTPUT_DIR \
    $NUM_MIXTURES

if [ $? -eq 0 ]; then
    echo "Successfully created overlapped enrollment data!"
    echo "You can find the results in: $OUTPUT_DIR"
    echo ""
    echo "Generated files:"
    echo "  - wav.scp: List of mixed audio files"
    echo "  - utt2spk: Utterance to speaker mapping (format: spk1_spk2)"
    echo "  - spk2utt: Speaker to utterance mapping"
    echo "  - text: Combined text transcriptions (if available)"
    echo "  - spk2gender: Speaker gender information (if available)"
    echo "  - mixed_audio/: Directory containing the actual mixed audio files"
    echo ""
    echo "Each mixed utterance has random SIR ∈ [-5, 5] dB"
else
    echo "Failed to create overlapped enrollment data!"
    exit 1
fi
