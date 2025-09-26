#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_100_sglspk"
valid_set="dev_sglspk"
test_sets="dev_sglspk test_sglspk"

expdir=exp/tswhisper

ngpu=4
device="3,4,5,6"

asr_config=conf/tswhisper/train_tsasr_whisper_medium_full_con20_q16_l2_crop10_lr5e-5.yaml

inference_config=conf/tswhisper/decode_asr_whisper_beam1.yaml
inference_args="--tgtspk_infer True"

# Add utils directory to PATH for run.pl
export PATH=$PWD/utils:$PATH

# 解决cuDNN版本不兼容问题
# 确保PyTorch使用自带的cuDNN而不是系统的cuDNN
export LD_LIBRARY_PATH=""
unset CUDNN_PATH
unset CUDNN_LIBRARY
unset CUDNN_INCLUDE_DIR

# 设置PyTorch使用自带的cuDNN
export TORCH_CUDNN_V8_API_ENABLED=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

# 训练阶段：使用4个GPU
CUDA_VISIBLE_DEVICES=${device} ./asr_my.sh \
    --stage 11 \
    --stop_stage 11 \
    --ngpu ${ngpu} \
    --expdir ${expdir} \
    --nj 32 \
    --gpu_inference true \
    --inference_nj 1 \
    --lang en \
    --tgtspk_asr true \
    --enroll_prefix "enroll" \
    --enroll_type "text" \
    --audio_format "sound" \
    --feats_type raw \
    --feats_normalize "" \
    --token_type whisper_multilingual \
    --max_wav_duration 30 \
    --use_lm false \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --inference_args "${inference_args}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    "$@"

# 推理阶段：切换为单GPU
export CUDA_VISIBLE_DEVICES="3"
inference_ngpu=1

./asr_my.sh \
    --stage 12 \
    --ngpu ${inference_ngpu} \
    --expdir ${expdir} \
    --nj 1 \
    --gpu_inference true \
    --inference_nj 1 \
    --lang en \
    --tgtspk_asr true \
    --enroll_prefix "enroll" \
    --enroll_type "text" \
    --audio_format "sound" \
    --feats_type raw \
    --feats_normalize "" \
    --token_type whisper_multilingual \
    --max_wav_duration 30 \
    --use_lm false \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --inference_args "${inference_args}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    "$@"
