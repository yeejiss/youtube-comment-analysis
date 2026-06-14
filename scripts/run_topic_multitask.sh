#!/bin/bash
#SBATCH --job-name=topic_mt_ml
#SBATCH --partition=batch_ce_ugrad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/topic_mt_%j.out
#SBATCH --error=logs/topic_mt_%j.err

set -e

# 1. 환경 설정
source /data/$USER/anaconda3/etc/profile.d/conda.sh
conda activate grad_torch

PROJECT_ROOT="/data/$USER/repos/grad_project"
cd "$PROJECT_ROOT"

echo "=========================" 
echo "Job started" 
echo "Date: $(date)" 
echo "Current directory:" 
pwd 
echo "Python path:" 
which python 
echo "Running node:" 
hostname 
echo "========================="


## HuggingFace 캐시 경로
export HF_HOME=/data/$USER/hf_cache
export TRANSFORMERS_CACHE=/data/$USER/hf_cache/transformers
export HF_DATASETS_CACHE=/data/$USER/hf_cache/datasets

mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"


# 2. 실행 설정
DATA_TAR=/data/$USER/repos/grad_project/data/yt_topic_labeled_2189.tar
DATA_DIR=/local_datasets/yt_topic_labeled_2189
OUTPUT_DIR=/data/$USER/repos/grad_project/results/topic_multitask
SCRIPT_PATH=/data/$USER/repos/grad_project/train_topic_multitask.py


# 3. 데이터 준비
mkdir -p $DATA_DIR

echo "Checking files..."
if [ ! -f "$DATA_DIR/yt_topic_train.csv" ] || [ ! -f "$DATA_DIR/yt_topic_valid.csv" ] || [ ! -f "$DATA_DIR/yt_topic_test.csv" ]; then
  echo "Dataset not found in $DATA_DIR"
  echo "Extracting $DATA_TAR to $DATA_DIR"
  tar -xvf $DATA_TAR -C $DATA_DIR
else
  echo "Dataset already exists in $DATA_DIR"
fi
echo "Dataset files:"
ls -lh $DATA_DIR


# 4. 실행
echo "=========================" 
echo "Running topic multitask training"
echo "========================="

python ${SCRIPT_PATH} \
  --data_dir ${DATA_DIR} \
  --output_dir ${OUTPUT_DIR} \
  --model_name beomi/KcELECTRA-base-v2022 \
  --exp_name topic_multitask_kcelectra \
  --epochs 10 \
  --batch_size 16 \
  --eval_batch_size 32 \
  --max_length 64 \
  --lr 3e-5 \
  --target_loss_weight 1.0 \
  --attribute_loss_weight 1.2 \
  --selection_metric mean_macro_f1 \
  --target_threshold 0.48 \
  --attribute_threshold 0.33

echo "=========================" 
echo "Job finished successfully" 
echo "Date: $(date)" 
echo "========================="

