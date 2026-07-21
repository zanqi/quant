#!/bin/bash
#SBATCH --job-name=quant-w8a8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zanqil@uw.edu
#SBATCH --account=sciencehub
#SBATCH --partition=gpu-a40
#SBATCH --gpus=4
#SBATCH --cpus-per-task=24
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --chdir=/gscratch/sciencehub/zanqil/quant
#SBATCH --output=logs/quant_w8a8_%j.log
#SBATCH --export=all

source ~/.bashrc
conda activate quant

# ---- EDIT these four ----
export REPO_ID=keylazy/Qwen3.5-122B-A10B-INT8-W8A8          # EDIT: your HF repo
export OUT_DIR=/gscratch/sciencehub/zanqil/models/Qwen3.5-122B-A10B-INT8-W8A8  # EDIT if desired
# -------------------------

export MODEL_ID=Qwen/Qwen3.5-122B-A10B
export HF_HOME=/gscratch/sciencehub/zanqil/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=1
export CUDA_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p "$OUT_DIR"
echo "Quantizing $MODEL_ID on $SLURMD_NODENAME"
echo "  out : $OUT_DIR"
echo "  repo: $REPO_ID"

python quantize_w8a8.py

# sbatch /gscratch/sciencehub/zanqil/projects/quant/quantize.slurm