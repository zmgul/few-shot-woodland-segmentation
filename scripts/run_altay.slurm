#!/bin/bash
#SBATCH --job-name=woodland
#SBATCH --partition=a100q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

CONTAINER_IMAGE="$HOME/containers/woodland.sif"
WORK_DIR="$HOME/woodland"

if [ ! -f "$CONTAINER_IMAGE" ]; then
    echo "ERROR: $CONTAINER_IMAGE not found. Run first: sbatch build_image.slurm"
    exit 1
fi

mkdir -p $WORK_DIR/experiments $WORK_DIR/mlruns $WORK_DIR/slurm_logs

echo "Hostname: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Train
apptainer exec --nv --pwd /app \
    --bind $WORK_DIR/data:/app/data \
    --bind $WORK_DIR/experiments:/app/experiments \
    --bind $WORK_DIR/mlruns:/app/mlruns \
    --bind $WORK_DIR/src:/app/src \
    --bind $WORK_DIR/train.py:/app/train.py \
    --bind $WORK_DIR/eval.py:/app/eval.py \
    $CONTAINER_IMAGE python -u train.py

# Eval
apptainer exec --nv --pwd /app \
    --bind $WORK_DIR/data:/app/data \
    --bind $WORK_DIR/experiments:/app/experiments \
    --bind $WORK_DIR/mlruns:/app/mlruns \
    --bind $WORK_DIR/src:/app/src \
    --bind $WORK_DIR/train.py:/app/train.py \
    --bind $WORK_DIR/eval.py:/app/eval.py \
    $CONTAINER_IMAGE python -u eval.py

echo "Done."
