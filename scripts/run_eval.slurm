#!/bin/bash
#SBATCH --job-name=eval_woodland
#SBATCH --partition=a100q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/eval_%j.out
#SBATCH --error=slurm_logs/eval_%j.err

CONTAINER_IMAGE="$HOME/containers/woodland.sif"
WORK_DIR="$HOME/woodland"

apptainer exec --nv --pwd /app \
    --bind $WORK_DIR/data:/app/data \
    --bind $WORK_DIR/experiments:/app/experiments \
    --bind $WORK_DIR/mlruns:/app/mlruns \
    --bind $WORK_DIR/src:/app/src \
    --bind $WORK_DIR/eval.py:/app/eval.py \
    $CONTAINER_IMAGE python eval.py
