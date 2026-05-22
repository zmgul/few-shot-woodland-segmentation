#!/bin/bash
#SBATCH --job-name=woodland-bench
#SBATCH --partition=a100q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/bench_%j.out
#SBATCH --error=slurm_logs/bench_%j.err

CONTAINER_IMAGE="$HOME/containers/woodland-ext.sif"
WORK_DIR="$HOME/woodland"

if [ ! -f "$CONTAINER_IMAGE" ]; then
    echo "ERROR: $CONTAINER_IMAGE not found. Run first: sbatch build_image.slurm"
    exit 1
fi

mkdir -p $WORK_DIR/experiments $WORK_DIR/slurm_logs

echo "Hostname: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

apptainer exec --nv --pwd /app \
    --bind $WORK_DIR/experiments:/app/experiments \
    --bind $WORK_DIR/src:/app/src \
    --bind $WORK_DIR/benchmark.py:/app/benchmark.py \
    $CONTAINER_IMAGE python -u benchmark.py

echo "Done."
