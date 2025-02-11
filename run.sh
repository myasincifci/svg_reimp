#!/bin/bash
#SBATCH --job-name=SVG
#SBATCH --partition=gpu-2d
#SBATCH --gpus-per-node=40gb:1
#SBATCH --ntasks-per-node=8
#SBATCH --output=logs/job-%j.out

#SBATCH --array=1-1

apptainer run --nv /home/myasincifci/containers/main/main.sif \
    python train.py --config-name svp