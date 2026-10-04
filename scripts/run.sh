#!/bin/bash
#SBATCH --job-name=ac_ex4
#SBATCH --account=project_2020210
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:gh200:1
#SBATCH -o ./slurm_log/out_%j.txt
#SBATCH -e ./slurm_log/err_%j.txt
#SBATCH --mem=217086

# Example usage:
# sbatch scripts/run.sh

#====================SLURM LOG===================
# 2026/08/10:
# slurm_log/out_.txt
# slurm_log/out_.txt
#====================SLURM LOG===================



# env
module load python-pytorch/2.10

# hf cache
export HF_HOME=/scratch/project_2020210/.cache/huggingface

# Run the training here

