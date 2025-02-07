#!/bin/bash
#SBATCH --job-name=SFT-1
#SBATCH --partition=seas_gpu,gpu_requeue,serial_requeue,gpu
#SBATCH -n 16 # Number of cores
#SBATCH --gres=gpu:4
#SBATCH --constraint="h100"
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 0-08:00 # Runtime in D-HH:MM
#SBATCH --mem=200G # Memory pool for all cores in MB
#SBATCH -o llama3-8B-full-train_SFT-1.out   # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e llama3-8B-full-train_SFT-1.err   # File to which STDERR will be written, %j inserts jobid

export HF_HOME=/n/holylabs/LABS/vlassak_lab/Users/ycchen/huggingface_saves/huggingface/
module load cuda/12.4.1-fasrc01

### Train the model
FORCE_TORCHRUN=1 llamafactory-cli train ./llama3_8B_train_SFT_CodeSteer.yaml
