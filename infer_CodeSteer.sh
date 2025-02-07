#!/bin/bash
#SBATCH --job-name=CodeSteer-test
#SBATCH --partition=seas_gpu,gpu_requeue,serial_requeue,gpu
#SBATCH -n 16 # Number of cores
#SBATCH --gres=gpu:4
#SBATCH --constraint="h100"
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 0-72:00 # Runtime in D-HH:MM
#SBATCH --mem=168G # Memory pool for all cores in MB
#SBATCH -o CodeSteer-test.out   # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e CodeSteer-test.err   # File to which STDERR will be written, %j inserts jobid

export PYTHONPATH="path-to-current-dir/CodeSteer_Submission_Code_and_Data":$PYTHONPATH
module load cuda/12.4.1-fasrc01
python benchmark_test_CodeSteer.py