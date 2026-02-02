#!/bin/bash
#SBATCH --job-name=openvla_training_subtraj           # Name of your job
#SBATCH --output=%x_%j.out            # Output file (%x for job name, %j for job ID)
#SBATCH --error=%x_%j.err             # Error file
#SBATCH --partition=H100              # Partition to submit to (A100, V100, etc.)
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --mem=80G                     # Request 80 GB of memory
#SBATCH --time=24:00:00               # Time limit for the job (hh:mm:ss)

# Print job details
echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

# # Define variables for your job
# DATA_DIR="~/data"
# LR="1e-3"
# EPOCHS=100
# BATCH_SIZE=32

# Activate the environment
# source ~/miniconda3/condabin/conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate openvla-oft
export PYTHONPATH=$PYTHONPATH:/home/ids/ext-5219/tokenizer
# Execute the Python script with specific arguments
torchrun --standalone --nnodes 1 --nproc-per-node 1 finetune_subtrajectory_id.py 




# Print job completion time
echo "Job finished at: $(date)"