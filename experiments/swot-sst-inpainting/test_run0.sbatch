#!/bin/bash
#
#SBATCH --job-name="train_0"
#SBATCH --output=/scratch/tm3076/project/LDP_OEM/Tatsu/experiments/swot-sst-inpainting/
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --export=ALL
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --mem=128GB
#SBATCH --time=1-00:00:00
#SBATCH --partition=h200
#SBATCH --wrap="hostname && sleep infinity"

export WANDB_SILENT=true

singularity exec --bind /opt/slurm/bin:/opt/slurm/bin --overlay /scratch/tm3076/singularity_container/my_conda.ext3:ro  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash -c 'export PATH="/opt/slurm/bin:$PATH"; source /ext3/env.sh; conda activate priors; python' /scratch/tm3076/project/LDP_OEM/Tatsu/experiments/swot-sst-inpainting/.dawgz/528b5d40ce274b8288e3576f8e2fbafb/0000_train_0.py

