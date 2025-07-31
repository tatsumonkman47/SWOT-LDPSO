#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --job-name=000_SIMVIP
#SBATCH --output=sbatch_output_logs/simvip_grad_loss_%A.out
#SBATCH --error=sbatch_output_logs/simvip_grad_loss_%A.err
#SBATCH --mail-user=tm3076@nyu.edu

module purge

singularity exec --nv \
        --overlay /scratch/tm3076/test_python_env/my_conda.ext3:ro \
        /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
        /bin/bash -c "source /ext3/env.sh; conda activate swot_python12_pytorch; python base_simvip_template.py --config-name=001_base_config"
