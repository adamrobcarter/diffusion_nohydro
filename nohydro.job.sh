#!/bin/bash
set -e # stop on error

# SBATCH --job-name=hydro
# SBATCH --mail-user=adam.carter@sorbonne-universite.fr
# SBATCH --mail-type=ALL
# SBATCH --partition=std
# SBATCH --ntasks=1
# SBATCH --nodes=1
# SBATCH --cpus-per-task=1
# SBATCH --gres=gpu:1
# SBATCH --time=00-01:00:00
ABOVE NEEDS CHANGING

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

## Load software modules
# module load cuda/12.8

## Move data (if necessary) and launch application
## (use 'srun' for launching mpi applications)
cd ~/libm
conda run --name libm --live-stream python nohydro.py
# https://stackoverflow.com/a/72395091/1103752

## Finish gracefully
exit 0