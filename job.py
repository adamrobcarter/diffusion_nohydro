import subprocess

hydro = False
phi = 0.114
L = 320
dt = 0.0125
t_save = 16
t_save2 = 0.5
t_max = 2560
# t_max = 2560000, # 2560000 is 711 hr
# t_max = 25600, # test_mixt
sigma = 2.972
suffix = "_testslurm"

HYDROCOMMAND = 'conda run --name libm --live-stream python run.py --hydro'
NOHYDROCOMMAND = 'python run.py --nohydro'
command = HYDROCOMMAND if hydro else NOHYDROCOMMAND
hydrotext = 'hydro' if hydro else 'nohydro'
partition = 'gpu' if hydro else 'std'
gres = '#SBATCH --gres=gpu:1' if hydro else ''

jobname = f'{hydrotext}-L{L}-phi{phi}'

job = f"""#!/bin/bash

#SBATCH --output=slurm%j-{jobname}.out
#SBATCH --job-name={jobname}
#SBATCH --mail-user=adam.carter@sorbonne-universite.fr
#SBATCH --mail-type=ALL
#SBATCH --partition={partition}
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
{gres}
#SBATCH --time=8-00:00:00

set -e # stop on error

export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}

## Load software modules
# module load cuda/12.8

## Move data (if necessary) and launch application
## (use 'srun' for launching mpi applications)
cd ~/libm
{command} --phi {phi} --L {L} --dt {dt} --t_save {t_save} --t_save2 {t_save2} --t_max {t_max} --sigma {sigma} --suffix "{suffix}"
# https://stackoverflow.com/a/72395091/1103752

## Finish gracefully
exit 0
"""

jobfilename = 'job.temp.sh'
with open(jobfilename, 'w') as jobfile:
    jobfile.write(job)

subprocess.run(['sbatch', jobfilename])