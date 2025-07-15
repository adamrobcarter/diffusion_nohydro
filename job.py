import subprocess
import numpy as np

def go(hydro, phi, L, sigma, t_save, t_max, t_save2=0, xy_potential_confinement=False, z_potential_confinement=False, suffix='', theta=0.0):
    if xy_potential_confinement:
        suffix += '_pot'
    if z_potential_confinement:
        suffix += '_zconf'
        assert xy_potential_confinement, 'must have open/open in x/y in order to have open in z'

    density = 4/3.14159 * phi / sigma**2
    N = L**2 * density
    num_rows_outputted = N * (t_max / t_save)
    filesize = num_rows_outputted * 3 * 4
    print(f'filesize will be {filesize/1e9:.1f}GB')
    assert filesize < 20e9, f'filesize will be {filesize/1e9:.1f}GB'

    HYDROCOMMAND = 'conda run --name libm --live-stream python run.py --hydro' # https://stackoverflow.com/a/72395091/1103752
    NOHYDROCOMMAND = 'python run.py --nohydro'
    command = HYDROCOMMAND if hydro else NOHYDROCOMMAND
    if xy_potential_confinement:
        command += ' --xypotentialconfinement'
    if z_potential_confinement:
        command += ' --zpotentialconfinement'
    hydrotext = 'hydro' if hydro else 'nohydro'
    partition = 'gpu' if hydro else 'std'
    gres = '#SBATCH --gres=gpu:1' if hydro else ''
    dt = 0.1 if hydro else 0.0125

    jobname = f'{hydrotext}_{phi}_L{L}{suffix}'


    fullcommand = f'{command} --phi {phi} --L {L} --dt {dt} --t_save {t_save} --t_save2 {t_save2} --t_max {t_max} --sigma {sigma} --theta {theta} --suffix "{suffix}" --jobname {jobname}'
    print('command:')
    print(fullcommand)
#B import sys
# sys.exit()

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
#SBATCH --time=6-00:00:00
set -e # stop on error
export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}
## Load software modules
# module load cuda/12.8
## Move data (if necessary) and launch application
## (use 'srun' for launching mpi applications)
cd ~/libm
{fullcommand}
## Finish gracefully
exit 0
"""

    jobfilename = 'job.temp.sh'
    with open(jobfilename, 'w') as jobfile:
        jobfile.write(job)

    subprocess.run(['sbatch', jobfilename])

# gravity / shear balance
# for phi in [0.02, 0.04, 0.06, 0.08, 0.1]:
# for phi in [0.2, 0.3, 0.4, 0.5]:
# for phi in [0.02, 0.04, 0.06, 0.08, 0.1]:
#     for hydro in [True]:
#         for theta in [0, 10]:
#         # for theta in [10]:
#             # hydro = False

#             if theta == 0 and phi != 0.02:
#                 continue

#             sigma_mult = 0.5
#             sigma = 2.972*sigma_mult
#             L = 640 * sigma_mult

#             t_save2 = 0
#             t_save = 0.5
#             # t_max = 1*60*60 # 1hr
#             # suffix = "_t0.5_1h"
#             t_max = 10*60 # 10m
#             suffix = f"_t0.5_10m_sigma{sigma}"

#             # theta = 10
#             # theta = 0
#             suffix += f'_theta{theta}'

#             xy_potential_confinement = False
#             z_potential_confinement = False
        
#             go(hydro, phi, L, sigma, t_save2, t_save, t_max, xy_potential_confinement, z_potential_confinement, suffix, theta=theta)


# PNV
# for phi in [0.01, 0.1, 0.2]:
#     go(
#         hydro = False,
#         phi   = phi,
#         L     = 320,
#         sigma = 2.972,
#         t_save = 0.5,
#         t_save2 = 0,
#         t_max = 1e4,
#         xy_potential_confinement = False,
#         z_potential_confinement = False,
#         suffix = '_short',
#         theta=0
#     )

# Centre of mass
# for L in np.logspace(np.log10(10), np.log10(1000), num=5):
#     L = int(L)
#     go(
#         hydro = True,
#         phi   = 0.1,
#         L     = L,
#         sigma = 2.972,
#         # t_save = 0.5,
#         # t_max = 1e4,
#         t_save = 5,
#         t_max = 1e4,
#         suffix = '_t1e4',
#         t_save2 = 0,
#         xy_potential_confinement = False,
#         z_potential_confinement = False,
#         theta=0,
#     )

# hydro (for donev)
# for L in [160, 320, 640, 1280]:
for L in np.logspace(np.log10(100), np.log10(2000), num=10):
    L = int(L)
    for pot in [True, False]:
        go(
            hydro = True,
            phi   = 0.114,
            L     = L,
            sigma = 2.972,
            t_save = 0.5,
            t_max = 1e2,
            suffix = '_t1e2_0.5',
            xy_potential_confinement = pot,
            z_potential_confinement = False,
        )