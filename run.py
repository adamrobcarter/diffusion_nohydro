import gen_start
import main_chain_no_hydro
import os
import argparse

def go(phi, L, t_save, dt, t_max, sigma, hydro, t_save2, suffix, xy_potential_confinement=False, z_potential_confinement=False):
    gen_start.go(sigma, L, phi, z_equals_zero=z_potential_confinement) # saves to f'suspensions/phi{phi}_L{L}.clones'
    structure_file = f'suspensions/phi{phi}_L{int(L)}.clones'

    n_steps = int(t_max / dt)
    n_save = int(t_save / dt)
    mixt_n = int(t_save2 / dt)

    # output_name = f'/store/cartera/2d_monolayer/hydro_dt{dt}'
    hydropart = 'hydro' if hydro else 'nohydro'
    output_name = f'/store/cartera/2d_monolayer/{hydropart}{suffix}'

    xy_potential_width = 0
    if xy_potential_confinement:
        xy_potential_width = [L, L, 0]
        L = 0

    dat = f"""# Select implementation to compute M and M*f
mobility_blobs_implementation            C++
mobility_vector_prod_implementation      pycuda

# Select implementation to compute the blobs-blob interactions
blob_blob_force_implementation           python

# Set time step, number of steps and save frequency
dt                                       {dt} # 2 is about the stability limit,  note: (6*pi*eta*a^3/kbT) ~ 20
n_steps                                  {n_steps}
n_save                                   {n_save}

# Set fluid viscosity (eta), gravity (g) and blob radius
eta                                      1.4e-03
g                                        0.0592
blob_radius                              {sigma/2}

# Stochastic parameters
kT                                       0.0041419464
solver_tolerance                         1.0e-4

periodic_length                          {float(L)}   {float(L)}    0.0

# Set parameters for the blob-blob interation
repulsion_strength                       0
debye_length                             0.1395

# Set interaction with the wall
repulsion_strength_wall                  0
debye_length_wall                        0.1395

# Set parameters for the firm-potential
repulsion_strength_firm                  0.0163 #0.0331
firm_delta                               1e-2

seed					                 0

# Set output name
output_name                              {output_name}

# Load rigid bodies configuration, provide
# *.vertex and *.clones files
structure ./blob.vertex {structure_file}"""
    
    tempfilename = 'temp.dat'
    with open(tempfilename, 'w') as file:
        file.write(dat)

    main_chain_no_hydro.go(tempfilename, mixt_n=mixt_n, hydro=hydro, xy_potential_confinement=xy_potential_confinement, xy_potential_width=xy_potential_width,
                           z_potential_confinement=z_potential_confinement)

    # delete temp.dat just to make sure
    # if os.path.isfile(tempfilename):
    #     os.remove(tempfilename)

    print('Finished.')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phi', type=float)
    parser.add_argument('--L', type=int)
    parser.add_argument('--t_save', type=float)
    parser.add_argument('--dt', type=float)
    parser.add_argument('--t_max', type=float)
    parser.add_argument('--sigma', type=float)
    parser.add_argument('--hydro', action='store_true')
    parser.add_argument('--nohydro', dest='hydro', action='store_false')
    parser.add_argument('--t_save2', type=float, default=0)
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--xypotentialconfinement', action='store_true')
    parser.add_argument('--zpotentialconfinement', action='store_true')

    args = parser.parse_args()

    go(
        phi     = args.phi,
        L       = args.L,
        t_save  = args.t_save,
        dt      = args.dt,
        t_max   = args.t_max,
        sigma   = args.sigma,
        hydro   = args.hydro,
        t_save2 = args.t_save2,
        suffix  = args.suffix,
        xy_potential_confinement = args.xypotentialconfinement,
        z_potential_confinement = args.zpotentialconfinement,
    )