import gen_start
import main_chain_no_hydro
import os

def go(phi, L, t_save, dt, t_max, sigma, hydro, t_save2=0):
    gen_start.go(sigma, L, phi) # saves to f'suspensions/phi{phi}_L{L}.clones'
    structure_file = f'suspensions/phi{phi}_L{L}.clones'

    n_steps = int(t_max / dt)
    n_save = int(t_save / dt)
    mixt_n = int(t_save2 / dt)

    # output_name = f'/store/cartera/2d_monolayer/hydro_dt{dt}'
    hydropart = 'hydro' if hydro else 'nohydro'
    output_name = f'/store/cartera/2d_monolayer/{hydropart}'

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

#seed					 2

# Set output name
output_name                              {output_name}

# Load rigid bodies configuration, provide
# *.vertex and *.clones files
structure ./blob.vertex {structure_file}"""
    
    tempfilename = 'temp.dat'
    with open(tempfilename, 'w') as file:
        file.write(dat)

    main_chain_no_hydro.go(tempfilename, mixt_n=mixt_n, hydro=hydro)

    # delete temp.dat just to make sure
    os.remove(tempfilename)