import argparse
import numpy as np
import scipy.spatial as spatial
import subprocess
try:
    import pickle as cpickle
except:
    try:
        import cpickle
    except:
        import _pickle as cpickle
from functools import partial
import sys
import time
#import matplotlib.pyplot as plt

from numba import jit, njit, prange
from numba.typed import List as nblist

# Find project functions
found_functions = False
path_to_append = ''
# set the path below to your RigidMultiblobsWall for the body and read_input imports
sys.path.append('/home/cartera/RigidMultiblobsWall/')
import read_input
from body import body
from read_input import read_input
from read_input import read_vertex_file
from read_input import read_clones_file
from read_input import read_slip_file
# sys.path.append('/home/cartera/RigidMultiblobsWall/multi_bodies')
import multi_bodies_functions
print(sys.version)
print(sys.executable)
# while found_functions is False:
#     try:
#         from body import body
#         from read_input import read_input
#         from read_input import read_vertex_file
#         from read_input import read_clones_file
#         from read_input import read_slip_file
#         import multi_bodies_functions

#         found_functions = True
#     except ImportError:
#         path_to_append += '../'
#         print('searching functions in path ', path_to_append)
#         sys.path.append(path_to_append)
#         if len(path_to_append) > 21:
#             print('\nProjected functions not found. Edit path in multi_bodies.py')
#             sys.exit()

import tqdm
import os, struct
import looptiming


BURN_IN_SECONDS = 2.0

##########################################################################################
def Calc_F(r_vecs, FT_calc, bodies, offsets, list_of_neighbors):
    FT = FT_calc(bodies, r_vecs, offsets, list_of_neighbors)
    return FT[::2,:] # I guess "::2" is because it is rows of forces, torques, forces, torques etc

#'float64[:,:],float64[:,:],float64[:,:],float64,float64,float64,int32',
@njit(parallel=True, fastmath=True)
def EM_Step(r_vecs,W,F,D,kT,dt,vx,vy,Nb):
    r_out = 0*r_vecs
    for k in prange(Nb):
        for part in range(3):
            r_out[k,part] = r_vecs[k,part] + (dt*D/kT)*F[k,part] + np.sqrt(2.0*D*dt)*W[k,part]
        r_out[k,0] += dt*vx # this adds a deterministic drift velocity
        r_out[k,1] += dt*vy
    return r_out

def EM_hydro(r_vecs,FT_calc,kT,dt,bodies,offsets,list_of_neighbors, periodic_length, solver):

    Nb = r_vecs.shape[0]

    r_vecs_periodic = nb_put_r_vecs_in_periodic_box(r_vecs, periodic_length, Nb)
    F = Calc_F(r_vecs_periodic, FT_calc, bodies, offsets, list_of_neighbors)

    solver.setPositions(r_vecs)
    MF, _ = solver.Mdot(F)

    MW, _ = solver.sqrtMdotW()

    div_M = RFD(solver, r_vecs)

    r_out = r_vecs + dt*MF  + np.sqrt(2.0*kT*dt)*MW + kT*dt*div_M

    return r_out

def RFD(solver, r_vecs):
    delta = 1e-3
    W = np.random.normal(0,1,size=np.shape(r_vecs))
    solver.setPositions(r_vecs + 0.5*delta*W)
    MW_1, _ = solver.Mdot(W)
    solver.setPositions(r_vecs - 0.5*delta*W)
    MW_2, _ = solver.Mdot(W)
    return (MW_1 - MW_2)/delta


def stochastic_trap_hydro(r_vecs,FT_calc,kT,dt,bodies,offsets,list_of_neighbors, periodic_length, solver, loop_timer):

    Nb = r_vecs.shape[0]

    r_vecs_periodic = nb_put_r_vecs_in_periodic_box(r_vecs, periodic_length, Nb)

    loop_timer.start('Calc_F')
    F = Calc_F(r_vecs_periodic, FT_calc, bodies, offsets, list_of_neighbors)
    loop_timer.stop('Calc_F')

    solver.setPositions(r_vecs)
    
    loop_timer.start('Mdot')
    MF, _ = solver.Mdot(F)
    loop_timer.stop('Mdot')

    loop_timer.start('sqrtMdotW')
    MW, _ = solver.sqrtMdotW()
    loop_timer.stop('sqrtMdotW')

    loop_timer.start('RFD')
    div_M = RFD(solver, r_vecs)
    loop_timer.stop('RFD')

    r_predict = r_vecs + dt*MF  + np.sqrt(2.0*kT*dt)*MW + kT*dt*div_M

    r_pred_periodic = nb_put_r_vecs_in_periodic_box(r_predict, periodic_length, Nb)
    loop_timer.start('Calc_F2')
    F_pred = Calc_F(r_pred_periodic, FT_calc, bodies, offsets, list_of_neighbors)
    loop_timer.stop('Calc_F2')
    loop_timer.start('Mdot2')
    MF_pred, _ = solver.Mdot(F_pred)
    loop_timer.stop('Mdot2')

    return r_vecs + 0.5*dt*(MF + MF_pred) + np.sqrt(2.0*kT*dt)*MW + kT*dt*div_M

@njit(parallel=True, fastmath=True)
def nb_put_r_vecs_in_periodic_box(r_vecs_np, L, Nb):
    r_vecs = np.copy(r_vecs_np)
    for k in prange(Nb):
        for i in range(3):
            if L[i] > 0:
                while r_vecs[k,i] < 0:
                    r_vecs[k,i] += L[i]
                while r_vecs[k,i] > L[i]:
                    r_vecs[k,i] -= L[i]
    return r_vecs

def Update_Bodies_With_Hydro(r_vecs_np, W_old, FT_calc, periodic_length, kT, dt, bodies, offsets, list_of_neighbors, solver, loop_timer):
    '''
    Calculate Brownian Dynamics WITH hydro
    '''

    # r_new = EM_hydro(r_vecs_np,FT_calc,kT,dt,bodies, offsets, list_of_neighbors, periodic_length, solver) #this should be r_vecs, not r_out!!!
    r_new = stochastic_trap_hydro(r_vecs_np,FT_calc,kT,dt,bodies, offsets, list_of_neighbors, periodic_length, solver, loop_timer) #this should be r_vecs, not r_out!!!

    return r_new, W_old

def Update_Bodies_No_Hydro_2D(r_vecs_np, W_old, FT_calc, D, periodic_length, kT, dt, vx, vy, bodies, offsets, list_of_neighbors):
    '''
    Calculate Brownian Dynamics without hydro in 2D given a diffusion coeficient D
    '''
    #for k, b in enumerate(self.bodies):
    #    np.copyto(b.location_old, b.location)
    Nb = r_vecs_np.shape[0]
    r_vecs = nb_put_r_vecs_in_periodic_box(r_vecs_np, periodic_length, Nb)
    Wp = np.random.normal(0,1,r_vecs.shape)


    ################################################################################
    ####################
    #start = time.time()
    #F_new = Calc_F(r_vecs, FT_calc, bodies, offsets, list_of_neighbors)
    #end = time.time()
    ##print(np.linalg.norm(F_new))
    #print('F eval : ' + str((end - start)))
    #####################
    #start = time.time()
    #r_p = EM_Step(r_vecs_np,Wp,F_new,D,kT,0.5*dt,Nb)
    #r_p = nb_put_r_vecs_in_periodic_box(r_p, periodic_length, Nb)
    #F_p = Calc_F(r_p, FT_calc, bodies, offsets, list_of_neighbors)
    #W_new = np.sqrt(0.5)*(Wp + np.random.normal(0,1,r_vecs.shape))
    #r_new = EM_Step(r_vecs_np,W_new,F_p,D,kT,dt,Nb)
    #end = time.time()
    #print('evolve rvecs : ' + str((end - start)))
    ################################################################################

    #################################################################################
    #####################
    start = time.time()
    F_new = Calc_F(r_vecs, FT_calc, bodies, offsets, list_of_neighbors)
    end = time.time()
    #print(np.linalg.norm(F_new))
    # print('F eval : ' + str((end - start)))
    ####################
    start = time.time()
    if W_old is None:
        Win = Wp
    else:
        Win = 0.5*(W_old + Wp) #Wp
    r_new = EM_Step(r_vecs_np,Win,F_new,D,kT,dt,vx,vy,Nb) #this should be r_vecs, not r_out!!!
    W_old = Wp
    end = time.time()
    # print('evolve rvecs : ' + str((end - start)))
    #################################################################################

    # print('*performace mode* - no error or jump checks!!')

    return r_new, W_old

def build_neighbor_list(r_vectors,L,r_cut,eps_bs):
    #start = time.time()
    rvecs2D = r_vectors[:,0:2]
    L2D = L[0:2]
    r_tree = spatial.cKDTree(rvecs2D,boxsize=L2D, balanced_tree=False, compact_nodes=False) #
    #end = time.time()
    #print('make tree : ' + str((end - start)))

    #start = time.time()
    pairs = r_tree.query_ball_point(rvecs2D, r_cut, return_sorted=False, workers=1, eps=eps_bs) ############ EPS IS A MJOR BS FACTOR!!!!!!!!!!!!!
    #end = time.time()
    #print('query tree : ' + str((end - start)))

    #start = time.time()
    offsets = np.cumsum([0] + [len(p) for p in pairs], dtype=int)
    list_of_neighbors = np.fromiter((item for sublist in pairs for item in sublist), dtype=int)
    #end = time.time()
    #print('other shit : ' + str((end - start)))
    return offsets, list_of_neighbors

def go(input_file, hydro=True, mixt_n=0, xy_potential_confinement=False, xy_potential_width=None, z_potential_confinement=False,
       save_data=True, theta=0.0):

    # Read input file
    read = read_input.ReadInput(input_file)
    print('input file', input_file)

    # Set some variables for the simulation
    eta = read.eta
    a = read.blob_radius
    output_name = read.output_name
    structures = read.structures
    # print(structures)
    structures_ID = read.structures_ID

    # Copy input file to output
    subprocess.call(["cp", input_file, output_name + '_' + structures_ID[0] + '.inputfile'])

    # Set random generator state
    if True:
        print('using random seed')
        np.random.seed() # with no arguament, np will randomise the seed
    else:
        if read.random_state is not None:
            print('loading random state from disk')
            with open(read.random_state, 'rb') as f:
                np.random.set_state(cpickle.load(f))
        elif read.seed is not None:
            print('setting random seed', int(read.seed))
            np.random.seed(int(read.seed))

        # Save random generator state
        with open(output_name + '_' + structures_ID[0] + '.random_state', 'wb') as f:
            cpickle.dump(np.random.get_state(), f)

    # Create rigid bodies
    bodies = []
    body_types = []
    body_names = []
    for ID, structure in enumerate(structures):
        print('Creating structures = ', structure[1])
        # Read vertex and clones files
        struct_ref_config = read_vertex_file.read_vertex_file(structure[0])
        num_bodies_struct, struct_locations, struct_orientations = read_clones_file.read_clones_file(
            structure[1])
        # Read slip file if it exists
        slip = None
        if(len(structure) > 2):
            slip = read_slip_file.read_slip_file(structure[2])
        body_types.append(num_bodies_struct)
        body_names.append(structures_ID[ID])
        # Create each body of type structure
        for i in range(num_bodies_struct):
            b = body.Body(
                struct_locations[i], struct_orientations[i], struct_ref_config, a)
            b.ID = structures_ID[ID]
            # Calculate body length for the RFD
            if i == 0:
                b.calc_body_length()
            else:
                b.body_length = bodies[-1].body_length
            # Append bodies to total bodies list
            bodies.append(b)
    bodies = np.array(bodies)

    # Set some more variables
    num_of_body_types = len(body_types)
    num_bodies = bodies.size
    num_particles = len(bodies)
    Nblobs = sum([x.Nblobs for x in bodies])
    cutoff = read.Lub_Cut
    L = read.periodic_length
    n_steps = read.n_steps
    n_save = read.n_save
    dt = read.dt

    # print(L)

    for b in bodies:
        for i in range(3):
            if L[i] > 0:
                while b.location[i] < 0:
                    b.location[i] += L[i]
                while b.location[i] > L[i]:
                    b.location[i] -= L[i]

    firm_delta = read.firm_delta
    debye_length_delta = 2.0*a*firm_delta/np.log(1.0e1)
    repulsion_strength_delta = read.repulsion_strength_firm



    kT = read.kT


    multi_bodies_functions.calc_blob_blob_forces = multi_bodies_functions.set_blob_blob_forces(
        read.blob_blob_force_implementation)
    multi_bodies_functions.calc_body_body_forces_torques = multi_bodies_functions.set_body_body_forces_torques(
        read.body_body_force_torque_implementation)

    import time

    total_rej = 0

    calculation_params = dict(
        g=read.g,
        theta=theta,
        repulsion_strength_firm=repulsion_strength_delta,
        debye_length_firm=debye_length_delta,
        firm_delta=firm_delta,
        repulsion_strength_wall=read.repulsion_strength_wall,
        debye_length_wall=read.debye_length_wall,
        repulsion_strength=read.repulsion_strength,
        debye_length=read.debye_length,
        periodic_length=read.periodic_length,
        potential_width=read.periodic_length,
        a=a,
        kT=kT,
        gravity=True,
        wall=True,
    )

    if xy_potential_confinement:
        calculation_params['xy_potential_confinement'] = True
        calculation_params['xy_potential_width'] = xy_potential_width

    if z_potential_confinement:
        calculation_params['z_potential_confinement'] = True
        calculation_params['gravity'] = False
        calculation_params['wall'] = False
        calculation_params['z_trap_width'] = a/2

        assert xy_potential_confinement, 'must have open/open in x/y in order to have open in z'

    FT_calc = partial(multi_bodies_functions.force_torque_calculator_sort_by_bodies, **calculation_params)

    metadata = dict(calculation_params) # some of these kT, a etc want renaming for my system
    metadata['dt'] = dt

    r_vecs = np.array([b.location for b in bodies])
    print('xy min max', r_vecs[:, 0].min(), r_vecs[:, 0].max(), r_vecs[:, 1].min(), r_vecs[:, 1].max())
    W_old = None

    r_cut = 4.0*a
    eps_bs = 0.0
    offsets, list_of_neighbors = build_neighbor_list(r_vecs,L,r_cut,eps_bs)
    r_vecs_old = np.copy(r_vecs)

    if hydro:
        if z_potential_confinement:
            from libMobility import NBody
            solver = NBody("open", "open", "open")
            solver.setParameters()
            solver.initialize( # NperBatch * Nbatch != numberParticles
                temperature=kT,
                viscosity=eta,
                hydrodynamicRadius=a,
                numberParticles=Nblobs,
                tolerance=5e-3,
            )
        elif xy_potential_confinement:
            from libMobility import NBody
            solver = NBody("open", "open", "single_wall")
            solver.setParameters()
            solver.initialize( # NperBatch * Nbatch != numberParticles
                temperature=kT,
                viscosity=eta,
                hydrodynamicRadius=a,
                numberParticles=Nblobs,
                tolerance=5e-3,
            )
        else:
            from libMobility import DPStokes
            solver = DPStokes("periodic", "periodic", "single_wall")
            assert L[0] > 0
            assert L[1] > 0
            print(dt, L, 4*a)
            solver.setParameters(Lx=L[0], Ly=L[1], zmin=0, zmax=4*a)
            solver.initialize(
                temperature=kT,
                viscosity=eta,
                hydrodynamicRadius=a,
                numberParticles=Nblobs,
                tolerance=5e-3,
            )
    
    name = output_name + '_' + structures_ID[0] + '.npz'
    if save_data:
        if os.path.exists(name):
            # input(f'Exit now if you do not want to delete {name}')
            os.remove(name)
            print('deleted')
        print('saving to:', name)
    else:
        print('warning: not saving data')
        

    
    skip_timesteps = BURN_IN_SECONDS / dt
    skip_saves = skip_timesteps / n_save
    num_output_rows = Nblobs * int(n_steps / n_save - 1 - skip_saves)
    particles = np.full((num_output_rows, 5), np.nan)
    print(f'particles array size, {particles.nbytes/1e9:.1f}GB')

    loop_timer = looptiming.LoopTiming()
    t_start = time.time()

    for n in (progress := tqdm.trange(n_steps, smoothing=0.1, mininterval=10)):
        # print(n)
        # if n == 10:
        #     progress.mininterval = 10*60
        #     progress.refresh()


        time_s = n*dt
        # print(time_s)
        t0 = time.time()
        # print('USING BULLSHIT 2D NO HYDRO THING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        # print('positions', r_vecs)

        # if hydro condition can be here
        loop_timer.start('update bodies')
        if not hydro:
            r_vecs, W_old = Update_Bodies_No_Hydro_2D(r_vecs, W_old, FT_calc,
                                                      D=0.04, #0.0444, #0.017, #
                                                      periodic_length=L,
                                                      kT=kT,
                                                      dt=dt,
                                                      vx = 0.0,
                                                      vy = 0.0,
                                                      bodies=bodies,
                                                      offsets=offsets,
                                                      list_of_neighbors=list_of_neighbors) #D=0.0444 for experiments
        else:
            r_vecs, W_old = Update_Bodies_With_Hydro(r_vecs,
                                                    W_old,
                                                    FT_calc,
                                                    periodic_length=L,
                                                    kT=kT,
                                                    dt=dt,
                                                    bodies=bodies,
                                                    offsets=offsets,
                                                    list_of_neighbors=list_of_neighbors,
                                                    solver=solver, loop_timer=loop_timer)

        loop_timer.stop('update bodies')

        disps = np.sqrt(np.sum((r_vecs - r_vecs_old)**2, axis=1))
        max_disp = np.max(disps)
        # print("max_disp: "+str(max_disp))

        if(max_disp > 10.0):
            print("ERROR: MAX DISP TOO LARGE")
            exit(-1)
        if (max_disp < 1.0*a):
            dt1 = time.time() - t0
        else:
            r_vecs_old = np.copy(r_vecs)
            # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            # print("+++++++++++++++++++++RECACLCULATING TREE+++++++++++++++++++++")
            # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            r_vecs_per = nb_put_r_vecs_in_periodic_box(r_vecs, L, Nblobs)
            offsets, list_of_neighbors = build_neighbor_list(r_vecs_per,L,r_cut,eps_bs)
            # print(list_of_neighbors)
            #for k in range(4):
            #    print(list_of_neighbors[offsets[k]:offsets[k+1]])
            dt1 = time.time() - t0


        # print(("walltime for time step : %s" % dt1))
        #print("min(z): "+str(np.min(r_vecs[:,2])/a-1))

        if ((n % n_save == 0) or (n % n_save == mixt_n)) and (time_s > BURN_IN_SECONDS):
            frame = int((n - skip_timesteps) / n_save) - 1
            row_i = frame * Nblobs
            assert row_i >= 0

            for j in range(Nblobs):
                particles[row_i+j, [0, 1, 2]] = r_vecs[j,[0, 1, 2]] # Xs is [x0, y0, z0, x1, y1, z1, ...]
                particles[row_i+j, 3] = frame
                particles[row_i+j, 4] = j # particle ID

    t_end = time.time()
    metadata['computation_time'] = t_end-t_start

    if save_data:
        np.savez(name, particles=particles, **metadata)
        print(f'saved {name}')

    loop_timer.print_results()