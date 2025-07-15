import numpy as np
import tqdm

# OMP_NUM_PROCS=1 OMP_PROC_BIND=true python main_chain_no_hydro.py --input-file input_L320_phi001.dat

def go(sigma, L, phi, z_equals_zero=False):
    np.random.seed(0)

    # Parameters
    sigma = 2.972
    r = sigma / 2.0
    kbt_over_mg_e = 0.07
    kbt = 0.0041419464
    me_g = kbt / kbt_over_mg_e
    eta = 1.6e-3

    diffusive_time = kbt / (6 * np.pi * eta * r**3)
    print(f'diffusive time = {diffusive_time}')

    nCircles = round(L**2 * phi / (np.pi * r**2))
    print(f'N = {nCircles}')

    circles = np.zeros((nCircles, 7))

    for i in tqdm.trange(nCircles, mininterval=1, desc='generating inital state'):
        newCircleFound = False

        while not newCircleFound:
            x = -L / 2.0 + L * np.random.rand()
            y = -L / 2.0 + L * np.random.rand()
            if z_equals_zero:
                z = 0
            else:
                z = r + np.random.exponential(kbt_over_mg_e)

            prevCirclesZ = circles[:i, 2]
            prevCirclesY = circles[:i, 1]
            prevCirclesX = circles[:i, 0]

            DX = np.abs(prevCirclesX - x)
            DY = np.abs(prevCirclesY - y)
            DZ = np.abs(prevCirclesZ - z)

            DX = np.minimum(DX, L - DX)
            DY = np.minimum(DY, L - DY)

            distFromPrevCircles = np.sqrt(DX**2 + DY**2 + DZ**2)

            if i == 0 or np.sum(distFromPrevCircles <= 2 * r) == 0:
                newCircleFound = True
                circles[i, :] = [x, y, z, 1, 0, 0, 0]

    configs_file = f'suspensions/phi{phi}_L{L}.clones'
    print('saving', configs_file)
    np.savetxt(configs_file, circles, delimiter='\t', fmt=('%.9f', '%.9f', '%.9f', '%.1f', '%.1f', '%.1f', '%.1f'), header=f'{nCircles}', comments='')

    # If you need the commented part, uncomment and modify as needed.
    # configs_file_eq = f'/home/hat/Toy_Multiblobs_Code/Lubrication/Lubrication_Examples/Sophie_Boxes/suspension_phi_{phi}_L_{L}_eq.clones'
    # np.savetxt(configs_file_eq, np.vstack((len(pos), pos)), delimiter='\t', fmt='%.5f')

if __name__ == '__main__':
    sigma = 2.972
    L = 320
    phi = 0.016
    go(sigma, L, phi)