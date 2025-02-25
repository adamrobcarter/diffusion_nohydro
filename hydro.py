import run

run.go(
    # phi = 0.016,
    hydro=True,
    phi = 0.114,
    L = 640,
    # dt = 0.0125 * 2**5,
    dt = 0.1,
    t_save = 16,
    t_save2 = 0.5,
    t_max = 25600 * 8, # * 1000
    sigma = 2.972,
)