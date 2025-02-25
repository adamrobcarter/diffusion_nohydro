import run

run.go(
    # phi = 0.016,
    hydro=False,
    phi = 0.114,
    L = 640,
    # dt = 0.0125 * 2**5,
    dt = 0.0125,
    t_save = 16,
    t_save2 = 0.5,
    t_max = 2560000, # * 1000
    # 2560000 is 711 hr
    sigma = 2.972,
)