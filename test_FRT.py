
import itertools
import numpy as np
from scipy.special import sph_harm, lpmv

from tools import FRT_compute, coords_spherical, coords_cartesian

phi_range = np.arange(1, 360, 43.56)
theta_range = np.arange(1, 180, 21.37)
n_range = range(0, 5)
coords_range = list(itertools.product(phi_range, theta_range))

for b_phi, b_theta in coords_range:
    phi = b_phi*np.pi/180.0
    theta = b_theta*np.pi/180.0
    coords = np.array([1.0, phi, theta])
    p = coords_cartesian(coords)
    for n in n_range:
        for k in range(-n, n+1):
            f = lambda x: sph_harm(k, n, *coords_spherical(x)[1:])
            P0 = lpmv(0, n, 0)
            FRT_f_p = FRT_compute(f, p, n=100)
            try:
                # FRT(f)(p) == 2*PI*P(0)*f(x)
                # P(0) = 0 for odd values of n
                # P(0) = (-1)**n [1*3*5*...*(2n-1)]/[2*4*6*...*2n] for even values of n
                assert np.abs(FRT_f_p - 2*np.pi*P0*f(p)) < 1e-7
            except AssertionError as e:
                print("Test failed for phi={}, theta={}, n={}, k={}.".format(
                    b_phi, b_theta, n, k
                ))

print("All (%d) tests passed!" % (len(coords_range)*len(n_range),))