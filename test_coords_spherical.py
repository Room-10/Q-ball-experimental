
import itertools, random
import numpy as np
from numpy.linalg import norm

from tools import coords_spherical, coords_cartesian

phi_range = np.arange(1, 360, 24.56)
theta_range = np.arange(1, 180, 13.7)
coords_range = list(itertools.product(phi_range, theta_range))

for b_phi, b_theta in coords_range:
    phi = b_phi*np.pi/180.0
    theta = b_theta*np.pi/180.0
    coords = np.array([1.0, phi, theta])
    p = coords_cartesian(coords)
    try:
        assert norm(coords_spherical(p) - coords) < 1e-10
    except AssertionError as e:
        print("Test failed for phi={}, theta={}!".format(b_phi, b_theta))

print("All (%d) tests passed!" % len(coords_range))