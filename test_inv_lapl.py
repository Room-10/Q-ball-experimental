
import itertools
import numpy as np
from scipy.special import sph_harm, lpmv
from dipy.data import fetch_stanford_hardi, read_stanford_hardi

from tools import InverseLaplaceBeltrami, coords_spherical
from manifold_sphere import load_sphere

fetch_stanford_hardi()
img, gtab = read_stanford_hardi()
b_vecs = gtab.bvecs[gtab.bvals > 0,...].T
b_sph = load_sphere(vecs=b_vecs)
sph = load_sphere(refinement=2)
print(sph.v.shape)
print(b_sph.v.shape)
U = InverseLaplaceBeltrami(sph, b_sph)

n_range = range(0, 5)

for n in n_range:
    for k in range(-n, n+1):
        # f is the spherical harmonics of degree n and order k, eigenvalue lbd
        f = lambda x: sph_harm(k, n, *coords_spherical(x)[1:])
        lbd = -n*(n+1)

        f_b_sph = np.array(list(map(f, b_sph.v.T)))
        f_sph = np.array(list(map(f, sph.v.T)))
        U_f = np.einsum('jk,k->j', U, f_b_sph)
        int_f = np.einsum('k,k->', b_sph.b, f_b_sph)

        err = np.abs(int_f/(4*np.pi) - lbd*U_f - f_sph)
        print("n={:d}, k={: 2d}: {:0.5f}".format(n, k, np.amax(err)))