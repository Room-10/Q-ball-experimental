
import sys

import logging
logging.basicConfig(
    stream=sys.stdout,
    format="[%(relativeCreated) 8d] %(message)s",
    level=logging.DEBUG
)

import pickle
import numpy as np
from dipy.data import fetch_stanford_hardi, read_stanford_hardi
from dipy.segment.mask import median_otsu
from scipy.spatial import SphericalVoronoi

from manifold_sphere import Sphere
from tools import FRT_linop

def load_sphere(vecs=None, refinement=2):
    if vecs is not None:
        sphere_lvl = vecs.shape[1]
    else:
        sphere_lvl = "r{}".format(refinement)
    sphere_file = "manifolds/sphere-{}.pickle".format(sphere_lvl)
    try:
        sph = pickle.load(open(sphere_file, 'rb'))
    except:
        sph = Sphere(vecs=vecs, refinement=refinement)
        pickle.dump(sph, open(sphere_file, 'wb'))
    return sph

logging.debug("Loading realworld data ...")
fetch_stanford_hardi()
img, gtab = read_stanford_hardi()
S_data = img.get_data()

assert(len(S_data.shape) == 4)
assert(gtab.bvals is not None)
assert(gtab.bvecs.shape[1] == 3)
assert(S_data.shape[-1] == gtab.bvals.size)

logging.debug("Normalizing data E = S/S_0 ...")
S0 = S_data[..., gtab.bvals == 0].clip(1.0).mean(-1)
E_data = S_data[..., gtab.bvals > 0]/S0[..., None]

# log(-log(E))
loglog_data = np.log(-np.log(E_data.clip(.001, .999)))

# determine background pixels
_, mask = median_otsu(S_data, median_radius=3, numpass=1, dilate=2,
                              vol_idx=np.where(gtab.bvals > 0)[0])

# get noise intervals from background pixels (using mask)
# TODO

logging.debug("Preparing FRT ...")
b_vecs = gtab.bvecs[gtab.bvals > 0,...].T
b_sph = load_sphere(vecs=b_vecs)
sph = load_sphere(refinement=2)
FRT_op = FRT_linop(b_sph, sph)

logging.debug("Applying FRT to data ...")
FRT_data = np.einsum('kl,...l->...k', loglog_data, FRT_op)

# prepare laplace-beltrami operator on sphere grid
# U(x,x0) = -1/(4*np.pi) * np.log(np.abs(1 - <x,x_0>))
# TODO

# Then: rhs_l < [2 - log(2)]/(4*np.pi) - U(odf) < rhs_u