
import sys

import logging
logging.basicConfig(
    stream=sys.stdout,
    format="[%(relativeCreated) 8d] %(message)s",
    level=logging.DEBUG
)

import numpy as np
from dipy.data import fetch_stanford_hardi, read_stanford_hardi
from dipy.segment.mask import median_otsu
from scipy.spatial import SphericalVoronoi

from manifold_sphere import load_sphere
from tools import FRT_linop, InverseLaplaceBeltrami

def compute_bounds(S_data, bvals, FRT_op):
    # get noise intervals from background pixels (using mask)

    # determine background pixels
    #_, mask = median_otsu(S_data, median_radius=3, numpass=1, dilate=2,
    #                              vol_idx=np.where(gtab.bvals > 0)[0])

    # for the moment using trivial bounds
    logging.debug("Getting bounds on S0 and Si's")
    c = 0.05
    S0_l = S_data[..., gtab.bvals == 0].clip(1.0).mean(-1)*(1.0-c)
    S0_u = S_data[..., gtab.bvals == 0].clip(1.0).mean(-1)*(1.0+c)
    S_l = S_data[..., gtab.bvals > 0]*(1.0-c)
    S_u = S_data[..., gtab.bvals > 0]*(1.0+c)

    logging.debug("Bounds for normalized data E = S/S_0 ...")
    E_u = S_u/S0_l[..., None]
    E_l = S_l/S0_u[..., None]

    logging.debug("Bounds for the monotone decreasing log(-log) transform")
    loglog_E_u = np.log(-np.log(E_l.clip(.001, .999)))
    loglog_E_l = np.log(-np.log(E_u.clip(.001, .999)))

    logging.debug("Bounds for the FRT")
    rhs_u = np.einsum('kl,...l->...k', FRT_op, loglog_E_u)
    rhs_l = np.einsum('kl,...l->...k', FRT_op, loglog_E_l)

    return rhs_l, rhs_u

logging.debug("Loading realworld data ...")
fetch_stanford_hardi()
img, gtab = read_stanford_hardi()
S_data = img.get_data()

assert(len(S_data.shape) == 4)
assert(gtab.bvals is not None)
assert(gtab.bvecs.shape[1] == 3)
assert(S_data.shape[-1] == gtab.bvals.size)

logging.debug("Preparing FRT ...")
b_vecs = gtab.bvecs[gtab.bvals > 0,...].T
b_sph = load_sphere(vecs=b_vecs)
sph = load_sphere(refinement=2)
FRT_op = FRT_linop(b_sph, sph)

rhs_l, rhs_u = compute_bounds(S_data, gtab.bvals, FRT_op)

logging.debug("Preparing inverse laplacian ...")
U = InverseLaplaceBeltrami(sph, b_sph)

"""
Then:
    minimize TV(odf)
    s.t. rhs_l < [2 - log(2)]/(4*np.pi) - U(odf) < rhs_u,
         integral(odf) = 1
         odf >= 0
"""
