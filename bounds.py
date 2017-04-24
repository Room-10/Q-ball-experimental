
import logging

import numpy as np

from tools import FRT_linop

def compute_bounds(data, sph):
    # get noise intervals from background pixels (using mask)

    # determine background pixels
    #_, mask = median_otsu(S_data, median_radius=3, numpass=1, dilate=2,
    #                              vol_idx=np.where(gtab.bvals > 0)[0])

    # for the moment using trivial bounds
    b_sph = data['sph']
    bvals = data['gtab'].bvals
    S_data = data['S']

    logging.debug("Getting bounds on S0 and Si's")
    c = 0.05
    S0_l = S_data[..., bvals == 0].clip(1.0).mean(-1)*(1.0-c)
    S0_u = S_data[..., bvals == 0].clip(1.0).mean(-1)*(1.0+c)
    S_l = S_data[..., bvals > 0]*(1.0-c)
    S_u = S_data[..., bvals > 0]*(1.0+c)

    logging.debug("Bounds for normalized data E = S/S_0 ...")
    E_u = S_u/S0_l[..., None]
    E_l = S_l/S0_u[..., None]

    logging.debug("Bounds for the monotone decreasing log(-log) transform")
    loglog_E_u = np.log(-np.log(E_l.clip(.001, .999)))
    loglog_E_l = np.log(-np.log(E_u.clip(.001, .999)))

    logging.debug("Bounds for the FRT")
    FRT_op = FRT_linop(b_sph, sph)
    rhs_u = np.einsum('kl,...l->...k', FRT_op, loglog_E_u)
    rhs_l = np.einsum('kl,...l->...k', FRT_op, loglog_E_l)

    c_shift = (2 - np.log(2))/(4*np.pi)
    rhs_u, rhs_l = -rhs_l, -rhs_u
    rhs_u += c_shift
    rhs_l += c_shift

    #from tools import InverseLaplaceBeltrami, normalize_odf
    #Phi = InverseLaplaceBeltrami(sph, b_sph)
    #u = np.ones(b_sph.mdims['l_labels'])
    #normalize_odf(u, b_sph.b)
    #w = np.einsum('jk,k->j', Phi, u)
    #print (w.mean())
    #for i in range(rhs_u.shape[0]):
    #    rhs_u[i,:] = w[:]+5e-3
    #    rhs_l[i,:] = w[:]-5e-3

    return rhs_l, rhs_u