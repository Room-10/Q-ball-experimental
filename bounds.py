
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

    assert((S_u >= S_l).all())
    assert((S0_u >= S0_l).all())

    logging.debug("Bounds for normalized data E = S/S_0 ...")
    E_l = S_l/S0_u[..., None]
    E_u = S_u/S0_l[..., None]
    assert((E_u >= E_l).all())

#    E_l[:] = 0.0
#    E_u[:] = 1.0

    logging.debug("Bounds for the monotone decreasing log(-log) transform")
    loglog_E_l = np.log(-np.log(E_u.clip(0.001, 0.999)))
    loglog_E_u = np.log(-np.log(E_l.clip(0.001, 0.999)))
    assert((loglog_E_u >= loglog_E_l).all())

    logging.debug("Bounds for the FRT")
    FRT_op = FRT_linop(b_sph, sph)
    rhs_l = np.einsum('kl,...l->...k', FRT_op, loglog_E_l)
    rhs_u = np.einsum('kl,...l->...k', FRT_op, loglog_E_u)
    assert((rhs_u >= rhs_l).all())

    c_shift = (2 - np.log(2))/(4*np.pi)
    rhs_u, rhs_l = -rhs_l, -rhs_u
    rhs_l += c_shift
    rhs_u += c_shift

    from tools import InverseLaplaceBeltrami, normalize_odf
    Phi = InverseLaplaceBeltrami(sph, b_sph)

    u_true = None
    synthetic = False
    if synthetic:
        example = 2
        if example == 1:
            # uniform
             u = np.ones(b_sph.mdims['l_labels'])
        elif example == 2:
            # bimodals
            from dipy.sims.voxel import multi_tensor_odf
            verts = b_sph.v.T
            directions = [30, 90]

            val_base = 1e-6*200
            vals = [8.5*val_base, val_base, val_base]
            vecs = [
                np.array([np.sin(phi*np.pi/180.0), np.cos(phi*np.pi/180.0), 0])
                for phi in directions
            ]
            u = multi_tensor_odf(verts, np.array((vals, vals)), vecs, [50, 50])

        normalize_odf(u, b_sph.b)
        w = np.einsum('jk,k->j', Phi, u)
        #    print (w.mean())
        c1 = 0.05
        for i in range(rhs_u.shape[0]):
            rhs_l[i,:] = w[:]*(1.0-c1)
            rhs_u[i,:] = w[:]*(1.0+c1)
        u_true = np.tile(u, (rhs_u.shape[0],1)).T

    assert((rhs_u >= rhs_l).all())

    return rhs_l, rhs_u, u_true
