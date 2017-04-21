
import logging

import numpy as np
import cvxpy as cvx

from manifold_sphere import load_sphere
from bounds import compute_bounds
from tools import InverseLaplaceBeltrami
from staggered_diff import staggered_diff_avgskips

def solve_cvx(data, lbd=1.0, refinement=2):
    b_vecs = data['gtab'].bvecs[data['gtab'].bvals > 0,...].T
    b_sph = data['sph']
    sph = load_sphere(refinement=refinement)

    rhs_l, rhs_u = compute_bounds(data, sph)

    imagedims = data['S'].shape[:-1]
    d_image = len(imagedims)
    n_image = np.prod(imagedims)
    s_manifold = sph.mdims['s_manifold']
    r_points = sph.mdims['r_points']

    logging.debug("Preparing inverse laplacian ...")
    Phi = InverseLaplaceBeltrami(sph, b_sph)

    rhs_l = rhs_l.reshape(n_image, -1)
    rhs_u = rhs_u.reshape(n_image, -1)

    logging.info("Solving (mf: s={s}, l={l}, m={m}; img: {imagedims}; " \
        "lambda={lbd:.3g}, using CVXPY)...".format(
        lbd=lbd,
        m=b_sph.mdims['m_gradients'],
        s=s_manifold,
        l=b_sph.mdims['l_labels'],
        imagedims="x".join(map(str,imagedims))
    ))

    p  = cvxVariable(b_sph.mdims['l_labels'], d_image, n_image)
    g  = cvxVariable(n_image, b_sph.mdims['m_gradients'], d_image, s_manifold)
    q0  = cvxVariable(n_image)
    q1 = cvxVariable(n_image, sph.mdims['l_labels'])
    q2 = cvxVariable(n_image, sph.mdims['l_labels'])

    obj = cvx.Maximize(
          cvx.vec(q1).T*cvx.vec(rhs_l)
        - cvx.vec(q2).T*cvx.vec(rhs_u)
        - cvx.sum_entries(q0)
    )

    div_op = sparse_div_op(imagedims)
    lbd = 1.0

    logging.debug("CVXPY: constraint setup...")
    constraints = []
    for k in range(sph.mdims['l_labels']):
        for i in range(n_image):
            constraints.append(q1[i,k] >= 0)
            constraints.append(q2[i,k] >= 0)
    for i in range(n_image):
        for j in range(b_sph.mdims['m_gradients']):
            constraints.append(cvx.norm(g[i][j], 2) <= lbd)

    logging.debug("CVXPY: constraint setup for variable w...")
    w_constr = []
    for j in range(b_sph.mdims['m_gradients']):
        Aj = b_sph.A[j,:,:]
        Bj = b_sph.B[j,:,:]
        Pj = b_sph.P[j,:]
        for i in range(n_image):
            for t in range(d_image):
                w_constr.append(
                    Aj*g[i][j][t,:].T == sum([Bj[:,m]*p[Pj[m]][t,i] for m in range(r_points)])
                )
    constraints += w_constr

    logging.debug("CVXPY: constraint setup for variable u...")
    u_constr = []
    for k in range(b_sph.mdims['l_labels']):
        for i in range(n_image):
            u_constr.append(
                   b_sph.b[k]*(q0[i] - cvxOp(div_op, p[k], i))
                 - cvx.vec(Phi[:,k]).T*cvx.vec((q1 - q2)[i,:])
                >= 0
            )
    constraints += u_constr

    logging.info("CVXPY: problem formulation...")
    prob = cvx.Problem(obj, constraints)
    logging.info("CVXPY: check problem is DCP...")
    logging.info(prob.is_dcp())
    logging.info("CVXPY: solving...")
    prob.solve(solver="SCS", verbose=True)

    u = np.zeros((b_sph.mdims['l_labels'], n_image), order='C')
    for k in range(b_sph.mdims['l_labels']):
        for i in range(n_image):
            u[k,i] = u_constr[k*n_image+i].dual_value

    return u.T.reshape(imagedims + (-1,))

def cvxVariable(*args):
    """ Create a multidimensional CVXPY variable

    Args:
        args : list of integers (dimensions)
    Returns:
        A multidimensional variable (using python dictionaries if more than two
        dimensions are needed)
    """
    if len(args) <= 2:
        return cvx.Variable(*args)
    else:
        var = {}
        for i in range(args[0]):
            var[i] = cvxVariable(*args[1:])
        return var

def sparse_div_op(dims):
    """ Sparse linear operator for divergence with dirichlet boundary

    Args:
        dims : dimensions of the image domain
    Returns:
        Sparse linear operator (can be used with cvxOp)
    """
    d_image = len(dims)
    n_image = np.prod(dims)
    avgskips = staggered_diff_avgskips(dims)
    navgskips =  1 << (d_image - 1)

    skips = (1,)
    for t in range(1,d_image):
        skips += (skips[-1]*dims[d_image-t],)

    op = [[] for i in range(n_image)]
    coords = np.zeros(d_image, dtype=np.int64)

    for t in range(d_image):
        coords *= 0
        for i in range(n_image):
            # ignore boundary points
            in_range = True
            for dc in reversed(range(d_image)):
                if coords[dc] >= dims[dc] - 1:
                    in_range = False
                    break

            if in_range:
                for avgskip in avgskips[t]:
                    base = i + avgskip
                    op[base + skips[t]].append(((t,i), -1.0/navgskips))
                    op[base].append(((t,i), 1.0/navgskips))

            # advance coordinates
            for dd in reversed(range(d_image)):
                coords[dd] += 1
                if coords[dd] >= dims[dd]:
                    coords[dd] = 0
                else:
                    break

    return op

def cvxOp(A, x, i):
    """ CVXPY expression for the application of A to x, evaluated at i

    Args:
        A : sparse representation of a linear operator
        x : variable whose size matches the requirements of A
        i : point at which to evaluate
    Returns:
        CVXPY expression for the application of A to x, evaluated at i
    """
    return sum([fact*x[coord] for coord,fact in A[i]])