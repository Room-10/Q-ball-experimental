
import pickle
import numpy as np
from numpy.linalg import norm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

def normalize_odf(odf, vol):
    odf_flat = odf.reshape(odf.shape[0], -1)
    odf_sum = np.einsum('k,ki->i', vol, odf_flat)
    odf_flat[:] = np.einsum('i,ki->ki', 1.0/odf_sum, odf_flat)

def coords_cartesian(r_phi_theta):
    """ Convert spherical coordinates to cartesian coordinates

    Args:
        r_phi_theta : numpy array of shape (3,), (r, phi, theta) such that
                      phi is azimuth [0, 2pi) and theta is inclination [0, pi].
    Returns:
        numpy array of shape (3,) containing xyz coordinates
    """
    r, phi, theta = r_phi_theta
    return np.array([
        r*np.sin(theta)*np.cos(phi),
        r*np.sin(theta)*np.sin(phi),
        r*np.cos(theta)
    ])

def coords_spherical(xyz):
    """ Convert cartesian coordinates to spherical coordinates

    Args:
        xyz : numpy array of shape (3,) containing xyz coordinates
    Returns:
        numpy array of shape (3,) containing (r, phi, theta) such that
        phi is azimuth [0, 2pi) and theta is inclination [0, pi].
    """
    xy = xyz[0]**2 + xyz[1]**2
    r = np.sqrt(xy + xyz[2]**2)
    theta = np.arccos(xyz[2]/r) # inclination
    phi = np.arctan2(xyz[1], xyz[0]) # azimuth
    phi = phi if phi > 0 else phi + 2*np.pi
    return np.array([r, phi, theta])

def normalize(u, p=2, thresh=0.0):
    """ Normalizes u along the columns with norm p.

    If  |u| <= thresh, 0 is returned (this mimicks the sign function).
    Not optimized for speed yet.
    """
    if u.ndim == 1:
        u = u.reshape(-1,1)
    ns = norm(u, ord=p, axis=0)
    fact = np.zeros(ns.size);
    fact[ns > thresh] = 1.0/ns[ns > thresh]
    return  u * fact # uses broadcasting

def create_ONB(v1):
    """ Compute v2, v3 such that (v1,v2,v3) is an orthonormal basis of R^3

    Args:
        v1 : numpy array of shape (3,)
    Returns:
        v2 : numpy array of shape (3,)
        v3 : numpy array of shape (3,)
    """
    v2 = np.zeros(3)
    m = np.argmin(np.abs(v1))
    v2[(m+0)%3] = 0
    v2[(m+1)%3] = -v1[(m+2)%3]
    v2[(m+2)%3] = v1[(m+1)%3]
    v2 *= 1/np.sqrt(1-v1[m]**2)
    v3 = np.cross(v1, v2)
    return v2, v3

def FRT_compute(f, p, n=20):
    """ Computes the Funk-Radon transform of f evaluated at p

    Args:
        f : a real- or complex-valued function on the Sphere
        p : a 3d-vector of norm 1
        n : resolution of the line integral in the definition of the FRT

    Returns:
        (Approximate) value of FRT(f) at p.
    """
    delta_th = 2*np.pi/n
    v1, v2 = create_ONB(p)

    # integrate over great circle orthogonal to p
    theta = 0.0
    result = 0.0*f(np.array([1,0,0]))
    for k in range(n):
        theta += delta_th
        v_th = np.cos(theta)*v1 + np.sin(theta)*v2
        assert np.abs(np.einsum('i,i->', v_th, p)) < 1e-10
        result += delta_th*f(v_th)
    return result

def FRT_linop(sph1, sph2, n=20):
    """ Gives the linear operator mapping a function on sph1 to its Funk-Radon
        transform, defined on sph2.

    Args:
        sph1 : Instance of Sphere
        sph2 : Instance of Sphere
        n : resolution of the line integral in the definition of the FRT

    Returns:
        numpy array of shape (l2, l1) where l_i are the number of vertices of
        the triangulations on sph_i
    """
    l1, l2 = sph1.mdims['l_labels'], sph2.mdims['l_labels']
    A = np.zeros((l2, l1))
    E = np.eye(l1)
    pickle_file = "cache/sphere-{}-{}-{}.pickle".format(l1, l2, n)
    try:
        A[:] = pickle.load(open(pickle_file, 'rb'))
    except:
        print("No cached FRT({}-{}-{}). Preparing...".format(l1, l2, n))
        for j in range(l2):
            f = lambda x: sph1.interpolate(E, x)
            A[j,:] = FRT_compute(f, sph2.v[:,j], n)
        pickle.dump(A, open(pickle_file, 'wb'))
    return A

def InverseLaplaceBeltrami(sph1, sph2):
    # U(x,x0) = -1/(4*np.pi) * np.log(np.abs(1 - <x,x_0>))
    U = np.abs(1 - np.einsum('ij,ik->jk', sph1.v, sph2.v))
    U = np.log(np.fmax(U, np.spacing(1)))
    U = -1/(4*np.pi) * np.einsum('jk,k->jk', U, sph2.b)
    return U

def plot_mesh3(ax, vecs, tris):
    """ Plots a surface according to a given triangulation.

    Args:
        ax : Instance of mpl_toolkits.mplot3d.Axes3d
        vecs : numpy array of shape (3, l_vecs) containing the grid points of the
               surface
        tris : numpy array of shape (3, l_vecs). Each column contains the three
               indices (wrt. `vecs`) of a triangle's vertices.

    Test code:
        import matplotlib.pyplot as plt
        fig = plt.figure(facecolor="white")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0.1)
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.set_xlim((-1.1,1.1))
        ax.set_ylim((-1.1,1.1))
        ax.set_zlim((-1.1,1.1))
        ax.view_init(azim=30)
        ax.scatter(1.01*sv.points[:,0], 1.01*sv.points[:,1], 1.01*sv.points[:,2])
        plot_mesh3(ax, sv.points.T, sv._tri.simplices.T)
        plt.show()
    """
    vx = vecs[0]
    vy = vecs[1]
    vz = vecs[2]

    tmp = np.tile(tris, (2,1))

    verts = np.dstack((vx[tmp].T, vy[tmp].T, vz[tmp].T))
    green = np.array([[10., 250., 31., 200.]])/255.0
    ax.add_collection3d(
        Poly3DCollection(verts, facecolors='w', linewidth=1, alpha=0.5)
    )
    ax.add_collection3d(
        Line3DCollection(verts, colors='k', linewidths=0.2, linestyle=':')
    )
    #for k in range(vecs.shape[1]):
    #    ax.text(1.1*vecs[0,k], 1.1*vecs[1,k], 1.1*vecs[2,k], str(k))
