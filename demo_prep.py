
import sys

import logging
logging.basicConfig(
    stream=sys.stdout,
    format="[%(relativeCreated) 8d] %(message)s",
    level=logging.DEBUG
)

import numpy as np
from dipy.data import fetch_stanford_hardi, read_stanford_hardi
import dipy.core.sphere
from dipy.segment.mask import median_otsu
from dipy.viz import fvtk

from manifold_sphere import load_sphere
from solve_shm import solve_shm
from solve_cvx import solve_cvx

logging.debug("Loading realworld data ...")
fetch_stanford_hardi()
img, gtab = read_stanford_hardi()
S_data = img.get_data()

assert(len(S_data.shape) == 4)
assert(gtab.bvals is not None)
assert(gtab.bvecs.shape[1] == 3)
assert(S_data.shape[-1] == gtab.bvals.size)

b_vecs = gtab.bvecs[gtab.bvals > 0,...].T
b_sph = load_sphere(vecs=b_vecs)
qball_sphere = dipy.core.sphere.Sphere(xyz=b_sph.v.T, faces=b_sph.faces.T)

maskdata, mask = median_otsu(S_data, 3, 1, True,
                             vol_idx=range(10, 50), dilate=2)
S_data = maskdata[20:30, 61, 28]

#u = solve_shm({ 'S': S_data, 'gtab': gtab, 'sph': qball_sphere })
u = solve_cvx({ 'S': S_data, 'gtab': gtab, 'sph': b_sph })

plotdata = u.copy()
if len(plotdata.shape) < 3:
    plotdata = plotdata[:,:,np.newaxis,np.newaxis].transpose(0,2,3,1)
else:
    plotdata = plotdata[:,:,:,np.newaxis].transpose(0,1,3,2)
plot_scale = 2.4
plot_norm = True

r = fvtk.ren()
fvtk.add(r, fvtk.sphere_funcs(plotdata, qball_sphere, colormap='jet',
                              norm=plot_norm, scale=plot_scale))
fvtk.show(r, size=(1024, 768))
