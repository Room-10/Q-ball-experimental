
import numpy as np
from dipy.reconst.shm import CsaOdfModel

def solve_shm(data):
    csamodel = CsaOdfModel(data['gtab'], 4)
    u = csamodel.fit(data['S']).odf(data['sph'])
    return np.clip(u, 0, np.max(u, -1)[..., None])
