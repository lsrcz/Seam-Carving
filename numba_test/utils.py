import numpy as np
from numba import jit, njit, prange

@njit(parallel=True,nogil=True)
def transpose(npimg):
    ret = np.zeros((npimg.shape[1], npimg.shape[0], 3), dtype=np.uint8)
    for i in prange(0, npimg.shape[0]):
        for j in prange(0,npimg.shape[1]):
            for k in range(3):
                ret[j,i,k] = npimg[i,j,k]
    return ret