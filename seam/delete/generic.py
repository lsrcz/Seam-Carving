from numba import njit, prange
import numpy as np

@njit(parallel=True,nogil=True)
def deleteOneColumnWithEnergyProvided(npimg,energy,lastDir):
    lastArray = np.zeros((npimg.shape[0]), dtype=np.int16)
    lastArray[-1] = np.argmin(energy[-1])
    for i in range(npimg.shape[0] - 1, 0, -1):
        lastArray[i - 1] = lastArray[i] + lastDir[i, lastArray[i]]
    ret = np.zeros((npimg.shape[0], npimg.shape[1] - 1, 3), dtype=np.uint8)
    for i in prange(0, npimg.shape[0]):
        for j in range(0, lastArray[i]):
            ret[i, j, 0] = npimg[i, j, 0]
            ret[i, j, 1] = npimg[i, j, 1]
            ret[i, j, 2] = npimg[i, j, 2]
        for j in range(lastArray[i], ret.shape[1]):
            ret[i, j, 0] = npimg[i, j + 1, 0]
            ret[i, j, 1] = npimg[i, j + 1, 1]
            ret[i, j, 2] = npimg[i, j + 1, 2]
    return ret