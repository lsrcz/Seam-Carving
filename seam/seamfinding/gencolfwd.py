import numpy as np
from numba import njit, jit, prange


@njit(parallel=True,nogil=True)
def generateColumnForward(gdimg):
    forward = forwardEnergy(gdimg)
    x = gdimg.shape[0]
    y = gdimg.shape[1]
    lastDir = np.zeros((x,y),np.int8)
    energy = np.zeros((x,y),np.uint32)
    for i in range(y):
        energy[0,i] = gdimg[0,i]
    for i in range(1,x):
        for j in range(y):
            idx = 0
            tmp = energy[i-1,j] + forward[i-1,j,1]
            if j != 0:
                a1 = energy[i-1,j-1] + forward[i-1,j-1,0]
                if a1 < tmp:
                    tmp = a1
                    idx = -1
            if j != y - 1:
                a2 = energy[i-1,j+1] + forward[i-1,j+1,2]
                if a2 < tmp:
                    tmp = a2
                    idx = 1
            lastDir[i,j] = idx
            energy[i,j] = tmp + gdimg[i,j]
    return energy, lastDir


@jit(nopython=True, parallel=True)
def forwardEnergy(I):
    n = I.shape[0]
    m = I.shape[1]
    ret = np.zeros((n, m, 3))
    for i in prange(n):
        for j in prange(m):
            if j < m-1:
                x = I[i, j+1]
            else:
                x = I[i, j]
            if j > 0:
                y = I[i, j-1]
            else:
                y = I[i, j]
            if i > 0:
                z = I[i-1, j]
            else:
                z = I[i, j]
            ret[i, j, 0] = np.abs(x - y) + np.abs(z - y)
            ret[i, j, 1] = np.abs(x - y)
            ret[i, j, 2] = np.abs(x - y) + np.abs(z - x)
    return ret