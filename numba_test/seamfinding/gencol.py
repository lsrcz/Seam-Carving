import numpy as np
from numba import njit


@njit(parallel=True,nogil=True)
def generateColumn(gdimg):
    x = gdimg.shape[0]
    y = gdimg.shape[1]
    lastDir = np.zeros((x,y),np.int8)
    energy = np.zeros((x,y),np.uint32)
    for i in range(y):
        energy[0,i] = gdimg[0,i]
    for i in range(1,x):
        for j in range(y):
            idx = 0
            tmp = energy[i-1,j]
            if j != 0:
                a1 = energy[i-1,j-1]
                if a1 < tmp:
                    tmp = a1
                    idx = -1
            if j != y - 1:
                a2 = energy[i-1,j+1]
                if a2 < tmp:
                    tmp = a2
                    idx = 1
            lastDir[i,j] = idx
            energy[i,j] = tmp + gdimg[i,j]
    return energy, lastDir