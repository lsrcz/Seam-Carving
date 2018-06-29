import numpy as np

import matplotlib.pyplot as plt
plt.ion()

from numba import jit, njit, prange

@njit(parallel=True,nogil=True)
def computeGD(npimg):
    npimg = npimg.astype(np.int16)
    x = npimg.shape[0]
    y = npimg.shape[1]
    gdsum = np.zeros((x,y),dtype=np.uint32)
    for i in prange(x):
        for j in range(y):
            #a = 0
            for k in range(3):
                pixel = npimg[i, j, k]
                for m in range(i - 1, i + 2):
                    for n in range(j - 1, j + 2):
                        if m != -1 and m != x and n != -1 and n != y and (m != i or n != j):
                            gdsum[i, j] += np.abs(npimg[m, n, k] - pixel)
                            #a += 1
            if (i == 0 or i == x - 1) and (j == 0 or j == y - 1):
                gdsum[i, j] *= 40
            elif i == 0 or i == x - 1 or j == 0 or j == y - 1:
                gdsum[i, j] *= 24
            else:
                gdsum[i, j] *= 15
            # gdsum[i, j] *= (360 // a)
    return gdsum

