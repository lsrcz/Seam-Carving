import numpy as np
from numba import njit, prange

from seam.utils import symmetricPadding2D4


def padding(img):
    return img.padding(img,((4,4),(4,4)),'symmatric')


@njit(parallel=True,nogil=True)
def localEntropy(npgray):
    entropy = np.zeros((npgray.shape[0], npgray.shape[1]))
    npgray = symmetricPadding2D4(npgray)
    for i in prange(4,npgray.shape[0] - 4):
        for j in range(4, npgray.shape[1] - 4):
            Hist=np.zeros(256,dtype=np.uint8)
            for p in range(i - 4, i + 5):
                for q in range(j - 4, j + 5):
                    Hist[npgray[p,q]] += 1
            for k in range(256):
                if Hist[k]!=0:
                    p1 = Hist[k] / 81
                    entropy[i-4,j-4]+=p1*np.log(1/p1)
    return entropy


@njit(parallel=True,nogil=True)
def localEntropyDP1D(npgray):
    entropy = np.zeros((npgray.shape[0], npgray.shape[1]),np.float32)
    npgray = symmetricPadding2D4(npgray)
    precompute = np.zeros(82,dtype=np.float32)
    for i in range(1,81):
        precompute[i] = (i/81) * np.log(81/i)
    for i in prange(4,npgray.shape[0] - 4):
        # initial
        Hist = np.zeros(256, dtype=np.uint8)
        for p in range(i - 4, i + 5):
            for q in range(0, 9):
                Hist[npgray[p, q]] += 1
        for k in range(256):
            if Hist[k] != 0:
                #p1 = Hist[k] / 81
                entropy[i - 4, 0] += precompute[Hist[k]]#p1 * np.log(1 / p1)
        for j in range(5, npgray.shape[1] - 4):
            for p in range(i - 4, i + 5):
                Hist[npgray[p, j - 5]] -= 1
                Hist[npgray[p, j + 4]] += 1
            for k in range(256):
                if Hist[k]!=0:
                    #p1 = Hist[k] / 81
                    #entropy[i-4,j-4]+=p1*np.log(1/p1)
                    entropy[i-4,j-4] += precompute[Hist[k]]
    return entropy


@njit(parallel=True,nogil=True)
def localEntropyDP1DVert(npgray):
    entropy = np.zeros((npgray.shape[0], npgray.shape[1]))
    npgray = symmetricPadding2D4(npgray)
    precompute = np.zeros(82, dtype=np.float32)
    for i in range(1, 81):
        precompute[i] = (i / 81) * np.log(81 / i)
    for j in prange(4,npgray.shape[1] - 4):
        # initial
        Hist = np.zeros(256, dtype=np.uint8)
        for p in range(0, 9):
            for q in range(j - 4, j + 5):
                Hist[npgray[p, q]] += 1
        for k in range(256):
            if Hist[k] != 0:
                #p1 = Hist[k] / 81
                #entropy[0, j - 4] += p1 * np.log(1 / p1)
                entropy[0, j - 4] += precompute[Hist[k]]
        for i in range(5, npgray.shape[0] - 4):
            for q in range(j - 4, j + 5):
                Hist[npgray[i - 5, q]] -= 1
                Hist[npgray[i + 4, q]] += 1
            for k in range(256):
                if Hist[k]!=0:
                    #p1 = Hist[k] / 81
                    #entropy[i-4,j-4]+=p1*np.log(1/p1)
                    entropy[i - 4, j - 4] += precompute[Hist[k]]
    return entropy