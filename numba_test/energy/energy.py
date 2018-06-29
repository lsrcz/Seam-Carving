import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from numba import njit, prange

from numba_test.config import NEED_DISPLAY
from numba_test.energy.entropy import localEntropyDP1D
from numba_test.energy.gradient import computeGD


@njit(parallel=True,nogil=True)
def computeEnergy(npimg, npgray, gdratio):

    gd = computeGD(npimg).astype(np.float32)
    entropy = localEntropyDP1D(npgray)

    for i in prange(gd.shape[0]):
        for j in range(gd.shape[1]):
            gd[i,j] = gd[i,j] * (1e-3) * gdratio + entropy[i,j] * (1-gdratio)
    return gd


@njit(parallel=True,nogil=True)
def computeDeepEnergy(npimg, npgray, gdratio, heat):

    gd = computeGD(npimg).astype(np.float32)
    entropy = localEntropyDP1D(npgray)

    for i in prange(gd.shape[0]):
        for j in range(gd.shape[1]):
            gd[i,j] = gd[i,j] * (1e-3) * gdratio + entropy[i,j] * (1-gdratio) + heat[i,j] * 10
    return gd


def showEnergyImg(gdimg):
    gdimg = gdimg.astype(np.float32)
    if NEED_DISPLAY:
        plt.figure(figsize=(19.2, 10.8))
        plt.title("gradient image")
        plt.imshow(Image.fromarray((gdimg / gdimg.max() * 255).astype(np.uint8)))
        plt.show()