from seam.seamfinding.gencol import generateColumn
from seam.utils import transposeImg, transposeGray
from seam.energy.energy import computeEnergy
import numpy as np
from numba import jit, njit, prange


@njit(parallel=True,nogil=True)
def addOneColumn(npimg,npgray,npnew,pos,gdratio):
    energy, lastDir = generateColumn(computeEnergy(npimg, npgray,gdratio))

    lastArray = np.zeros((npimg.shape[0]),dtype=np.int16)
    lastArray[-1] = np.argmin(energy[-1])
    for i in range(npimg.shape[0] - 1, 0, -1):
        lastArray[i - 1] = lastArray[i] + lastDir[i,lastArray[i]]
    retpos = np.zeros((pos.shape[0], pos.shape[1] - 1),dtype=np.uint32)
    retnew = np.zeros((npnew.shape[0], npnew.shape[1] + 1, 3), dtype=np.uint8)
    ret = np.zeros((npimg.shape[0], npimg.shape[1] - 1, 3), dtype=np.uint8)
    for i in prange(0, npimg.shape[0]):
        for j in prange(0,lastArray[i]):
            retpos[i,j] = pos[i,j]
            for k in range(3):
                ret[i,j,k] = npimg[i,j,k]
        for j in prange(lastArray[i], ret.shape[1]):
            retpos[i,j] = pos[i,j + 1]+1
            for k in range(3):
                ret[i,j,k] = npimg[i,j+1,k]
    for i in prange(0, retnew.shape[0]):
        for j in prange(0,pos[i,lastArray[i]] + 1):
            for k in range(3):
                retnew[i,j,k] = npnew[i,j,k]
        retnew[i,pos[i,lastArray[i]] + 1,:] = npnew[i,pos[i,lastArray[i]],:]
        for j in prange(pos[i,lastArray[i]] + 2, retnew.shape[1]):
            for k in range(3):
                retnew[i,j,k] = npnew[i,j-1,k]
    return ret,retnew,retpos

def addOneRow(npimg,npgray,gdratio):
    npimgt = transposeImg(npimg)
    npgrayt = transposeGray(npgray)
    return transposeImg(addOneColumn(npimgt, npgrayt,gdratio))