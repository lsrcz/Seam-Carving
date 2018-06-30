from numba import njit

from seam.add.generic import addOneColumnWithEnergyProvided
from seam.energy.gradient import computeGD
from seam.seamfinding.gencol import generateColumn
from seam.utils import transposeImg, transposeGray


@njit(parallel=True,nogil=True)
def addOneColumn(npimg,npgray,npnew, pos, gdratio):
    energy, lastDir = generateColumn(computeGD(npimg))
    return addOneColumnWithEnergyProvided(npimg,energy,lastDir, npnew, pos)


@njit(parallel=True,nogil=True)
def addOneRow(npimg,npgray,npnew, pos, gdratio):
    npimgt = transposeImg(npimg)
    npnewt = transposeImg(npnew)
    npgrayt = transposeGray(npgray)
    post = transposeGray(pos)
    ret, retnew, retpos = addOneColumn(npimgt, npgrayt, npnewt, post, gdratio)
    return transposeImg(ret), transposeImg(retnew), transposeGray(retpos)