from numba import njit

from seam.add.generic import addOneColumnWithEnergyProvided
from seam.energy.energy import computeEnergy
from seam.seamfinding.gencol import generateColumn
from seam.utils import transposeImg, transposeGray


@njit(parallel=True,nogil=True)
def addOneColumnLocal(npimg,npgray,npnew, pos, gdratio):
    energy, lastDir = generateColumn(computeEnergy(npimg, npgray, gdratio))
    return addOneColumnWithEnergyProvided(npimg,energy,lastDir, npnew, pos)


@njit(parallel=True,nogil=True)
def addOneRowLocal(npimg,npgray,npnew, pos, gdratio):
    npimgt = transposeImg(npimg)
    npnewt = transposeImg(npnew)
    npgrayt = transposeGray(npgray)
    post = transposeGray(pos)
    ret, retnew, retpos = addOneColumnLocal(npimgt, npgrayt, npnewt, post, gdratio)
    return transposeImg(ret), transposeImg(retnew), transposeGray(retpos)