from numba import njit

from seam.add.generic import addOneColumnWithEnergyProvided
from seam.energy.energy import computeEnergy
from seam.seamfinding.gencolfwd import generateColumnForward
from seam.utils import transposeImg, transposeGray


@njit(parallel=True,nogil=True)
def addOneColumnForward(npimg,npgray,npnew, pos, gdratio):
    energy, lastDir = generateColumnForward(computeEnergy(npimg, npgray, gdratio))
    return addOneColumnWithEnergyProvided(npimg,energy,lastDir, npnew, pos)


@njit(parallel=True,nogil=True)
def addOneRowForward(npimg,npgray,npnew, pos, gdratio):
    npimgt = transposeImg(npimg)
    npnewt = transposeImg(npnew)
    npgrayt = transposeGray(npgray)
    post = transposeGray(pos)
    ret, retnew, retpos = addOneColumnForward(npimgt, npgrayt, npnewt, post, gdratio)
    return transposeImg(ret), transposeImg(retnew), transposeGray(retpos)