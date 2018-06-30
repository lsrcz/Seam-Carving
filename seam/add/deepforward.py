from numba import njit

from seam.add.generic import addOneColumnWithEnergyProvided
from seam.energy.energy import computeDeepEnergy
from seam.seamfinding.gencolfwd import generateColumnForward
from seam.utils import transposeImg, transposeGray


@njit(parallel=True,nogil=True)
def deepAddOneColumnForward(npimg,npgray, npnew, pos, gdratio, heat):
    energy, lastDir = generateColumnForward(computeDeepEnergy(npimg,npgray, gdratio, heat))
    return addOneColumnWithEnergyProvided(npimg,energy,lastDir, npnew, pos)


@njit(parallel=True,nogil=True)
def deepAddOneRowForward(npimg,npgray,npnew, pos, gdratio,heat):
    npimgt = transposeImg(npimg)
    npnewt = transposeImg(npnew)
    npgrayt = transposeGray(npgray)
    heatt = transposeGray(heat)
    post = transposeGray(pos)
    ret, retnew, retpos = deepAddOneColumnForward(npimgt, npgrayt, npnewt, post, gdratio, heatt)
    return transposeImg(ret), transposeImg(retnew), transposeGray(retpos)