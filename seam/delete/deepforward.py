from numba import njit

from seam.delete.generic import deleteOneColumnWithEnergyProvided
from seam.energy.energy import computeDeepEnergy
from seam.seamfinding.gencolfwd import generateColumnForward
from seam.utils import transposeImg, transposeGray


@njit(parallel=True,nogil=True)
def deepDeleteOneColumnForward(npimg,npgray, gdratio, heat):
    energy, lastDir = generateColumnForward(computeDeepEnergy(npimg,npgray, gdratio, heat))
    return deleteOneColumnWithEnergyProvided(npimg,energy,lastDir)


@njit(parallel=True,nogil=True)
def deepDeleteOneRowForward(npimg,npgray,gdratio,heat):
    npimgt = transposeImg(npimg)
    npgrayt = transposeGray(npgray)
    heat = transposeGray(heat)
    return transposeImg(deepDeleteOneColumnForward(npimgt, npgrayt,gdratio,heat))