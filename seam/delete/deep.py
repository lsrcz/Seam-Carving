from numba import njit

from seam.delete.generic import deleteOneColumnWithEnergyProvided
from seam.energy.energy import computeDeepEnergy
from seam.seamfinding.gencol import generateColumn
from seam.utils import transposeImg, transposeGray


@njit(parallel=True,nogil=True)
def deepDeleteOneColumn(npimg,npgray, gdratio, heat):
    energy, lastDir = generateColumn(computeDeepEnergy(npimg,npgray, gdratio, heat))
    return deleteOneColumnWithEnergyProvided(npimg,energy,lastDir)


@njit(parallel=True,nogil=True)
def deepDeleteOneRow(npimg,npgray,gdratio,heat):
    npimgt = transposeImg(npimg)
    npgrayt = transposeGray(npgray)
    heat = transposeGray(heat)
    return transposeImg(deepDeleteOneColumn(npimgt, npgrayt,gdratio,heat))