from numba import njit

from seam.delete.generic import deleteOneColumnWithEnergyProvided
from seam.energy.energy import computeDeepEnergy
from seam.seamfinding.gencol import generateColumn
from seam.utils import transposeImg, transposeGray


@njit(parallel=True,nogil=True)
def deepdeleteOneColumn(npimg,npgray, gdratio, heat):
    energy, lastDir = generateColumn(computeDeepEnergy(npimg,npgray, gdratio, heat))
    return deleteOneColumnWithEnergyProvided(npimg,energy,lastDir)


@njit(parallel=True,nogil=True)
def deepdeleteOneRow(npimg,npgray,gdratio):
    npimgt = transposeImg(npimg)
    npgrayt = transposeGray(npgray)
    return transposeImg(deepdeleteOneColumn(npimgt, npgrayt,gdratio))