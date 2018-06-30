from numba import njit

from seam.delete.generic import deleteOneColumnWithEnergyProvided
from seam.energy.energy import computeEnergy
from seam.seamfinding.gencol import generateColumn
from seam.utils import transposeImg, transposeGray


@njit(parallel=True,nogil=True)
def deleteOneColumn(npimg,npgray,gdratio):
    energy, lastDir = generateColumn(computeEnergy(npimg, npgray, gdratio))
    return deleteOneColumnWithEnergyProvided(npimg,energy,lastDir)


@njit(parallel=True,nogil=True)
def deleteOneRow(npimg,npgray,gdratio):
    npimgt = transposeImg(npimg)
    npgrayt = transposeGray(npgray)
    return transposeImg(deleteOneColumn(npimgt, npgrayt,gdratio))