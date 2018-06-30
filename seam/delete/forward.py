from numba import njit

from seam.delete.generic import deleteOneColumnWithEnergyProvided
from seam.energy.energy import computeEnergy
from seam.seamfinding.gencolfwd import generateColumnForward
from seam.utils import transposeImg, transposeGray


@njit(parallel=True,nogil=True)
def deleteOneColumnForward(npimg,npgray,gdratio):
    energy, lastDir = generateColumnForward(computeEnergy(npimg, npgray, gdratio))
    return deleteOneColumnWithEnergyProvided(npimg,energy,lastDir)


@njit(parallel=True,nogil=True)
def deleteOneRowForward(npimg,npgray,gdratio):
    npimgt = transposeImg(npimg)
    npgrayt = transposeGray(npgray)
    return transposeImg(deleteOneColumnForward(npimgt, npgrayt,gdratio))