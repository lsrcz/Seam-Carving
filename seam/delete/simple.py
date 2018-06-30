from numba import njit

from seam.delete.generic import deleteOneColumnWithEnergyProvided
from seam.energy.gradient import computeGD
from seam.seamfinding.gencol import generateColumn
from seam.utils import transposeImg, transposeGray


@njit(parallel=True,nogil=True)
def deleteOneColumn(npimg,npgray,gdratio):
    energy, lastDir = generateColumn(computeGD(npimg))
    return deleteOneColumnWithEnergyProvided(npimg,energy,lastDir)


@njit(parallel=True,nogil=True)
def deleteOneRow(npimg,npgray,gdratio):
    npimgt = transposeImg(npimg)
    npgrayt = transposeGray(npgray)
    return transposeImg(deleteOneColumn(npimgt, npgrayt,gdratio))