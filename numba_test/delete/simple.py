from numba import njit

from numba_test.delete.generic import deleteOneColumnWithEnergyProvided
from numba_test.energy.energy import computeEnergy
from numba_test.seamfinding.gencol import generateColumn
from numba_test.utils import transposeImg, transposeGray


@njit(parallel=True,nogil=True)
def deleteOneColumn(npimg,npgray,gdratio):
    energy, lastDir = generateColumn(computeEnergy(npimg, npgray, gdratio))
    return deleteOneColumnWithEnergyProvided(npimg,energy,lastDir)


@njit(parallel=True,nogil=True)
def deleteOneRow(npimg,npgray,gdratio):
    npimgt = transposeImg(npimg)
    npgrayt = transposeGray(npgray)
    return transposeImg(deleteOneColumn(npimgt, npgrayt,gdratio))