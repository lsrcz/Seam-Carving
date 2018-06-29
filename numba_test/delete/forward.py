from numba import njit

from numba_test.delete.generic import deleteOneColumnWithEnergyProvided
from numba_test.energy.energy import computeEnergy
from numba_test.seamfinding.gencolfwd import generateColumnForward
from numba_test.utils import transposeImg, transposeGray


@njit(parallel=True,nogil=True)
def deleteOneColumnForward(npimg,npgray,gdratio):
    energy, lastDir = generateColumnForward(computeEnergy(npimg, npgray, gdratio))
    return deleteOneColumnWithEnergyProvided(npimg,energy,lastDir)


@njit(parallel=True,nogil=True)
def deleteOneRowForward(npimg,npgray,gdratio):
    npimgt = transposeImg(npimg)
    npgrayt = transposeGray(npgray)
    return transposeImg(deleteOneColumnForward(npimgt, npgrayt,gdratio))