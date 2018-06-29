from numba import njit

from numba_test.delete.generic import deleteOneColumnWithEnergyProvided
from numba_test.energy.energy import computeDeepEnergy
from numba_test.seamfinding.gencol import generateColumn
from numba_test.utils import transposeImg, transposeGray


@njit(parallel=True,nogil=True)
def deepdeleteOneColumn(npimg,npgray, gdratio, heat):
    energy, lastDir = generateColumn(computeDeepEnergy(npimg,npgray, gdratio, heat))
    return deleteOneColumnWithEnergyProvided(npimg,energy,lastDir)


@njit(parallel=True,nogil=True)
def deepdeleteOneRow(npimg,npgray,gdratio):
    npimgt = transposeImg(npimg)
    npgrayt = transposeGray(npgray)
    return transposeImg(deepdeleteOneColumn(npimgt, npgrayt,gdratio))