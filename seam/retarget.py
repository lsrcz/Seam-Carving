import time

from PIL import Image

from seam.delete.generic import deleteOneColumnWithEnergyProvided
from seam.delete.simple import deleteOneColumn, deleteOneRow
from seam.delete.forward import deleteOneColumnForward
from seam.energy.energy import computeEnergy
from seam.seamfinding.gencol import generateColumn
from seam.utils import npimg2npgray, transposeImg, transposeGray
import numpy as np

# assume r and c are smaller than the size of the image
def retargetRowfirst(npimg, gdratio, r, c):
    origr, origc, _ = npimg.shape
    for i in range(origr - r):
        npgray = npimg2npgray(npimg)
        npimg = deleteOneRow(npimg, npgray, gdratio)
        print("delete 1 row")
    for i in range(origc - c):
        npgray = npimg2npgray(npimg)
        npimg = deleteOneColumn(npimg, npgray, gdratio)
        print("delete 1 col")
    return npimg

def retargetColfirst(npimg, gdratio, r, c):
    origr, origc, _ = npimg.shape
    for i in range(origc - c):
        npgray = npimg2npgray(npimg)
        npimg = deleteOneColumn(npimg, npgray, gdratio)
        print("delete 1 col")
    for i in range(origr - r):
        npgray = npimg2npgray(npimg)
        npimg = deleteOneRow(npimg, npgray, gdratio)
        print("delete 1 row")
    return npimg

def retargetOptimal(npimg, gdratio, r, c):
    origr, origc, _ = npimg.shape
    rdelete = origr - r
    cdelete = origc - c
    T = np.zeros((rdelete + 1, cdelete + 1))
    # 1 means delete col
    I = np.zeros((rdelete + 1, cdelete + 1), dtype=np.int8)
    savedImgs = []
    savedGrays = []
    savedImgs.append(npimg)
    for i in range(rdelete + 1):
        print(i)
        for j in range(cdelete + 1):
            if i == 0:
                if j == 0:
                    savedGrays.append(npimg2npgray(npimg))
                    continue
                else:
                    npimg = savedImgs[-1]
                    npgray = npimg2npgray(savedImgs[-1])
                    energy, lastDir = generateColumn(computeEnergy(npimg, npgray, gdratio))
                    minVal = np.min(energy[-1])
                    T[i,j] = T[i,j-1] + minVal
                    I[i,j] = 1
                    newimg = deleteOneColumnWithEnergyProvided(npimg,energy,lastDir)
                    savedImgs.append(newimg)
                    savedGrays.append(npimg2npgray(newimg))
            else:
                if j == 0:
                    npimg = savedImgs[0]
                    npgray = savedGrays[0]
                    npimgt = transposeImg(npimg)
                    npgrayt = transposeGray(npgray)
                    energy, lastDir = generateColumn(computeEnergy(npimgt, npgrayt, gdratio))
                    minVal = np.min(energy[-1])
                    T[i,j] = T[i - 1,j] + minVal
                    newimgt = deleteOneColumnWithEnergyProvided(npimgt, energy, lastDir)
                    newimg = transposeImg(newimgt)
                    savedImgs[0] = newimg
                    savedGrays[0] = npimg2npgray(newimg)
                else:
                    # col
                    npimgCol = savedImgs[j - 1]
                    npgrayCol = savedGrays[j - 1]
                    energyCol, lastDirCol = generateColumn(computeEnergy(npimgCol, npgrayCol, gdratio))
                    minValCol = np.min(energyCol[-1])

                    # row
                    npimgRow = savedImgs[j]
                    npgrayRow = savedGrays[j]
                    npimgRowt = transposeImg(npimgRow)
                    npgrayRowt = transposeGray(npgrayRow)
                    energyRow, lastDirRow = generateColumn(computeEnergy(npimgRowt, npgrayRowt, gdratio))
                    minValRow = np.min(energyRow[-1])

                    if T[i - 1,j] + minValRow < T[i, j - 1] + minValCol:
                        newimgt = deleteOneColumnWithEnergyProvided(npimgRowt, energyRow, lastDirRow)
                        newimg = transposeImg(newimgt)
                    else:
                        newimg = deleteOneColumnWithEnergyProvided(npimgCol, energyCol, lastDirCol)
                        I[i, j] = 1

                    savedImgs[j] = newimg
                    savedGrays[j] = npimg2npgray(newimg)
    for i in range(rdelete + 1):
        for j in range(cdelete + 1):
            print(I[i,j],end=' ')
        print()
    return savedImgs[-1]




