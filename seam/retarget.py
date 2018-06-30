import time

from PIL import Image

from seam.delete.generic import deleteOneColumnWithEnergyProvided
from seam.delete.deep import deepDeleteOneColumn, deepDeleteOneRow
from seam.delete.deepforward import deepDeleteOneColumnForward, deepDeleteOneRowForward
from seam.delete.forward import deleteOneColumnForward, deleteOneRowForward
from seam.delete.localent import deleteOneColumnLocal,deleteOneRowLocal
from seam.delete.simple import deleteOneRow, deleteOneColumn
from seam.add.localent import addOneRowLocal, addOneColumnLocal
from seam.add.simple import addOneColumn, addOneRow
from seam.add.forward import addOneRowForward,addOneColumnForward
from seam.add.deep import deepAddOneRow, deepAddOneColumn
from seam.add.deepforward import deepAddOneRowForward, deepAddOneColumnForward
from seam.energy.heatmap import heatmap
from seam.energy.energy import computeEnergy
from seam.seamfinding.gencol import generateColumn
from seam.utils import npimg2npgray, transposeImg, transposeGray
import numpy as np

# assume r and c are smaller than the size of the image
#method: 0:without entropy 1:with entropy 2:with forward 3:with CNN 4:with CNN & forward
def retargetRowfirst(npimg, gdratio, r, c, method, model = 'squeezenet'):
    origr, origc, _ = npimg.shape
    if (origr - r > 0):
        for i in range(origr - r):
            npgray = npimg2npgray(npimg)
            if method == 0:
                npimg = deleteOneRow(npimg, npgray, gdratio)
            if method == 1:
                npimg = deleteOneRowLocal(npimg, npgray, gdratio)
            if method == 2:
                npimg = deleteOneRowForward(npimg, npgray, gdratio)
            if method == 3:
                heat = heatmap(npimg, model)[0]
                npimg = deepDeleteOneRow(npimg, npgray, gdratio, heat)
            if method == 4:
                heat = heatmap(npimg, model)[0]
                npimg = deepDeleteOneRowForward(npimg, npgray, gdratio, heat)
    else:
        npnew = npimg
        pos = np.zeros((npimg.shape[0], npimg.shape[1]), dtype=np.int32) + np.arange(npimg.shape[0]).reshape(
            (npimg.shape[0], 1))
        for i in range(r - origr):
            npgray = npimg2npgray(npimg)

            if method == 0:
                npimg, npnew, pos = addOneRow(npimg, npgray, npnew, pos, gdratio)
            if method == 1:
                npimg, npnew, pos = addOneRowLocal(npimg, npgray, npnew, pos, gdratio)
            if method == 2:
                npimg, npnew, pos = addOneRowForward(npimg, npgray, npnew, pos, gdratio)
            if method == 3:
                heat = heatmap(npimg,model)[0]
                npimg, npnew, pos = deepAddOneRow(npimg, npgray, npnew, pos, gdratio, heat)
            if method == 4:
                heat = heatmap(npimg,model)[0]
                npimg, npnew, pos = deepAddOneRowForward(npimg, npgray, npnew, pos, gdratio, heat)
        npimg, npnew = npnew, npimg
    if (origc - c > 0):
        for i in range(origc - c):
            npgray = npimg2npgray(npimg)
            if method == 0:
                npimg = deleteOneColumn(npimg, npgray, gdratio)
            if method == 1:
                npimg = deleteOneColumnLocal(npimg, npgray, gdratio)
            if method == 2:
                npimg = deleteOneColumnForward(npimg, npgray, gdratio)
            if method == 3:
                heat = heatmap(npimg,model)[0]
                npimg = deepDeleteOneColumn(npimg, npgray, gdratio, heat)
            if method == 4:
                heat = heatmap(npimg,model)[0]
                npimg = deepDeleteOneColumnForward(npimg, npgray, gdratio, heat)
    else:
        pos = np.zeros((npimg.shape[0], npimg.shape[1]), dtype=np.int32) + np.arange(npimg.shape[1]).reshape(
            (1, npimg.shape[1]))
        npnew = npimg
        for i in range(c - origc):
            npgray = npimg2npgray(npimg)
            if method == 0:
                npimg, npnew, pos = addOneColumn(npimg, npgray, npnew, pos, gdratio)
            if method == 1:
                npimg, npnew, pos = addOneColumnLocal(npimg, npgray, npnew, pos, gdratio)
            if method == 2:
                npimg, npnew, pos = addOneColumnForward(npimg, npgray, npnew, pos, gdratio)
            if method == 3:
                heat = heatmap(npimg,model)[0]
                npimg, npnew, pos = deepAddOneColumn(npimg, npgray, npnew, pos, gdratio, heat)
            if method == 4:
                heat = heatmap(npimg,model)[0]
                npimg, npnew, pos = deepAddOneColumnForward(npimg, npgray, npnew, pos, gdratio, heat)
        npimg, npnew = npnew, npimg
    return npimg

def retargetColfirst(npimg, gdratio, r, c, method, model = 'squeezenet'):
    origr, origc, _ = npimg.shape
    print(origc,c,origc-c)
    if (origc - c > 0):
        for i in range(origc - c):
            npgray = npimg2npgray(npimg)
            if method == 0:
                npimg = deleteOneColumn(npimg, npgray, gdratio)
            if method == 1:
                npimg = deleteOneColumnLocal(npimg, npgray, gdratio)
            if method == 2:
                npimg = deleteOneColumnForward(npimg, npgray, gdratio)
            if method == 3:
                heat = heatmap(npimg, model)[0]
                npimg = deepDeleteOneColumn(npimg, npgray, gdratio, heat)
            if method == 4:
                heat = heatmap(npimg, model)[0]
                npimg = deepDeleteOneColumnForward(npimg, npgray, gdratio, heat)
    else:
        npnew = npimg
        pos = np.zeros((npimg.shape[0], npimg.shape[1]), dtype=np.int32) + np.arange(npimg.shape[1]).reshape(
            (1, npimg.shape[1]))
        for i in range(c - origc):
            npgray = npimg2npgray(npimg)

            if method == 0:
                npimg, npnew, pos = addOneColumn(npimg, npgray, npnew, pos, gdratio)
            if method == 1:
                npimg, npnew, pos = addOneColumnLocal(npimg, npgray, npnew, pos, gdratio)
            if method == 2:
                npimg, npnew, pos = addOneColumnForward(npimg, npgray, npnew, pos, gdratio)
            if method == 3:
                heat = heatmap(npimg, model)[0]
                npimg, npnew, pos = deepAddOneColumn(npimg, npgray, npnew, pos, gdratio, heat)
            if method == 4:
                heat = heatmap(npimg, model)[0]
                print(npimg.shape)
                npimg, npnew, pos = deepAddOneColumnForward(npimg, npgray, npnew, pos, gdratio, heat)
                print(npimg.shape)
        npimg, npnew = npnew, npimg
    if (origr - r > 0):
        for i in range(origr - r):
            npgray = npimg2npgray(npimg)
            if method == 0:
                npimg = deleteOneRow(npimg, npgray, gdratio)
            if method == 1:
                npimg = deleteOneRowLocal(npimg, npgray, gdratio)
            if method == 2:
                npimg = deleteOneRowForward(npimg, npgray, gdratio)
            if method == 3:
                heat = heatmap(npimg, model)[0]
                npimg = deepDeleteOneRow(npimg, npgray, gdratio, heat)
            if method == 4:
                heat = heatmap(npimg, model)[0]
                npimg = deepDeleteOneRowForward(npimg, npgray, gdratio, heat)
    else:
        npnew = npimg
        pos = np.zeros((npimg.shape[0], npimg.shape[1]), dtype=np.int32) + np.arange(npimg.shape[0]).reshape(
            (npimg.shape[0], 1))
        for i in range(r - origr):
            npgray = npimg2npgray(npimg)

            if method == 0:
                npimg, npnew, pos = addOneRow(npimg, npgray, npnew, pos, gdratio)
            if method == 1:
                npimg, npnew, pos = addOneRowLocal(npimg, npgray, npnew, pos, gdratio)
            if method == 2:
                npimg, npnew, pos = addOneRowForward(npimg, npgray, npnew, pos, gdratio)
            if method == 3:
                heat = heatmap(npimg,model)[0]
                npimg, npnew, pos = deepAddOneRow(npimg, npgray, npnew, pos, gdratio, heat)
            if method == 4:
                heat = heatmap(npimg,model)[0]
                npimg, npnew, pos = deepAddOneRowForward(npimg, npgray, npnew, pos, gdratio, heat)
        npimg, npnew = npnew, npimg
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




