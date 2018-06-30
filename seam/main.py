import numpy as np
import matplotlib.pyplot as plt
from seam.delete.deep import deepDeleteOneColumn, deepDeleteOneRow
from seam.delete.deepforward import deepDeleteOneColumnForward, deepDeleteOneRowForward
from seam.energy.heatmap import heatmap
from seam.delete.forward import deleteOneColumnForward, deleteOneRowForward
from seam.delete.localent import deleteOneColumnLocal,deleteOneRowLocal
from seam.delete.simple import deleteOneRow, deleteOneColumn
from seam.add.localent import addOneRowLocal, addOneColumnLocal
from seam.add.simple import addOneColumn, addOneRow
from seam.add.forward import addOneRowForward,addOneColumnForward
from seam.add.deep import deepAddOneRow, deepAddOneColumn
from seam.add.deepforward import deepAddOneRowForward, deepAddOneColumnForward
from seam.retarget import *

gdratio = 0.1


def main():

    print('start, ', time.asctime(time.localtime(time.time())))
    print('For better performance on large images, we didn\'t compile the numba modules ahead of time, so please wait when compiling the modules')
    img = Image.open('../pics/tiger.jpg')
    #img = img.resize((480,300))
    npimg = np.array(img)
    '''
    npnew = npimg
    #pos = np.zeros((npimg.shape[0],npimg.shape[1]),dtype=np.int32) + np.arange(npimg.shape[1]).reshape((1,npimg.shape[1]))
    pos = np.zeros((npimg.shape[0],npimg.shape[1]),dtype=np.int32) + np.arange(npimg.shape[0]).reshape((npimg.shape[0],1))

    #for i in range(300):
    #    a = computeGD(npimg)
    #    b = localEntropyDP1D(npgray)
    #    print(i, time.asctime(time.localtime(time.time())))
    for i in range(200):
        npgray = npimg2npgray(npimg)
        #heat = heatmap(npimg)[0]
        #print(computeEnergy(npimg, npgray,gdratio))
        npimg, npnew, pos= addOneRow(npimg, npgray, npnew, pos, gdratio)
        print(i, time.asctime(time.localtime(time.time())))
        if i % 30 == 0:
            plt.imshow(Image.fromarray(npimg))
            plt.show()
    newimg = Image.fromarray(npnew)
    newimg.save('../out/resultsimple.jpg')
    npimg = np.array(img)
    npnew = npimg
    # pos = np.zeros((npimg.shape[0],npimg.shape[1]),dtype=np.int32) + np.arange(npimg.shape[1]).reshape((1,npimg.shape[1]))
    pos = np.zeros((npimg.shape[0], npimg.shape[1]), dtype=np.int32) + np.arange(npimg.shape[0]).reshape(
        (npimg.shape[0], 1))
    for i in range(200):
        npgray = npimg2npgray(npimg)
        npimg, npnew, pos = addOneRowForward(npimg, npgray, npnew, pos, gdratio)
        #npgray = npimg2npgray(npimg)
        #npimg = deleteOneRow(npimg, npgray,gdratio)
        print(i, time.asctime(time.localtime(time.time())))
        if i % 30 == 0:
            plt.imshow(Image.fromarray(npimg))
            plt.show()
    newimg = Image.fromarray(npnew)
    newimg.save('../out/resultforward.jpg')
    npimg = np.array(img)
    npnew = npimg
    # pos = np.zeros((npimg.shape[0],npimg.shape[1]),dtype=np.int32) + np.arange(npimg.shape[1]).reshape((1,npimg.shape[1]))
    pos = np.zeros((npimg.shape[0], npimg.shape[1]), dtype=np.int32) + np.arange(npimg.shape[0]).reshape(
        (npimg.shape[0], 1))
    for i in range(200):
        npgray = npimg2npgray(npimg)
        heat = heatmap(npimg,'densenet')[0]
        npimg, npnew, pos = deepAddOneRow(npimg, npgray, npnew, pos, gdratio, heat)
        # npgray = npimg2npgray(npimg)
        # npimg = deleteOneRow(npimg, npgray,gdratio)
        print(i, time.asctime(time.localtime(time.time())))
        if i % 30 == 0:
            plt.imshow(Image.fromarray(npimg))
            plt.show()
    newimg = Image.fromarray(npnew)
    newimg.save('../out/resultdl.jpg')
    npimg = np.array(img)
    npnew = npimg
    # pos = np.zeros((npimg.shape[0],npimg.shape[1]),dtype=np.int32) + np.arange(npimg.shape[1]).reshape((1,npimg.shape[1]))
    pos = np.zeros((npimg.shape[0], npimg.shape[1]), dtype=np.int32) + np.arange(npimg.shape[0]).reshape(
        (npimg.shape[0], 1))
    for i in range(200):
        npgray = npimg2npgray(npimg)
        heat = heatmap(npimg,'densenet')[0]
        npimg, npnew, pos = deepAddOneRowForward(npimg, npgray, npnew, pos, gdratio,heat)
        # npgray = npimg2npgray(npimg)
        # npimg = deleteOneRow(npimg, npgray,gdratio)
        print(i, time.asctime(time.localtime(time.time())))
        if i % 30 == 0:
            plt.imshow(Image.fromarray(npimg))
            plt.show()
    newimg = Image.fromarray(npnew)
    newimg.save('../out/resultdlforward.jpg')

    '''
    #for i in range(300):
    #    a = computeGD(npimg)
    #    b = localEntropyDP1D(npgray)
    #    print(i, time.asctime(time.localtime(time.time())))
    for i in range(100):
        npgray = npimg2npgray(npimg)
        #heat = heatmap(npimg)[0]
        #print(computeEnergy(npimg, npgray,gdratio))
        npimg= deleteOneColumn(npimg, npgray, gdratio)
        print(i, time.asctime(time.localtime(time.time())))
        if i % 30 == 0:
            plt.imshow(Image.fromarray(npimg))
            plt.show()
    newimg = Image.fromarray(npimg)
    newimg.save('../out/resultsimple1.jpg')
    npimg = np.array(img)
    for i in range(100):
        npgray = npimg2npgray(npimg)
        npimg = deleteOneColumnForward(npimg, npgray, gdratio)
        #npgray = npimg2npgray(npimg)
        #npimg = deleteOneRow(npimg, npgray,gdratio)
        print(i, time.asctime(time.localtime(time.time())))
        if i % 30 == 0:
            plt.imshow(Image.fromarray(npimg))
            plt.show()
    newimg = Image.fromarray(npimg)
    newimg.save('../out/resultforward1.jpg')
    npimg = np.array(img)
    for i in range(100):
        npgray = npimg2npgray(npimg)
        heat = heatmap(npimg,'densenet')[0]
        npimg = deepDeleteOneColumn(npimg, npgray, gdratio, heat)
        # npgray = npimg2npgray(npimg)
        # npimg = deleteOneRow(npimg, npgray,gdratio)
        print(i, time.asctime(time.localtime(time.time())))
        if i % 30 == 0:
            plt.imshow(Image.fromarray(npimg))
            plt.show()
    newimg = Image.fromarray(npimg)
    newimg.save('../out/resultdl1.jpg')
    npimg = np.array(img)
    for i in range(100):
        npgray = npimg2npgray(npimg)
        heat = heatmap(npimg,'densenet')[0]
        npimg = deepDeleteOneColumnForward(npimg, npgray, gdratio,heat)
        # npgray = npimg2npgray(npimg)
        # npimg = deleteOneRow(npimg, npgray,gdratio)
        print(i, time.asctime(time.localtime(time.time())))
        if i % 30 == 0:
            plt.imshow(Image.fromarray(npimg))
            plt.show()
    newimg = Image.fromarray(npimg)
    newimg.save('../out/resultdlforward1.jpg')
if __name__ == '__main__':
    main()