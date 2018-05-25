from numba_test.config import *
import time

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from numba_test.deleting import deleteOneRow, deleteOneColumn
from numba_test.utils import npimg2npgray
from numba_test.energy import computeEnergy, showEnergyImg
from numba_test.entropy import localEntropyDP1D
from numba_test.gradient import computeGD

gdratio = 0.1


def main():
    print('start, ', time.asctime(time.localtime(time.time())))
    img = Image.open('../pics/butterfly.jpg')

    npgray = np.array(img.convert('L'))
    npimg = np.array(img)

    #for i in range(300):
    #    a = computeGD(npimg)
    #    b = localEntropyDP1D(npgray)
    #    print(i, time.asctime(time.localtime(time.time())))

    for i in range(3001):
        npgray = npimg2npgray(npimg)
        npimg = deleteOneColumn(npimg, npgray,gdratio)
        npgray = npimg2npgray(npimg)
        npimg = deleteOneRow(npimg, npgray,gdratio)
        if i % 30 == 0 and NEED_DISPLAY:
            showEnergyImg(computeEnergy(npimg, npgray,gdratio))
            plt.imshow(Image.fromarray(npimg))
            plt.show()
        print(i, time.asctime(time.localtime(time.time())))
    '''
    print('ready, ', time.asctime(time.localtime(time.time())))
    for j in range(10):
        for i in range(10):
            npimg = deleteOneRow(npimg)
            print('finished', j * 10 + i + 1, ", ", time.asctime(time.localtime(time.time())))
    newimg = Image.fromarray(npimg)
    newimg.save('../out/ori.jpg')
    '''

if __name__ == '__main__':
    main()