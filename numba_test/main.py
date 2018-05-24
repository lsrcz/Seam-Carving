from numba_test.config import *
import time

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from numba_test.energy import showGDimg, computeGD
from numba_test.deleting import deleteOneRow, deleteOneColumn

def main():
    print('start, ', time.asctime(time.localtime(time.time())))
    img = Image.open('../pics/1080p.jpg')

    npimgr = np.array(img)
    npimgc = np.array(img)
    npimg = np.array(img)
    '''
    npimgr = npimg[:,:,0]
    npimgg = npimg[:,:,1]
    npimgb = npimg[:,:,2]
    showGDimg(computeGDOld(npimg))
    showGDimg(computeGD(npimgr,npimgg,npimgb))
    for i in range(200):
        a = computeGD(npimgr,npimgg,npimgb)
        b = computeGDOld(npimg)
        print(i, time.asctime(time.localtime(time.time())))
    '''
    showGDimg(computeGD(npimgr))

    for i in range(3001):
        npimg = deleteOneColumn(npimg)
        npimg = deleteOneRow(npimg)
        if i % 50 == 0 and NEED_DISPLAY:
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