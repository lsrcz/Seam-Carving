import time

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from numba_test.energy import showGDimg, computeGD
from numba_test.deleting import generatePath, deleteOneRow

def main():
    print('start, ', time.asctime(time.localtime(time.time())))
    img = Image.open('../pics/ori.jpg')

    npimg = np.array(img)

    showGDimg(computeGD(npimg))
    for i in range(100):
        energy, lastDir = generatePath(computeGD(npimg))
        print(i, time.asctime(time.localtime(time.time())))

    for i in range(501):
        npimg = deleteOneRow(npimg)
        if i % 100 == 0:
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