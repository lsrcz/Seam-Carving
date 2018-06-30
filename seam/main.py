import numpy as np
import matplotlib.pyplot as plt
from seam.delete.deep import deepdeleteOneColumn
from seam.energy.heatmap import heatmap
from seam.retarget import *

gdratio = 0.1


def main():

    print('start, ', time.asctime(time.localtime(time.time())))
    print('For better performance on large images, we didn\'t compile the numba modules ahead of time, so please wait when compiling the modules')
    img = Image.open('../pics/dog.jpg')
    #img = img.resize((480,300))
    npimg = np.array(img)

    '''


    plt.imshow(img)
    plt.show()

    plt.imshow(Image.fromarray(retargetOptimal(npimg, gdratio, 1, 1)))
    plt.show()

    plt.imshow(Image.fromarray(retargetColfirst(npimg, gdratio, 1, 1)))
    plt.show()

    plt.imshow(Image.fromarray(retargetRowfirst(npimg, gdratio, 1, 1)))
    plt.show()

    exit(0)
    '''



    #for i in range(300):
    #    a = computeGD(npimg)
    #    b = localEntropyDP1D(npgray)
    #    print(i, time.asctime(time.localtime(time.time())))
    for i in range(500):
        npgray = npimg2npgray(npimg)
        heat = heatmap(npimg)[0]
        #print(heat)
        #print(computeEnergy(npimg, npgray,gdratio))
        npimg = deepdeleteOneColumn(npimg, npgray, gdratio,heat)
        print(i, time.asctime(time.localtime(time.time())))
        if i % 30 == 0:
            plt.imshow(Image.fromarray(npimg))
            plt.show()

    plt.imshow(Image.fromarray(npimg))
    plt.show()
    '''
    npimg = np.array(img)
    for i in range(500):
        npgray = npimg2npgray(npimg)
        npimg = deleteOneColumnForward(npimg, npgray,gdratio)
        #npgray = npimg2npgray(npimg)
        #npimg = deleteOneRow(npimg, npgray,gdratio)
        print(i, time.asctime(time.localtime(time.time())))
        if i % 30 == 0:
            plt.imshow(Image.fromarray(npimg))
            plt.show()
    newimg = Image.fromarray(npimg)
    plt.imshow(newimg)
    plt.show()
    '''
    print('ready, ', time.asctime(time.localtime(time.time())))
    for j in range(10):
        for i in range(10):
            npimg = deleteOneRow(npimg)
            print('finished', j * 10 + i + 1, ", ", time.asctime(time.localtime(time.time())))
    newimg = Image.fromarray(npimg)
    newimg.save('../out/ori.jpg')


if __name__ == '__main__':
    main()