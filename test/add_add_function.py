from PIL import Image
import numpy as np
import numexpr as ne
import matplotlib.pyplot as plt
import cv2
import os
#from multiprocessing import Pool
import time

# NUM_OF_PROCESS = 8
# pool = Pool(processes=NUM_OF_PROCESS)
plt.ion()
#ne.set_num_threads(ne.detect_number_of_cores() // 2)


def computeGD(npimg):
    tmp = npimg
    tmp = tmp.astype('double')
    x = tmp.shape[0]
    y = tmp.shape[1]
    time = np.ones((x,y)) * 5
    time[0,0] = time[0,y-1] = time[x-1,0] = time[x-1,y - 1] = 3
    time[1 : x - 1, 1 : y - 1] += np.ones((x-2,y-2)) * 3
    gdsum = np.zeros((x,y,3))
    V = np.fabs(tmp[0:x-1,:,:]-tmp[1:x,:,:])
    H = np.fabs(tmp[:,0:y-1,:]-tmp[:,1:y,:])
    lurd = np.fabs(tmp[0:x-1,0:y-1,:]-tmp[1:x,1:y,:])
    ldru = np.fabs(tmp[1:x,0:y-1,:]-tmp[0:x-1,1:y,:])
    gdsum[0:x-1,:,:] += V
    gdsum[1:x,:,:] += V
    gdsum[:,0:y-1,:] += H
    gdsum[:,1:y,:] += H
    gdsum[0:x-1,0:y-1,:] += lurd
    gdsum[1:x,1:y,:] += lurd
    gdsum[0:x-1,1:y,:] += ldru
    gdsum[1:x,0:y-1,:] += ldru
    gdsum=gdsum.sum(axis=2)/3
    gdsum=gdsum / time
    return gdsum

def showGDimg(gdimg):
    plt.figure(figsize=(19.2, 10.8))
    plt.title("gradient image")
    plt.imshow(Image.fromarray((gdimg / gdimg.max() * 255).astype(np.uint8)))
    plt.show()


def generatePath(gdimg):
    lastDir = np.zeros_like(gdimg, dtype=np.uint8)
    energy = np.zeros_like(gdimg, dtype=np.float32)
    energy[0] = gdimg[0]
    tmp = np.zeros((3,gdimg.shape[1]), dtype=np.float32)
    t = np.array(range(tmp.shape[1]))

    for i in np.arange(1, gdimg.shape[0]):
        tmp[0] = np.hstack((np.array([np.Inf]), energy[i - 1,:-1])) + gdimg[i]
        tmp[1] = energy[i - 1] + gdimg[i]
        tmp[2] = np.hstack((energy[i - 1,1:], np.array([np.Inf]))) + gdimg[i]
        energy[i] = np.min(tmp,axis=0)
        lastDir[i] = np.argmin(tmp,axis=0)
        #energy[i] = tmp[lastDir[i],t]
    return energy, lastDir

def addKRow(npimg,k):
    gdimg = computeGD(npimg)
    energy, lastDir = generatePath(gdimg)
    x = gdimg.shape[0]
    y = gdimg.shape[1]
    newgre = np.ones((x,y+k)) * 100000
    newgre[:,0:y]=energy
    new = np.zeros((x,y+k,3))
    new[:,0:y,:] = npimg
    for i in range(k):
        minx=np.argmin(newgre[x-1,:])
        newgre[x-1,minx]=100000
        count = 1
        if minx != y - 1:
            newgre[x-1,(minx+2):(y + 1)]=newgre[x-1, (minx+1):y]
            new[x-1,(minx+2):(y+1),:]=new[x-1,(minx+1):y,:]
            new[x-1,(minx+1),:] = new[x-1,(minx+2),:]
            count = count + 1
        newgre[x-1,minx+1]=100000
        new[x - 1,minx+1,:] = new[x - 1,minx+1,:] + new[x - 1,minx,:]
        if minx != 0 :
            new[x - 1, minx + 1,:]=new[x - 1,minx+1,:]+new[x - 1,minx-1,:]
            count = count + 1  
        new[x - 1, minx + 1,:]=new[x - 1,minx+1,:]/count
        for j in range(x - 1):
            b = np.argmin(newgre[x - j - 2, max(minx-1,0):min(minx + 2,y)])
            minx=b+max(minx-1,0)
            newgre[x - j - 2, minx]=100000
            count = 1
            if minx != y-1:
                newgre[x- j - 2,(minx+2):(y+1)]=newgre[x- j -2 , (minx + 1):y]
                new[x- j - 2,(minx+2):(y+1),:]=new[x- j - 2,(minx+1):y,:]
                new[x- j - 2,minx+1,:] = new[x - j - 2,minx+2,:]
                count = count + 1
            newgre[x-j-2,minx+1]=100000
            new[x-j-2,minx+1,:] = new[x-j-2,minx+1,:] + new[x-j-2,minx,:]
            if minx != 0:
                new[x-j-2, minx + 1,:]=new[x-j-2,minx+1,:]+new[x-j-2,minx-1,:]
                count = count + 1
            new[x - j-2, minx + 1,:]=new[x-j-2,minx+1,:]/count
        y = y + 1
        print('finished', i + 1, ", ", time.asctime(time.localtime(time.time())))
    npimg = np.array(new,dtype=np.uint8)
    return npimg

def deleteOneRow(npimg):
    gdimg = computeGD(npimg)
    energy, lastDir = generatePath(gdimg)
    #mask = np.full(gdimg.shape, True)

    last = np.argmin(energy[-1])
    ret = np.ndarray((gdimg.shape[0], gdimg.shape[1] - 1, 3),dtype=np.uint8)
    for i in range(gdimg.shape[0] - 1, -1, -1):
        #mask[i][last] = False
        ret[i] = np.delete(npimg[i], last, axis=0)
        last = last + lastDir[i][last] - 1
    return ret #npimg[mask].reshape((gdimg.shape[0], gdimg.shape[1] - 1, 3))


if __name__ == '__main__':
    print('start, ', time.asctime(time.localtime(time.time())))
    img = Image.open('D:/pics/bubble.jpg')
    #plt.figure(figsize=(19.2, 10.8))
    #plt.imshow(img)
    #plt.show()

    npimg = np.array(img)
    #showGDimg(computeGD(npimg))

    # plt.imshow(img.convert('L'))
    print('ready, ', time.asctime(time.localtime(time.time())))
    #for j in range(40):
    #    for i in range(10):
    #        npimg = deleteOneRow(npimg)
    #        print('finished', j * 10 + i + 1, ", ", time.asctime(time.localtime(time.time())))
    npimg = addKRow(npimg,100)
    print('finished, ', time.asctime(time.localtime(time.time())))
    newimg=Image.fromarray(npimg)
    newimg.save('D:/pics/testpy.jpg')

