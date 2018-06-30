import numpy as np
from numba import jit, njit, prange
from PIL import Image

@njit(parallel=True,nogil=True)
def transposeImg(npimg):
    ret = np.zeros((npimg.shape[1], npimg.shape[0], 3), dtype=np.uint8)
    for i in prange(0, npimg.shape[0]):
        for j in range(0,npimg.shape[1]):
            for k in range(3):
                ret[j,i,k] = npimg[i,j,k]
    return ret

@njit(parallel=True,nogil=True)
def transposeGray(npimg):
    ret = np.zeros((npimg.shape[1], npimg.shape[0]), dtype=np.uint8)
    for i in prange(0, npimg.shape[0]):
        for j in range(0,npimg.shape[1]):
            ret[j,i] = npimg[i,j]
    return ret

@njit(parallel=True,nogil=True)
def symmetricPadding2D4(npgray):
    ret = np.zeros((npgray.shape[0] + 8, npgray.shape[1] + 8), dtype=npgray.dtype)
    for i in prange(4, npgray.shape[0] + 4):
        for j in range(4):
            ret[i][j] = npgray[i - 4][3 - j]
        for j in range(4, npgray.shape[1] + 4):
            ret[i][j] = npgray[i - 4][j - 4]
        t = npgray.shape[1] + 4
        for j in range(4):
            ret[i][j + t] = npgray[i - 4][npgray.shape[1] - j - 1]
    for i in prange(4):
        for j in range(ret.shape[1]):
            ret[i][j] = ret[7 - i][j]
    for i in prange(4):
        for j in range(ret.shape[1]):
            ret[i + npgray.shape[0] + 4][j] = ret[npgray.shape[0] + 3 - i][j]
    return ret

def npimg2npgray(npimg):
    return np.array(Image.fromarray(npimg).convert('L'))

def main():
    x = np.arange(0,1920*1080, dtype=np.uint8).reshape((1920,1080))
    y = np.arange(0,1920*1080*3, dtype=np.uint8).reshape((1920,1080,3))
    for i in range(200):
        t = symmetricPadding2D4(x)
        t1 = transposeImg(y)
        print(i)
    i = 1

if __name__ == '__main__':
    main()
