import numpy as np
from numba import jit, njit, prange

def padding(img):
    return img.padding(img,((4,4),(4,4)),'symmatric')

@njit(parallel=True,nogil=True)
def localEntropy(npgray):
    entropy = np.zeros((npgray.shape[0],npgray.shape[1]))
    #npgray = padding(npgray)
    for i in prange(4,npgray.shape[0] - 4):
        for j in range(4, npgray.shape[1] - 4):
            Hist=np.zeros(256)
            for p in range(i - 4, i + 5):
                for q in range(j - 4, j + 5):
                    Hist[npgray[p,q]] += 1
            total = 0
            for k in range(256):
                total += Hist[k]
            for k in range(256):
                Hist[k] /= total
                if Hist[k]!=0:  
                    entropy[i,j]+=Hist[k]*np.log(1/Hist[k])
    return entropy