import numpy as np

def maskGen():
    mask = np.zeros([1,1,144,1722], dtype="float32")
    for i in range(21):
        s = i*82
        e = (i+1)*82
        mask[0,0,:,s:s+4] = 1
        mask[0,0,:,e-4:e] = 1
    return mask