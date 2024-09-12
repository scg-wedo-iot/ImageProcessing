import numpy as np

def findClosetPoint(arrX, arrY, xref, yref):
    arrX = np.reshape(arrX, (-1, 1))
    arrY = np.reshape(arrY, (-1, 1))

    # axis=1 >> concat by col
    arrDiff = np.concatenate((arrX - xref, arrY - yref), axis=1)

    # axis=1 >> norm array in each row
    arrDist = np.linalg.norm(arrDiff, ord=2, axis=1)

    idxMin = np.argmin(arrDist)
    distMin = arrDist[idxMin]
    closetCoor = np.concatenate((arrX[idxMin], arrY[idxMin]))

    return closetCoor, distMin
