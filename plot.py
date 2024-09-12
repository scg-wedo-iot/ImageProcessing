
import matplotlib.pyplot as plt
import numpy as np
import cv2
from .morphology import*

def imshow(imgIn):
    plt.imshow(cv2.cvtColor(imgIn, cv2.COLOR_BGR2RGB))
    plt.show()
    
def imshowCVFig(figNum, imgIn):
    plt.figure(figNum)
    plt.imshow(cv2.cvtColor(imgIn, cv2.COLOR_BGR2RGB))
    plt.show()

def plotshow_cvt(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def sumPixelAndPlot(imgToSum,maxIntense,rowOrCol):
    # sumRow = np.sum(myMet_grey_inv,axis=1)

    matImgOne = imgToSum.copy()
    # matImgOne[matImgOne > 0] = 1

    sumRow = np.sum(matImgOne,axis=1)
    # print(sumRow.shape)
    
    # sumCol = np.sum(myMet_grey_inv,axis=0)
    sumCol = np.sum(matImgOne,axis=0)
    # print(sumCol.shape)

    # imshowCV(imgToSum)
    # plt.plot(np.arange(len(sumRow))+1,sumRow,'b-',np.arange(len(sumCol))+1,sumCol,'r-')
    if rowOrCol == 'row':
        plt.plot(np.arange(len(sumRow))+1,sumRow,'b-')
        plt.legend(['sumRow'])

    elif rowOrCol == 'col':
        plt.plot(np.arange(len(sumCol))+1,sumCol,'b-')
        plt.legend(['sumCol'])    

    return sumRow, sumCol

def sumRowCol(imgToSum, norm = False, plot = True):
    '''
        imgToSum = binary image
    '''    
    
    matImgOne = imgToSum.copy()

    if norm:
        matImgOne[matImgOne > 0] = 1

        imgShape = imgToSum.shape
        nrow = imgShape[0]
        ncol = imgShape[1]
        sumRow = np.sum(matImgOne,axis=1) / ncol
        sumCol = np.sum(matImgOne,axis=0) / nrow

    else:
        sumRow = np.sum(matImgOne,axis=1)
        sumCol = np.sum(matImgOne,axis=0)

    if plot:
        plt.figure()
        plt.plot(np.arange(len(sumRow))+1,sumRow,'b-')
        plt.legend(['sumRow'])

        # elif rowOrCol == 'col':
        plt.figure()
        plt.plot(np.arange(len(sumCol))+1,sumCol,'r-')
        plt.legend(['sumCol'])    

    return sumRow, sumCol

def plotArrayDiff(arr, arrTH = 0):
    
    arrDiff = np.diff(arr)
    plt.plot(arrDiff)
    plt.show()

    return arrDiff

def plotThresholdAndDiff(arr, arrTH):
    
    arrDiff, arrThresholded = thresholdAndDiff(arr, arrTH)

    plt.figure()
    plt.plot(arrThresholded)

    plt.figure()
    plt.plot(arrDiff)

def plotHist(img,nbins,densityType):
    mn=img.shape[0]*img.shape[1]
    arrIntense = img.reshape((1,mn))
    histShade = np.histogram(arrIntense, bins=nbins, range=None, normed=None, weights=None,density=densityType)
    plt.plot(histShade[1][0:nbins],histShade[0])
    plt.show()

def plotDiffSq(arrIn):
    arrDiff = np.diff(arrIn)
    arrSq = np.power(arrDiff,2)

    plt.plot(arrSq)
    plt.show()

    return arrSq
