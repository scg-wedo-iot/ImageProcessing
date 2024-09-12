import numpy as np
import cv2
import math
from .distance import *

# def findClosetPoint(arrX,arrY,xref,yref):
#     arrX = np.reshape(arrX,(-1,1))
#     arrY = np.reshape(arrY,(-1,1))
#
#     # axis=1 >> concat by col
#     arrDiff = np.concatenate((arrX-xref,arrY-yref),axis=1)
#
#     # axis=1 >> norm array in each row
#     arrDist = np.linalg.norm(arrDiff, ord=2, axis=1)
#
#     idxMin = np.argmin(arrDist)
#     distMin = arrDist[idxMin]
#     closetCoor = np.concatenate((arrX[idxMin],arrY[idxMin]))
#
#     return closetCoor, distMin

def findCorner(contours,shapeType):
    xmin,ymin,w,h = cv2.boundingRect(contours)
    xmax = xmin + w
    ymax = ymin + h

    xcen = xmin + w/2
    ycen = ymin + h/2

    arrX = contours[:,0,0]
    arrY = contours[:,0,1]
        
    if shapeType == 'rectan':
        corner = np.zeros((4,2))

        listQ=[2,1,4,3]
        closetCoor = np.zeros((4,2),dtype = "float32")
        countQ = -1
        for q in listQ:
            countQ = countQ + 1
            # arrX_q, arrY_q = filterQuadrant(arrX,arrY,xcen,ycen,q)
            
            if q==1:
                xref = xmax
                yref = ymin
            elif q==2:
                xref = xmin
                yref = ymin
            elif q==3:
                xref = xmin
                yref = ymax
            elif q==4:
                xref = xmax
                yref = ymax
            
            closetCoor[countQ,:], distMin = findClosetPoint(arrX,arrY,xref,yref)
            
    return closetCoor, closetCoor[0,:], closetCoor[1,:], closetCoor[2,:], closetCoor[3,:]

def getPerspectiveDest(tl,tr,br,bl):
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
                    [0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]], dtype = "float32")

    return dst, maxWidth, maxHeight

def splitImage(image, splitBy, nSplits):
    imgShape = image.shape

    w = imgShape[1]
    h = imgShape[0]
    if len(imgShape) == 2:
        nChannels = 1

    else:
        nChannels = imgShape[2]
        if nChannels != 3:
            ValueError('nChanels image > 3')

    if splitBy == 'row':
        splitSize = math.floor(h/nSplits)
        imgSplit = np.zeros((nSplits,splitSize,w,nChannels), dtype=np.uint8)

    elif splitBy == 'col':
        splitSize = math.floor(w/nSplits)
        imgSplit = np.zeros((nSplits,h,splitSize,nChannels), dtype=np.uint8)

    for isplit in range(nSplits):
        '''
        index  1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 
        digit        1       1          1        
            0           0 - sizeCol
            1     sizeCol - sizeCol*2
            2   2*sizeCol - sizeCol*3
        '''
        # nameSingleDigit = nameImgWithoutExten + '_Digit_' + str(idigit+1) + '.jpg'

        # seqDigit = idigit + 1
        # select row
        istartCol = isplit * splitSize
        iendCol = istartCol + splitSize 

        if nChannels > 1:
            imgSplit[isplit,:,:,:] = image[:, istartCol:iendCol,:]
        elif nChannels == 1:
            imgSplit[isplit,:,:,0] = image[:, istartCol:iendCol]
        # if idigit == nDigits:
        #     # the last digit must be invert black to white for "Mitsu_15A"
        #     digit_single_grey = maxIntense - digit_single_grey

    return imgSplit

def removeEdge(image, edgeType, nRemove):
    imageShape = image.shape
    h = imageShape[0]
    w = imageShape[1]
    if edgeType == 'row':
        image = image[nRemove:h-nRemove,:]
    elif edgeType == 'col':
        image = image[:,nRemove:w-nRemove]

    return image

def rotate(img, angle_deg=0, scale=1, center=None):
    w = img.shape[1]
    h = img.shape[0]
    if center is None:
        center = (int(h/2), int(w/2))
    M = cv2.getRotationMatrix2D(center, angle_deg, scale)
    img_rotated = cv2.warpAffine(img, M, (w, h))

    return img_rotated

def get_shape_detail(list_shape):
    shape_info = {}
    if list_shape[0] == 1:
        shape_info['n'] = 1

