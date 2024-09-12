import numpy as np
from .plot import*

def thresholdAndDiff(arrIn, arrTH):
    arr = arrIn.copy()
    indexHigh = (arr > arrTH)
    arr[indexHigh] = 1
    arr[~indexHigh] = 0
    arrDiff = np.diff(arr)

    return arrDiff, arr

def thresholdAndDiffRange(arrIn, rangeTH, newVal = (1,0,-1)):
    arr = arrIn.copy()
    indexHigh = (arr > rangeTH[0])
    indexLow = (arr < rangeTH[1])
    indexMid = ~(indexHigh | indexLow)

    arr[indexHigh] = newVal[0]
    arr[indexMid] = newVal[1]
    arr[indexLow] = newVal[2]

    arrDiff = np.diff(arr)

    return arrDiff, arr

def findRangeObject1D(arrSumDiff, lenObjectTH):
    '''
        lenObject = no. Pixel object / no. all
    '''
    
    findRiseOrFall = 1
    # h_frame = frameShape[0]
    # w_frame = frameShape[1]
    lenPixel = len(arrSumDiff) + 1

    # lenObjectTH = 0.15

    for i in range(lenPixel):
        rowDiff = arrSumDiff[i]

        if findRiseOrFall == 1:
            # find rise (top of frame)
            if rowDiff == 1:
                r1 = i
                findRiseOrFall = 0

        elif findRiseOrFall == 0:
            # find fall (bottom of frame)
            if rowDiff == -1:
                r2 = i
                
                lenObject = (r2-r1)/lenPixel

                if lenObject > lenObjectTH:
                    break
                else:
                    findRiseOrFall = 1

    return r1,r2

def findRangeArrayDiff(arrSumDiff):
    '''
        lenObject = no. Pixel object / no. all
    '''
    
    tableObject = {'r1':[], 'r2':[], 'lenObject':[]}
    findRiseOrFall = 1 # 1=find rise, 0=find fall
    # h_frame = frameShape[0]
    # w_frame = frameShape[1]
    lenPixel = len(arrSumDiff)
    countPeak = 0
    countZeroBetweenPeak = 0
    countValley = 0
    countZeroBetweenValley = 0

    tolZeroBetweenPeak = 5
    tolZeroBetweenValley = 5

    r1 = 0
    r2 = 0

    for i in range(lenPixel):
        diffVal = arrSumDiff[i]

        if findRiseOrFall == 1:
            # find rise (top of frame)
            if diffVal == 2:
                # found index start object
                r1 = i
                findRiseOrFall = 2 # go find index end object
            
            elif diffVal == 1:

                if countPeak == 0:
                    countPeak = countPeak + 1
                elif countPeak == 1 and countZeroBetweenPeak <= tolZeroBetweenPeak:
                    # found peak
                    r1 = i
                    findRiseOrFall = 2 # go find index end object
                else:
                    # reset, it noise
                    countZeroBetweenPeak = 0
                    
            elif diffVal == 0:
                if countPeak == 1:
                    countZeroBetweenPeak = countZeroBetweenPeak + 1

            elif diffVal == -1:
                countPeak = 0
                countZeroBetweenPeak = 0


        elif findRiseOrFall == 2:
            # find fall, end of object (top of frame)
            if diffVal == -2:
                # found index end object
                r2 = i
                findRiseOrFall = 2 # go find index end object
            
            elif diffVal == -1:

                if countValley== 0:
                    countValley = countValley + 1
                elif countValley == 1 and countZeroBetweenValley <= tolZeroBetweenValley:
                    # found valley
                    r2 = i
                    findRiseOrFall = 1 # go log info
                
                else:
                    # reset, it noise
                    countZeroBetweenValley = 0
                    
            elif diffVal == 0:
                if countValley == 1:
                    countZeroBetweenValley = countZeroBetweenValley + 1

            elif diffVal == 1:
                countValley = 0
                countZeroBetweenValley = 0

        if r1 > 0 and r2 > 0:
            # log info 
            lenObject = (r2-r1)/lenPixel
            tableObject['r1'].append(r1)
            tableObject['r2'].append(r2)
            tableObject['lenObject'].append(lenObject)

            r1 = 0
            r2 = 0

            countPeak = 0
            countZeroBetweenPeak = 0
            countValley = 0
            countZeroBetweenValley = 0

            # # find fall (bottom of frame)
            # if diffVal <= diffValRange[1]:
            #     r2 = i
            #     findRiseOrFall = 1

            #     lenObject = (r2-r1)/lenPixel
            #     # print(f'len:{lenObject}, lenTH:{lenObjectTH} \n')

            #     if lenObject >= lenObjectTH[0] and lenObject <= lenObjectTH[1]:
            #         # break
            #         tableObject['r1'].append(r1)
            #         tableObject['r2'].append(r2)
            #         tableObject['lenObject'].append(lenObject)
                    

    return tableObject

def cropObject(frame_bin, sumRowTH, lenObjectRow, sumColTH, lenObjectCol):
    sumRow, sumCol = sumRowCol(frame_bin, norm = True, plot = False)

    sumRowDiff = thresholdAndDiff(sumRow, sumRowTH)
    sumColDiff = thresholdAndDiff(sumCol, sumColTH)

    r1, r2 = findRangeObject1D(sumRowDiff, lenObjectRow)
    c1, c2 = findRangeObject1D(sumColDiff, lenObjectCol)

    frame_bin_crop = frame_bin[r1:r2,c1:c2]

    return frame_bin_crop

def filterObject(matCoor, imgShape, rangeLengthObject, boxOfCenter):
    '''
        Select only the right object
    '''
    w = imgShape[0]
    h = imgShape[1]
    nObjects = matCoor.shape[0]
    indexPass = np.zeros((nObjects), dtype=bool)
    listCondition = []
    for i in range(nObjects):
        lengthRow = (matCoor[i,1] - matCoor[i,0])/h
        h_object = ( (matCoor[i,1] + matCoor[i,0])/2 )/w
        lengthCol = ( matCoor[i,3] - matCoor[i,2] )/w
        w_object = ( (matCoor[i,3] + matCoor[i,2])/2 )/h

        listCondition.append(lengthRow >= rangeLengthObject[0] and lengthRow <= rangeLengthObject[1])
        listCondition.append(lengthCol >= rangeLengthObject[2] and lengthCol <= rangeLengthObject[3])
        
        listCondition.append(h_object > boxOfCenter[0] and h_object < boxOfCenter[1])
        listCondition.append(w_object > boxOfCenter[2] and w_object < boxOfCenter[3])

        if all(listCondition):
            # log object
            indexPass[i] = True

    matCoorFiltered = matCoor[indexPass,:]

    return matCoorFiltered

        
                                

        

def cropMultipleObject(frame_bin, sumRowTH, sumColTH, listDiffVal, orderCut = False):
    '''
    Input:
        frame_bin : binary image (support both 0 and 1 , 0 and 255)
        sumRowTH  : Thresholding for sum row pixel value (before find diff)
        sumColTH  : Thresholding for sum col pixel value (before find diff)

        diffVal : ()

    '''

    listImage = []

    # ----- Sum pixel in row -----
    sumRow, _ = sumRowCol(frame_bin, norm=True, plot=False)
    # Diff of sum pixel
    sumRowDiff, _ = thresholdAndDiffRange(sumRow, sumRowTH, listDiffVal)

    # find range that is interested object
    tableRangeRow = findRangeArrayDiff(sumRowDiff)

    nObjFromRow = len(tableRangeRow['r1'])

    # crop row and find object from sumCol
    matCoor = np.zeros((nObjFromRow,4),dtype=np.int)
    conutObj = -1
    for iobjR in range(nObjFromRow):
        r1 = tableRangeRow['r1'][iobjR]
        r2 = tableRangeRow['r2'][iobjR]

        frame_bin_Crop = frame_bin[r1:r2,:]

        _, sumCol = sumRowCol(frame_bin_Crop, norm=True, plot=False)

        sumColDiff, _ = thresholdAndDiffRange(sumCol, sumColTH, listDiffVal)

        tableRangeCol = findRangeArrayDiff(sumColDiff)

        nObjFromCol = len(tableRangeCol['r1'])
        for iobjC in range(nObjFromCol):
            conutObj = conutObj+1
            # print(f'iobjRow:{iobjR+1}/{nObjFromRow}, iobjCol:{iobjC+1}/{nObjFromCol}')
            c1 = tableRangeCol['r1'][iobjC]
            c2 = tableRangeCol['r2'][iobjC]

            # frame_bin_Crop = frame_bin_Crop[:,c1:c2]

            # listImage.append(frame_bin_Crop)

            matCoor[conutObj,0] = r1
            matCoor[conutObj,1] = r2
            matCoor[conutObj,2] = c1
            matCoor[conutObj,3] = c2
            
            # plt.figure(conutObj)
            # wedoimg.imshow(frame_bin_invCrop)

            # conutObj = conutObj + 1
    # return listImage, matCoor
    return matCoor
