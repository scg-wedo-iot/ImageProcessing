import numpy as np

# ----- 7 segment -----
# template for number digit

# segmentTemp, boxSegment = get7SegmentTemp(offsetRow,offsetCol,w1,w2,w3,h1,h2)
# sumSegDigit = sum7Segment(digitPreprocess,segmentTemp)
# number, arrSeg_norm = segmentToNumber(sumSegDigit,0.3)
# arrNumber = matching7Segment(digitPreprocess,0.3,w1,w2,w3,h1,h2)

def get7SegmentTempl(offset,segmentBlockSize):
    '''
    called by "matching7Segment"

                w1           w2            w3
            0           w1      w1+w2         w1+w2+w3-1
        h1                1_top_top
        h2    2_top_left    3_cen_top     4_top_right
        h1    5_cen_left    6_center      7_cen_right
        h2    8_down_left   9_cen_down    10_down_right
        h1                11_down_down
    '''
    offsetRow = offset[0]
    offsetCol = offset[1]

    w1 = segmentBlockSize[0]
    w2 = segmentBlockSize[1]
    w3 = segmentBlockSize[2]
    h1 = segmentBlockSize[3]
    h2 = segmentBlockSize[4]
    # create segment, range row and col of each segment
    # segmentTempl = np.array([[0,h1-1,w1,w1+w2-1],
    #                         [h1,h1+h2-1,0,w1-1],
    #                         [2*h1+h2,2*h1+2*h2-1,w1,w1+w2-1],
    #                         [h1,h1+h2-1,w1+w2,w1+w2+w3-1],
    #                         [h1+h2,2*h1+h2-1,0,w1-1],
    #                         [h1+h2,2*h1+h2-1,w1,w1+w2-1],
    #                         [h1+h2,2*h1+h2-1,w1+w2,w1+w2+w3-1],
    #                         [2*h1+h2,2*h1+2*h2-1,0,w1-1],
    #                         [h1,h1+h2-1,w1,w1+w2-1],
    #                         [2*h1+h2,2*h1+2*h2-1,w1+w2,w1+w2+w3-1],
    #                         [2*h1+2*h2,3*h1+2*h2-1,w1,w1+w2-1] ])

    segmentTempl = np.array([[0,h1,w1,w1+w2],
                        [h1,h1+h2,0,w1],
                        [2*h1+h2,2*h1+2*h2,w1,w1+w2],
                        [h1,h1+h2,w1+w2,w1+w2+w3],
                        [h1+h2,2*h1+h2,0,w1],
                        [h1+h2,2*h1+h2,w1,w1+w2],
                        [h1+h2,2*h1+h2,w1+w2,w1+w2+w3],
                        [2*h1+h2,2*h1+2*h2,0,w1],
                        [h1,h1+h2,w1,w1+w2],
                        [2*h1+h2,2*h1+2*h2,w1+w2,w1+w2+w3],
                        [2*h1+2*h2,3*h1+2*h2,w1,w1+w2] ])

    nSeg = segmentTempl.shape[0]
    
    segmentTempl[:,0] = segmentTempl[:,0] + offsetRow
    segmentTempl[:,1] = segmentTempl[:,1] + offsetRow
    segmentTempl[:,2] = segmentTempl[:,2] + offsetCol
    segmentTempl[:,3] = segmentTempl[:,3] + offsetCol        

    boxSegment = np.concatenate((segmentTempl[:,2].reshape(nSeg,1), segmentTempl[:,0].reshape(nSeg,1), segmentTempl[:,3].reshape(nSeg,1), segmentTempl[:,1].reshape(nSeg,1)),axis=1)
    
    return segmentTempl, boxSegment

def sum7Segment(imgBin, segmentTempl):
    ## create grid segment of number
    '''
    Input: imgBin with 0 and 1

        called by "matching7Segment"

                      1_top_top
        2_top_left    3_cen_top     4_top_right
        5_cen_left    6_center      7_cen_right
        8_down_left   9_cen_down    10_down_right
                      11_down_down

    '''

    h_digit, w_digit = imgBin.shape

    if np.any(imgBin == 255):
        imgBin = imgBin/255

    nSeg = segmentTempl.shape[0]
    
    rowMax = np.max(segmentTempl[:,1])
    colMax = np.max(segmentTempl[:,3])

    arrayRatioAreaSeg = np.zeros((1,nSeg))
    # print(f'rowMax:{rowMax}, colMax:{colMax}')
    
    if (rowMax <= h_digit and colMax <= w_digit):
        # segmentTemp must not over img size
        for i in range(nSeg):
            # row and column
            rowStart = segmentTempl[i,0]
            rowEnd = segmentTempl[i,1]
            colStart = segmentTempl[i,2]
            colEnd = segmentTempl[i,3]

            matSeg = imgBin[rowStart:rowEnd,colStart:colEnd]

            # area 
            A_seg = (rowEnd - rowStart) * (colEnd - colStart)
            # print(f'Asum:{np.sum(matSeg)} A:{A_seg}')
            arrayRatioAreaSeg[0,i] = (np.sum(matSeg))/A_seg
    else: 
        # print('exceed row col')
        ValueError('segment template out of range !')

    return arrayRatioAreaSeg

def segmentToNumber(arraySegment, th, indexSegmentNumber):
    '''
    called by "matching7Segment"
                      1_top_top
        2_top_left    3_cen_top     4_top_right
        5_cen_left    6_center      7_cen_right
        8_down_left   9_cen_down    10_down_right
                      11_down_down
    input: array

        indexSegmentNumber = {'0':[1,2,4,5,7,8,10,11],
                        '1':[1,2,3,6,9,11],
                        '2':[1,4,6,11],
                        '3':[1,2,4,6,7,10,11],
                        '4':[1,3,5,8,9,10],
                        '5':[1,2,5,6,7,10,11],
                        '6':[1,2,5,6,7,8,10,11],
                        '7':[1,4,7,10],
                        '8':[1,2,4,5,6,7,8,10,11],
                        '9':[1,2,4,5,6,7,10,11] }

    '''
    arrSeg = arraySegment.copy()
    arrSeg[arrSeg >= th] = 1
    arrSeg[arrSeg != 1] = 0


    number = 99
    nSegments = 11
    nDigits = len(indexSegmentNumber)
    for idigit in range(nDigits):
        # array 
        arraySegmentRef = np.zeros((nSegments))

        # indexSeg = list(map(int,indexSegmentNumber[str(idigit)]))
        indexSeg = [int(i)-1 for i in indexSegmentNumber[str(idigit)]]
        
        arraySegmentRef[indexSeg] = 1

        diffSeg = np.sum(np.abs(arraySegmentRef - arrSeg))

        if diffSeg == 0:
            number = idigit
            break


    # # 1 
    # if top_top and top_left and top_right==0 and center and down_left==0 and down_right==0 and down_down and center_top and center_down:
    #     number = 1        
    # # 2
    # elif top_top and top_left==0 and top_right and center and down_left==0 and down_right and down_down:
    #     number = 2
    # # 3
    # elif top_top and top_left==0 and top_right and center and down_left==0 and down_right and down_down:
    #     number = 3        
    # # 4
    # elif top_top==0 and top_left==0 and top_right and center and down_left==0 and down_right and down_down==0:
    #     number = 4
    # # 5 
    # elif top_top and top_left and top_right==0 and center and down_left==0 and down_right and down_down:
    #     number = 5
    # # 6 
    # elif top_top and top_left and top_right==0 and center and down_left and down_right and down_down:
    #     number = 6
    # # 7
    # elif top_top and top_left==0 and top_right and center==0 and down_left==0 and down_right and down_down==0:
    # #   [1., 0., 1., 0., 0., 1., 0., 0.]
    #     number = 7
    # # 8
    # elif top_top and top_left and top_right and center and down_left and down_right and down_down:
    #     number = 8
    # # 9
    # elif top_top and top_left and top_right and center and down_left==0 and down_right and down_down:
    #     number = 9
    # else:
    #     number = 99

    return number, arrSeg

def matching7Segment(img,segmentBlockSize,offset,sumTH,indexSegmentNumber):
    h, w = img.shape

    segmentTemp, _ = get7SegmentTempl(offset,segmentBlockSize)

    rowEndMax = max(segmentTemp[:,1]) + 1
    colEndMax = max(segmentTemp[:,3]) + 1

    # 1 2 3 4 5 6 7
    #       4 
    # 1 1 1 1
    #             7
    # 1 1 1 1 1 1 1
    #       0 1 2 3
    nOffsetRow = h - rowEndMax + 1
    nOffsetCol = w - colEndMax + 1

    print(f'w:{w} h:{h} rowEndMax:{rowEndMax} colEndMax:{colEndMax} offsetRow:{nOffsetRow} offsetCol:{nOffsetCol}')

    # arrMatchedNumber = []
    countMatch = np.zeros((10))
    for offsetCol in range(nOffsetCol):
        for offsetRow in range(nOffsetRow):
            
            segmentTemp, _ = get7SegmentTempl((offsetRow, offsetCol), segmentBlockSize)

            arrSumSegment = sum7Segment(img,segmentTemp)
            numberOutput, _ = segmentToNumber(arrSumSegment,sumTH,indexSegmentNumber)

            # print(f'offsetRow:{offsetRow}, offsetCol:{offsetCol}, number:{matchedNumber}')
            if numberOutput != 99:
                # arrMatchedNumber.append(numberOutput)
                countMatch[numberOutput] = countMatch[numberOutput] + 1
            # arrMatchedNumber.append(matchedNumber)

    if np.sum(countMatch) > 0:
        numberMatchMax = np.argmax(countMatch)
    else: 
        numberMatchMax = 99

    return numberMatchMax, countMatch