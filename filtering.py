
import numpy as np
import cv2 

def zConvolute(img,kernel,posOut):
    if posOut=='center':
        '''
        - convolution filter without padding
        - kernel size must be odd*odd (either that we can not define center)
        '''
        
        [rImg,cImg]=img.shape
        [rk,ck]=kernel.shape
        imgOut=np.zeros((rImg,cImg))

        '''
             1 1 1 1 1
         1   1 1 1 1 1 1 1 1 1 1
         1   1 1 1 1 1 1 1 1 1 1
         1   1 1 1 1 1 1 1 1 1 1
         1   1 1 1 1 1 1 1 1 1 1
         1   1 1 1 1 1 1 1 1 1 1
             1 1 1 1 1 1 1 1 1 1
             
        '''

        kernelOffsetC=(ck-1)/2   # 3 >> 1=(3-1)/2, 5 >> 2=(5-1)/2, 7 >> 3=(7-1)/2
        kernelOffsetR=(rk-1)/2
        idxWinC=np.array([0,0])
        idxWinR=np.array([0,0])
        countSlide=0
        for ir in np.arange(kernelOffsetR,rImg-1-kernelOffsetR,1,dtype=int):
            countSlide=countSlide+1
            # print(f'slide:{countSlide}')
            
            for ic in np.arange(kernelOffsetC,cImg-1-kernelOffsetC,1,dtype=int):
                idxWinC[0] = ic - kernelOffsetC
                idxWinC[1] = ic + kernelOffsetC +1
                idxWinR[0] = ir - kernelOffsetR

                idxWinR[1] = ir + kernelOffsetR+1
                # print(f'ir={ir},ic={ic}')
                # print(f'{idxWinR[0]},{idxWinR[1]},{idxWinC[0]},{idxWinC[1]}')
                winImg=img[idxWinR[0]:idxWinR[1],idxWinC[0]:idxWinC[1]]
                # print(winImg)

                # print(kernel)
                #print(np.inner(winImg,kernel))
                #print(np.tensordot(winImg, kernel, axes=2))
                # imgOut[ir,ic]=np.dot(winImg,kernel)
                imgOut[ir, ic] = np.tensordot(winImg, kernel, axes=2)
                # print(imgOut[ir, ic])
    return imgOut

def zFilImage(img,kernel,filType):

    [rImg,cImg]=img.shape
    [rk,ck]=kernel.shape
    imgOut=np.zeros((rImg,cImg))

    '''
         1 1 1 1 1
     1   1 1 1 1 1 1 1 1 1 1
     1   1 1 1 1 1 1 1 1 1 1
     1   1 1 1 1 1 1 1 1 1 1
     1   1 1 1 1 1 1 1 1 1 1
     1   1 1 1 1 1 1 1 1 1 1
         1 1 1 1 1 1 1 1 1 1

    '''
    
    maxIntense = 255

    kernelOffsetC=(ck-1)/2   # 3 >> 1=(3-1)/2, 5 >> 2=(5-1)/2, 7 >> 3=(7-1)/2
    kernelOffsetR=(rk-1)/2
    idxWinC=np.array([0,0])
    idxWinR=np.array([0,0])
    
    sumKernel = np.sum(kernel)*maxIntense
    
    countSlide=0
    for ir in np.arange(kernelOffsetR,rImg-1-kernelOffsetR,1,dtype=int):
        countSlide=countSlide+1
        # print(f'slide:{countSlide}')

        for ic in np.arange(kernelOffsetC,cImg-1-kernelOffsetC,1,dtype=int):
            idxWinC[0] = ic - kernelOffsetC
            idxWinC[1] = ic + kernelOffsetC +1
            idxWinR[0] = ir - kernelOffsetR
            idxWinR[1] = ir + kernelOffsetR+1
            # print(f'ir={ir},ic={ic}')
            # print(f'{idxWinR[0]},{idxWinR[1]},{idxWinC[0]},{idxWinC[1]}')
            winImg=img[idxWinR[0]:idxWinR[1],idxWinC[0]:idxWinC[1]]
            # print(winImg)

            # print(kernel)
            #print(np.inner(winImg,kernel))
            #print(np.tensordot(winImg, kernel, axes=2))
            z=np.tensordot(winImg,kernel,axes=2)
            
            # print(sumKernel)
            # print(z)

            if filType == 'erode':
                if z >= maxIntense:
                    # Get output even just match only one points
                    imgOut[ir,ic]=maxIntense
                    
            elif filType == 'dilation':
                if z == sumKernel:
                    # Get output when fully match
                    imgOut[ir,ic]=maxIntense
                
            #imgOut[ir, ic] = np.tensordot(winImg, kernel, axes=2)
            # print(imgOut[ir, ic])
    return imgOut

def fillPointInEdge(img_edge):

    kernel_all = []
    kernel = np.array([[0,0,1,0,0],
                        [0,0,0,0,0],
                        [0,0,0,0,0],
                        [0,0,1,0,0],
                        [0,0,1,0,0] ], dtype='uint8')
    kernel_all.append(kernel)

    kernel = np.array([[0,1,0],
                        [0,0,0],
                        [0,1,0]], dtype='uint8')
    kernel_all.append(kernel)

    kernel = np.array([[0,0,0,0,0],
                        [0,0,0,0,0],
                        [1,0,0,1,1],
                        [0,0,0,0,0],
                        [0,0,0,0,0]], dtype='uint8')
    kernel_all.append(kernel)

    kernel = np.array([[0,0,0],
                        [1,0,1],
                        [0,0,0]], dtype='uint8')
    kernel_all.append(kernel)

    kernel = np.array([[0,0,1],
                        [0,0,0],
                        [1,0,0]], dtype='uint8')
    kernel_all.append(kernel)

    kernel = np.array([[1,0,0],
                        [0,0,0],
                        [0,0,1]], dtype='uint8')
    kernel_all.append(kernel)

    for i in range(len(kernel_all)):
        img_edge_e = cv2.erode(img_edge,kernel_all[i])

        if i == 0:
            img_edge_fix = img_edge + img_edge_e
        else:
            img_edge_fix = img_edge_fix + img_edge_e

    return img_edge_fix

