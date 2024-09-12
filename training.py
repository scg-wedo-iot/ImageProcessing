from glob import glob
import numpy as np
import random
import os
from .common import*

def loadLabeledImage(folderImg,imgType = '*.jpg',labeledType = 'folderName',new_size=None,
                     flag_color=0):
    '''
    myFolder
      -ClassA
        -img1.jpg
        -img2.jpg
        -img3.jpg
      -ClassB
        -img4.jpg
        -img5.jpg

    output1: image in 4D array
            loaddedImg(5,w,h,nCHs)
    output2: array label in 1D
            label = [A,A,A,B,B]
    '''
    listAbs, listFolder, listName = findFile(folderImg, imgType, 1)

    # image in 4D matric
    loadedImg = imreadFromPath(listAbs, new_size=new_size, flag_color=flag_color)

    # label in 1D array
    labelImg = np.array(listFolder)

    return loadedImg, labelImg