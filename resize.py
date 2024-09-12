import numpy as np

def downSampling(img,nDown):
    # nDown = 5
    nx=img.shape[0]
    ny=img.shape[1]

    # create array down x and down y
    nxNew = np.floor(nx/nDown)
    nyNew = np.floor(ny/nDown)

    xDown = np.arange(1,nDown*nxNew,nDown,dtype=int)
    yDown = np.arange(1,nDown*nyNew,nDown,dtype=int)
    img_down = img[xDown,:]
    img_down = img_down[:,yDown]
    
    # imshowCV(img_down)
    return(img_down)