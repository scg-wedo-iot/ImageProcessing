import cv2

def imshowCV(name, imgIn):
    cv2.imshow(name, imgIn)
    cv2.waitKey(0)
