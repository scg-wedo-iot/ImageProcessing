# import numpy as np
# # import tensorflow as tf
# import cv2
# from .bbox import *
#
# def getRangePulse(sumPixel,ivalley,sumTH):
#     # sumTH = 100
#     sumRow = sumPixel
#
#     countFrame = 0
#     countRise = 0
#     countDown = 0
#
#     diffArr = np.diff(sumRow)
#
#     countRiseTH = ivalley+1
#     countDownTH = ivalley
#
#     rowStart =0
#     rowEnd = 0
#
#     findRiseOrDown = 1 # 1=find rise, 0=find down
#     for ir in np.arange(1,len(sumRow)):
#         sumVal = sumRow[ir]
#         diffVal = diffArr[ir-1]
#
#         if findRiseOrDown==1 and (diffVal>0) and (sumVal>sumTH):
#             countRise = countRise + 1
#             findRiseOrDown = 2
#
#             if countRise == countRiseTH:
#                 rowEnd = ir
#                 break
#
#         elif findRiseOrDown==2 and (diffVal<0) and (sumVal<sumTH):
#             # print('Down')
#             countDown = countDown + 1
#             findRiseOrDown = 1
#
#             if countDown == countDownTH:
#                 rowStart = ir
#
#         # print(f'ir:{ir}, find:{findRiseOrDown}, diffVal:{diffVal}, sumVal:{sumVal}, rise:           {countRise}, down:{countDown}')
#
#     return rowStart, rowEnd
#
# def imageToTensor(img):
#     # img_tf = tf.convert_to_tensor(img, dtype=tf.float32)
#
#     # # Add dimension to match with input model
#     # # model need image with dim = [1,h,w,1]
#     # img_tf = tf.expand_dims(img_tf, axis=0) # [h,w] >> [1,h,w]
#     # if len(img_tf.shape) == 3:
#     #     # image has no channel, add it
#     #     img_tf = tf.expand_dims(img_tf, axis=-1) # [1,h,w] >> [1,h,w,1]
#
#     '''
#     This function require tensorflow that slow importing time
#     Please implement this func in your main script
#     '''
#     print("ImageProcessing.object_detection.imageToTensor: This function require tensorflow that slow importing time Please implement this func in your main script")
#
#     # return img_tf
#
# def set_input_tensor(interpreter, image):
#     """Set the input tensor."""
#     tensor_index = interpreter.get_input_details()[0]['index']
#     input_tensor = interpreter.tensor(tensor_index)()[0]
#     input_tensor[:, :] = image
#
# def get_output_tensor(interpreter, index):
#     """Return the output tensor at the given index."""
#     output_details = interpreter.get_output_details()[index]
#     # remove the 1 dimension
#     # a.shape >> [1,3,4]
#     # b = np.squeeze(a)
#     # b.shape >> [3,4]
#     # tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
#     tensor = interpreter.get_tensor(output_details['index'])
#
#     return tensor
#
# def objectDetection(interpreter, image, threshold):
#     """Returns a list of object_detection results, each a dictionary of object info."""
#     # Feed the input image to the model
#     set_input_tensor(interpreter, image)
#     interpreter.invoke()
#
#     # resnet_ssd50 tflite
#     # ref for output shape
#     # https://hub.tensorflow.google.cn/tensorflow/retinanet/resnet50_v1_fpn_640x640/1
#     boxes = get_output_tensor(interpreter, 1)
#     boxes = boxes[0,:,:]
#     classes = get_output_tensor(interpreter, 2)
#     scores = get_output_tensor(interpreter, 3)
#     scores = scores[0,:,1]
#
#     # get only detected object that has score above threshold
#     indexPassTH = (scores >= threshold)
#     boxesPassTH = boxes[indexPassTH,:]
#
#     return boxesPassTH,classes,scores
#
# def cropDetectedObject(img,boxes):
#
#     ymin, xmin, ymax, xmax = box2RowCol(boxes,img.shape[1],img.shape[0])
#
#     detectedObject = img[ymin:ymax,xmin:xmax]
#
#     return detectedObject
#
# def plotMultiBox(img,arrBoxes, color = (0,255,0), thickness = 3):
#     '''
#         color = BGR
#     '''
#     nBoxes = arrBoxes.shape[0]
#     imgOut = img.copy()
#     for ibox in range(nBoxes):
#         boxes = box2RowCol(arrBoxes[ibox,:],img.shape[1],img.shape[0])
#
#         # cv2.boundingRect(img,boxes[ibox,0],boxes[ibox,1],boxes[ibox,2],boxes[ibox,3])
#         imgOut = cv2.rectangle(imgOut,( int(boxes[1]), int(boxes[0]) ),
#                                 ( int(boxes[3]), int(boxes[2]) ),
#                                 color,thickness)
#         # cv2.imshow('plot box',imgOut)
#
#     return imgOut
#
# def readLabelMap(label_map_path,header_id,header_name):
#   item_id = None
#   item_name = None
#   items = {'id':[], 'nameObject':[]}
#
#   with open(label_map_path, "r") as file:
#       for line in file:
#           line.replace(" ", "")
#           if line == "item{" or line == "}":
#               pass
#           elif header_id in line:
#               item_id = int(line.split(":", 1)[1].strip())
#               items['id'].append(item_id)
#           elif header_name in line:
#               item_name = line.split(":", 1)[1].replace("'", "").strip()
#               item_name = item_name.replace('"','')
#               items['nameObject'].append(item_name)
#
#   return items
#
# def getBoxesScoreClass(detectionObj, scoreTH):
#     '''
#         detectionObj: output from object_detection model(saved_model)
#         scoreTH:      threholds for filter image processing
#
#     '''
#
#     # All outputs are batches tensors.
#     # Convert to numpy arrays, and take index [0] to remove the batch dimension.
#     # We're only interested in the first num_detections.
#     num_detections = int(detectionObj.pop('num_detections'))
#     detectionObj = {key: value[0, :num_detections].numpy()
#                     for key, value in detectionObj.items()}
#
#     boxes = detectionObj['detection_boxes']
#     # get only detected object that has score above threshold
#     indexPassTH = (detectionObj['detection_scores'] >= scoreTH)
#     boxesPassTH = boxes[indexPassTH,:]
#
#     scorePassTH = detectionObj['detection_scores'][indexPassTH]
#
#     # detection_classes should be ints.
#     class_detections = detectionObj['detection_classes'].astype(np.int64)
#
#     return boxesPassTH, scorePassTH, class_detections
