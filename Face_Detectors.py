import cv2
import numpy as np
import dlib
import imutils
import os
import joblib
from sklearn.preprocessing import *
from skimage.measure import moments
from skimage.feature import greycomatrix, greycoprops
from skimage import io, color, img_as_ubyte
from imutils.video import VideoStream
from sklearn.preprocessing import *
import time




def Get_Model_Face_Detector(Type_Face_Detector):
    
    if Type_Face_Detector == "Haar_Cascades":
        model_detector = cv2.CascadeClassifier('/home/pi/opencv_build/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml')
    elif Type_Face_Detector == "DNN":
        modelFile = "./model/res10_300x300_ssd_iter_140000.caffemodel"
        configFile = "./model/deploy.prototxt.txt"
        model_detector = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    else:
        model_detector = dlib.get_frontal_face_detector()
    
    return model_detector
    
def face_detector(Type_Face_Detector, frame, model_detector):
    
    faces = []
    
    if Type_Face_Detector == "Haar_Cascades":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = model_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(200, 200),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
    elif Type_Face_Detector == "DNN":
        
        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))


        model_detector.setInput(blob)
        detections = model_detector.forward()
        
        
        # loop over the detections
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            if confidence < 0.5:
                continue

            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            x, y, w, h = startX, startY, endX - startX, endY - startY
            faces.append([x, y, w, h])
            
    else:

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rects = model_detector(rgb)
        for r in rects:
            # extract the starting and ending (x, y)-coordinates of the
            # bounding box
            startX = r.left()
            startY = r.top()
            endX =   r.right()
            endY =   r.bottom()
            # ensure the bounding box coordinates fall within the spatial
            # dimensions of the image
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(endX, frame.shape[1])
            endY = min(endY, frame.shape[0])
            # compute the width and height of the bounding box
            w = endX - startX
            h = endY - startY
            faces.append([startX, startY, w, h])
            
    return faces

def Draw_Rectangle(frame, y_predicted, x, y, w, h, With_Mask, Without_Mask,  Incorrect_Mask):
    
    #frame.flags.writeable = True

    # Draw a rectangle around the faces
    if y_predicted == 'Without_Mask':
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
        Image = Without_Mask
    elif y_predicted == 'With_Mask':
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        Image = With_Mask
    else:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(1,210,250),3)
        Image = Incorrect_Mask
        
    Image.flags.writeable = True
        
    h_total, w_total = 720, 1280
    endX = x + w
    y0, x0 = y-20, endX-20
    y1, x1 = y+Image.shape[0]-20, endX+Image.shape[1]-20
    
    #if (y0 < h_total) and (y1 < h_total) and (x0 < w_total) and (x1 < w_total):
        #frame[y0:y1, x0:x1] = Image

        
    return frame  



    
    
