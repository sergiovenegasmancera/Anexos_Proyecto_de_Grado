##############################################################################################################################################################################################

from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import time
from Face_Detectors import *
from Feature_Extractor import *
from Classifier_ import *
import dlib
import numpy as np

##############################################################################################################################################################################################

def resize_Image(path, scale_percent):
    Image = cv2.imread(path)
    width = int(Image.shape[1] * scale_percent / 100)
    height = int(Image.shape[0] * scale_percent / 100)
    dim = (width, height)    
    # resize image
    Image = cv2.resize(Image, dim, interpolation = cv2.INTER_AREA)
    
    return Image

##############################################################################################################################################################################################

With_Mask = resize_Image('/home/pi/Desktop/Proyecto_de_Grado/Images/With_Mask2.jpg', 30)
Without_Mask = resize_Image('/home/pi/Desktop/Proyecto_de_Grado/Images/Without_Mask2.jpg', 30)
Incorrect_Mask = resize_Image('/home/pi/Desktop/Proyecto_de_Grado/Images/Incorrect_Mask2.jpg', 30)

fps = 0
# font
font = cv2.FONT_HERSHEY_SIMPLEX
# org
org = (15, 25)  
# fontScale
fontScale = 0.75   
# Blue color in BGR
color = (255, 255, 255) 
# Line thickness of 2 px
thickness = 2


#Type_Face_Detector = "Haar_Cascades"
Type_Face_Detectors = ["Haar_Cascades", "DNN", "Dlib"]
Type_Face_Detector = Type_Face_Detectors[1]
model_detector = Get_Model_Face_Detector(Type_Face_Detector)


Type_features = ["Color_quotient", "Texture"]
Type_Classifiers = ["SVM","KNN","MLP"]
Classifier_1_Etapa, Scaler_1_Etapa, Classifier_2_Etapa, Scaler_2_Etapa = Get_Classifier_model(Type_features[1], Type_Classifiers[0])

le_1_etapa = preprocessing.LabelEncoder().fit(['Without_Mask', 'With_Mask'])
le_2_etapa = preprocessing.LabelEncoder().fit(['Incorrect_Mask', 'With_Mask'])

##############################################################################################################################################################################################

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (1280, 720)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(1280, 720))
# allow the camera to warmup
time.sleep(0.1)
contador = 0
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    start = time.time()
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    image = frame.array
    #image.setflags(write=1)
    #print(image.flags)
    #image.flags.writeable = True
    contador = contador + 1
    if contador % 3 == 0:
        faces = face_detector(Type_Face_Detector, image, model_detector)
        for (x, y, w, h) in faces:
            face = image[y:y+h, x:x+w]
            X = Feature_Extractor_1_Etapa(face, Type_features[1])
            y_predicted = classifier(X, Classifier_1_Etapa, Scaler_1_Etapa, 'SVM', le_1_etapa)
            if y_predicted == 'With_Mask':
                X = Feature_Extractor_2_Etapa(face, Type_features[1])
                y_predicted = classifier(X, Classifier_2_Etapa, Scaler_2_Etapa, 'SVM', le_2_etapa)
            image = Draw_Rectangle(image, y_predicted, x, y, w, h, With_Mask, Without_Mask,  Incorrect_Mask)
               
        end = time.time()
        fps = 1/((end-start))
            
        # Using cv2.putText() method
        image = cv2.rectangle(image,(0,0),(150,35),(0,0,0),-1)
        image = cv2.putText(image, str(round(fps,3))+' FPS', org, font, fontScale, color, thickness, cv2.LINE_AA)

        # show the frame
        cv2.imshow("Frame", image)
        key = cv2.waitKey(1) & 0xFF
        
        end = time.time()
        fps = 1/(end - start)
        #print(fps)

        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
    

cv2.destroyAllWindows()