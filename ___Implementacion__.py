from Face_Detectors import *
from Feature_Extractor import *
from Classifier_ import *
import cv2
import time
import dlib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn import preprocessing



def resize_Image(path, scale_percent):
    Image = cv2.imread(path)
    width = int(Image.shape[1] * scale_percent / 100)
    height = int(Image.shape[0] * scale_percent / 100)
    dim = (width, height)    
    # resize image
    Image = cv2.resize(Image, dim, interpolation = cv2.INTER_AREA)
    
    return Image

    
With_Mask = resize_Image('/home/pi/Desktop/Proyecto_de_Grado/Images/With_Mask2.jpg', 30)
Without_Mask = resize_Image('/home/pi/Desktop/Proyecto_de_Grado/Images/Without_Mask2.jpg', 30)
Incorrect_Mask = resize_Image('/home/pi/Desktop/Proyecto_de_Grado/Images/Incorrect_Mask2.jpg', 30)



fps = 0
#delay = 0.014183019518143256

# font
font = cv2.FONT_HERSHEY_SIMPLEX
# org
#org = (15, 25)
#org = (img.shape[1]-135, img.shape[0]-10)
org = (572-135, 1017-10)
# fontScale
fontScale = 0.75   
# Blue color in BGR
#color = (255, 255, 255)
color = (0, 0, 0) 
# Line thickness of 2 px
thickness = 2



Type_Face_Detectors = ["Haar_Cascades", "DNN", "Dlib"]
Type_Face_Detector = Type_Face_Detectors[1]
model_detector = Get_Model_Face_Detector(Type_Face_Detector)


Type_features = ["Color_quotient", "Texture"]
Type_feature = Type_features[1]
Type_Classifiers = ["SVM","KNN","MLP"]
Type_Classifier = Type_Classifiers[2]
Classifier_1_Etapa, Scaler_1_Etapa, Classifier_2_Etapa, Scaler_2_Etapa = Get_Classifier_model(Type_feature, Type_Classifier)

le_1_etapa = preprocessing.LabelEncoder().fit(['Without_Mask', 'With_Mask'])
le_2_etapa = preprocessing.LabelEncoder().fit(['Incorrect_Mask', 'With_Mask'])



movie = cv2.VideoCapture(r"/home/pi/Desktop/Proyecto_de_Grado/IMG_9451.mp4")
#movie = cv2.VideoCapture(r"/home/pi/Downloads/IMG_126749668.mp4")
opened = movie.isOpened

contador = 0 
 
if not opened:
    print("Stream error")

while opened:
    #time.sleep(delay)
    start = time.time()
    ret, img = movie.read()
    if ret == True:
        contador = contador + 1
        if contador % 5 == 0:
            scale_percent = 53 # percent of original size
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)
              
            # resize image
            img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            
            faces = face_detector(Type_Face_Detector, img, model_detector)
            for (x, y, w, h) in faces:
                face = img[y:y+h, x:x+w]
                X = Feature_Extractor_1_Etapa(face, Type_feature)
                y_predicted = classifier(X, Classifier_1_Etapa, Scaler_1_Etapa, Type_Classifier, le_1_etapa)
                if y_predicted == 'With_Mask':
                    X = Feature_Extractor_2_Etapa(face, Type_feature)
                    y_predicted = classifier(X, Classifier_2_Etapa, Scaler_2_Etapa, Type_Classifier, le_2_etapa)
                
                img = Draw_Rectangle(img, y_predicted, x, y, w, h, With_Mask, Without_Mask,  Incorrect_Mask)
                
            end = time.time()
            fps = 1/((end-start))
            
            # Using cv2.putText() method
            #img = cv2.rectangle(img,(0,0),(150,35),(0,0,0),-1)
            img = cv2.rectangle(img,(img.shape[1]-150,img.shape[0]-35),(img.shape[0],img.shape[0]),(0,0,0),-1)
            img = cv2.putText(img, str(round(fps,3))+' FPS', org, font, fontScale, (255,255,255), thickness, cv2.LINE_AA)
            
            x = 80
            y = 32
            
            org_ = (30, 30+20)
            org__ = (30+3, 30+40)
            fontScale_ = 0.5
            thickness_ = 1
            
    
            ############################### Module Camera #################################
            img = cv2.rectangle(img,(20+0,30),(20+x,80),(255,255,255),-1)
            img = cv2.putText(img, "Camera", (30, 30+20), font, 0.5, color, 1, cv2.LINE_AA)
            img = cv2.putText(img, "Module", (30+3, 30+40), font, 0.5, color, 1, cv2.LINE_AA)
            
            ############################### Face Detector #################################
            img = cv2.rectangle(img,(20+x+y,30),(20+2*x+y,80),(255,255,255),-1)
            if Type_Face_Detector == "Haar_Cascades":
                img = cv2.putText(img, "Haar", (22+x+y+20, 30+20), font, 0.5, color, 1, cv2.LINE_AA)
                img = cv2.putText(img, "Cascades", (20+x+y+3, 30+40), font, 0.5, color, 1, cv2.LINE_AA)
            elif Type_Face_Detector == "DNN":
                img = cv2.putText(img, "DNN", (22+x+y+17, 30+30), font, 0.6, color, 1, cv2.LINE_AA)
            else:
                img = cv2.putText(img, "HOG", (22+x+y+17, 30+30), font, 0.6, color, 1, cv2.LINE_AA)
                
            ############################### Feature Extractor #################################
            #Type_feature = Type_features[0]
            img = cv2.rectangle(img,(20+2*x+2*y,30),(20+3*x+2*y,80),(255,255,255),-1)
            if Type_feature == "Color_quotient":
                img = cv2.putText(img, "Color", (20+2*x+2*y+20, 30+20), font, 0.5, color, 1, cv2.LINE_AA)
                img = cv2.putText(img, "Quotient", (20+2*x+2*y+8, 30+40), font, 0.5, color, 1, cv2.LINE_AA)
            else:
                img = cv2.putText(img, "Texture", (20+2*x+2*y+9, 30+30), font, 0.55, color, 1, cv2.LINE_AA)
                
            ############################### Classifier #################################
            img = cv2.rectangle(img,(20+3*x+3*y,30),(20+4*x+3*y,80),(255,255,255),-1)
            img = cv2.putText(img, Type_Classifier, (20+3*x+3*y+20, 30+30), font, 0.6, color, 1, cv2.LINE_AA)
            
            ############################### Display #################################
            img = cv2.rectangle(img,(20+4*x+4*y,30),(20+5*x+4*y,80),(255,255,255),-1)
            img = cv2.putText(img, "Display", (20+4*x+4*y+8, 30+30), font, 0.6, color, 1, cv2.LINE_AA)
            
            img = cv2.putText(img, "-->", (20+x-7, 60), font, fontScale_, (255, 255, 255), 2, cv2.LINE_AA)
            img = cv2.putText(img, "-->", (20+2*x+y-7, 60), font, fontScale_, (255, 255, 255), 2, cv2.LINE_AA)
            img = cv2.putText(img, "-->", (20+3*x+2*y-7, 60), font, fontScale_, (255, 255, 255), 2, cv2.LINE_AA)
            img = cv2.putText(img, "-->", (20+4*x+3*y-7, 60), font, fontScale_, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.imshow('img',img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        
    else:
        break
 
movie.release()
 
cv2.destroyAllWindows()