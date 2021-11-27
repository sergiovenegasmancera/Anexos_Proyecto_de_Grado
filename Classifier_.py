import cv2
import numpy as np
import joblib
from sklearn.preprocessing import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn import preprocessing


def Get_Classifier_model(Type_features, Type_Classifier):
    
    
    if Type_features == "Color_quotient":
        
        Scaler_1_Etapa = joblib.load('/home/pi/Desktop/Proyecto_de_Grado/Modelos/Scalers/Scaler_Color_quotient_1_Etapa.pkl')
        Scaler_2_Etapa = joblib.load('/home/pi/Desktop/Proyecto_de_Grado/Modelos/Scalers/Scaler_Color_quotient_2_Etapa.pkl')
        
        if Type_Classifier == "SVM":
            Classifier_1_Etapa = joblib.load('/home/pi/Desktop/Proyecto_de_Grado/Modelos/Classifiers/SVM/Clasificador_Color_quotient_1_Etapa.pkl')
            Classifier_2_Etapa = joblib.load('/home/pi/Desktop/Proyecto_de_Grado/Modelos/Classifiers/SVM/Clasificador_Color_quotient_2_Etapa.pkl') 
            
            
        elif Type_Classifier == "KNN":
            Classifier_1_Etapa = joblib.load('/home/pi/Desktop/Proyecto_de_Grado/Modelos/Classifiers/KNN/Clasificador_Color_quotient_1_Etapa.pkl') 
            Classifier_2_Etapa = joblib.load('/home/pi/Desktop/Proyecto_de_Grado/Modelos/Classifiers/KNN/Clasificador_Color_quotient_2_Etapa.pkl') 
            
            
        elif Type_Classifier == "MLP":
            Classifier_1_Etapa = keras.models.load_model('/home/pi/Desktop/Proyecto_de_Grado/Modelos/Classifiers/MLP/Color_quotient_1_Etapa_MPL.h5') 
            Classifier_2_Etapa = keras.models.load_model('/home/pi/Desktop/Proyecto_de_Grado/Modelos/Classifiers/MLP/Color_quotient_2_Etapa_MPL.h5') 
    
    else:
        
        Scaler_1_Etapa = joblib.load('/home/pi/Desktop/Proyecto_de_Grado/Modelos/Scalers/Scaler_Texture_1_Etapa.pkl')
        Scaler_2_Etapa = joblib.load('/home/pi/Desktop/Proyecto_de_Grado/Modelos/Scalers/Scaler_Texture_2_Etapa.pkl')
        
        if Type_Classifier == "SVM":
            Classifier_1_Etapa = joblib.load('/home/pi/Desktop/Proyecto_de_Grado/Modelos/Classifiers/SVM/Clasificador_Texture_1_Etapa.pkl') 
            Classifier_2_Etapa = joblib.load('/home/pi/Desktop/Proyecto_de_Grado/Modelos/Classifiers/SVM/Clasificador_Texture_2_Etapa.pkl')
            
        elif Type_Classifier == "KNN":
            Classifier_1_Etapa = joblib.load('/home/pi/Desktop/Proyecto_de_Grado/Modelos/Classifiers/KNN/Clasificador_Texture_1_Etapa.pkl') 
            Classifier_2_Etapa = joblib.load('/home/pi/Desktop/Proyecto_de_Grado/Modelos/Classifiers/KNN/Clasificador_Texture_2_Etapa.pkl') 
            
        elif Type_Classifier == "MLP":
            Classifier_1_Etapa = keras.models.load_model('/home/pi/Desktop/Proyecto_de_Grado/Modelos/Classifiers/MLP/Texture_1_Etapa_MPL.h5') 
            Classifier_2_Etapa = keras.models.load_model('/home/pi/Desktop/Proyecto_de_Grado/Modelos/Classifiers/MLP/Texture_2_Etapa_MPL.h5') 


    return Classifier_1_Etapa, Scaler_1_Etapa, Classifier_2_Etapa, Scaler_2_Etapa
        

def classifier(X, classifier_model, scaler, Type_Classifier, encoder):
    
    
    if np.isnan(np.sum(X)):
        y_predicted = "NULL"
        
    else:
        
        X = scaler.transform(X)
        if Type_Classifier == "MLP":
            y_predicted = classifier_model.predict_classes(X)
            y_predicted = encoder.inverse_transform(y_predicted)
        else:
            y_predicted = classifier_model.predict(X)
        
    return y_predicted