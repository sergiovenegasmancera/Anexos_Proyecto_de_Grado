from Feature_Extractor import *
import csv
import cv2
import numpy as np
import imutils
import os
from sklearn.preprocessing import *
from skimage.measure import moments
from skimage.feature import greycomatrix, greycoprops
from skimage import io, color, img_as_ubyte



def getGLCM(gray):
    image = img_as_ubyte(gray)

    bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]) #16-bit
    inds = np.digitize(image, bins)

    max_value = inds.max()+1
    matrix_coocurrence = greycomatrix(inds, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=max_value, normed=False, symmetric=False)
    return matrix_coocurrence

# GLCM properties
def contrast_feature(matrix_coocurrence):
    contrast = greycoprops(matrix_coocurrence, 'contrast')
    return "Contrast = ", contrast

def dissimilarity_feature(matrix_coocurrence):
    dissimilarity = greycoprops(matrix_coocurrence, 'dissimilarity')
    return "Dissimilarity = ", dissimilarity

def homogeneity_feature(matrix_coocurrence):
    homogeneity = greycoprops(matrix_coocurrence, 'homogeneity')
    return "Homogeneity = ", homogeneity

def energy_feature(matrix_coocurrence):
    energy = greycoprops(matrix_coocurrence, 'energy')
    return "Energy = ", energy

def correlation_feature(matrix_coocurrence):
    correlation = greycoprops(matrix_coocurrence, 'correlation')
    return "Correlation = ", correlation

def asm_feature(matrix_coocurrence):
    asm = greycoprops(matrix_coocurrence, 'ASM')
    return "ASM = ", asm

def get_moments(image):
    
    b, g, r = cv2.split(image)
    
    M = moments(b,2)
    M_b = [M[0][0], M[0][1],M[0][2]]

    M = moments(g,2)
    M_g = [M[0][0], M[0][1],M[0][2]]

    M = moments(r,2)
    M_r = [M[0][0], M[0][1],M[0][2]]

    return M_b, M_g, M_r

def get_texture_features(image):
    
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    matrix_coocurrence = getGLCM(img_gray)

    contrast = contrast_feature(matrix_coocurrence)
    dissimilarity = dissimilarity_feature(matrix_coocurrence)
    homogeneity = homogeneity_feature(matrix_coocurrence)
    energy = energy_feature(matrix_coocurrence)
    correlation = correlation_feature(matrix_coocurrence)
    asm = asm_feature(matrix_coocurrence)

    return contrast, dissimilarity, homogeneity, energy, correlation, asm

def arr1_divide_by_arr2(arr1, arr2):
    
    array1 = [arr1[1][0][0], arr1[1][0][1], arr1[1][0][2], arr1[1][0][3]]
    array2 = [arr2[1][0][0], arr2[1][0][1], arr2[1][0][2], arr2[1][0][3]]
    
    return np.divide(array2, array1)
    



header = 'contrast[0] contrast[1] contrast[2] contrast[3] dissimilarity[0] dissimilarity[1] dissimilarity[2] dissimilarity[3] homogeneity[0] homogeneity[1] homogeneity[2] homogeneity[3] energy[0] energy[1] energy[2] energy[3] correlation[0] correlation[1] correlation[2] correlation[3] asm[0] asm[1] asm[2] asm[3] M_b[0] M_b[1] M_b[2] M_g[0] M_g[1] M_g[2] M_r[0] M_r[1] M_r[2] label'.split()
file = open('/home/pi/Desktop/Proyecto_de_Grado/Features/Texture_1_Etapa.csv', 'w', newline='')


with file:
    writer = csv.writer(file)
    writer.writerow(header)
labels = 'With_Mask Without_Mask'.split()

for l in labels:
    for filename in os.listdir(f'/home/pi/Desktop/Proyecto_de_Grado/Base_de_Datos_2/1_Etapa_Clasificacion/{l}'):
        file = f'/home/pi/Desktop/Proyecto_de_Grado/Base_de_Datos_2/1_Etapa_Clasificacion/{l}/{filename}'
        face = cv2.imread(file)
        h, w, c = face.shape
        if h > 0 and w > 0:
            
            X = Feature_Extractor_1_Etapa(face, "Texture")
            
            to_append = f'{X[0][0]}'
            
            for i in range(1, X.shape[1]):
                to_append = to_append + ' ' + f'{X[0][i]}'
                
            to_append = to_append + ' ' + f'{l}'
            
            file = open('/home/pi/Desktop/Proyecto_de_Grado/Features/Texture_1_Etapa.csv', 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())