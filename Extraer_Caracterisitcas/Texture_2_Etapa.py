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



header = 'contrast_1[0] contrast_1[1] contrast_1[2] contrast_1[3] dissimilarity_1[0] dissimilarity_1[1] dissimilarity_1[2] dissimilarity_1[3] homogeneity_1[0] homogeneity_1[1] homogeneity_1[2] homogeneity_1[3] energy_1[0] energy_1[1] energy_1[2] energy_1[3] correlation_1[0] correlation_1[1] correlation_1[2] correlation_1[3] asm_1[0] asm_1[1] asm_1[2] asm_1[3] M_b_1[0] M_b_1[1] M_b_1[2] M_g_1[0] M_g_1[1] M_g_1[2] M_r_1[0] M_r_1[1] M_r_1[2] contrast_2[0] contrast_2[1] contrast_2[2] contrast_2[3] dissimilarity_2[0] dissimilarity_2[1] dissimilarity_2[2] dissimilarity_2[3] homogeneity_2[0] homogeneity_2[1] homogeneity_2[2] homogeneity_2[3] energy_2[0] energy_2[1] energy_2[2] energy_2[3] correlation_2[0] correlation_2[1] correlation_2[2] correlation_2[3] asm_2[0] asm_2[1] asm_2[2] asm_2[3] M_b_2[0] M_b_2[1] M_b_2[2] M_g_2[0] M_g_2[1] M_g_2[2] M_r_2[0] M_r_2[1] M_r_2[2] contrast_3[0] contrast_3[1] contrast_3[2] contrast_3[3] dissimilarity_3[0] dissimilarity_3[1] dissimilarity_3[2] dissimilarity_3[3] homogeneity_3[0] homogeneity_3[1] homogeneity_3[2] homogeneity_3[3] energy_3[0] energy_3[1] energy_3[2] energy_3[3] correlation_3[0] correlation_3[1] correlation_3[2] correlation_3[3] asm_3[0] asm_3[1] asm_3[2] asm_3[3] M_b_3[0] M_b_3[1] M_b_3[2] M_g_3[0] M_g_3[1] M_g_3[2] M_r_3[0] M_r_3[1] M_r_3[2] h w h/w label'.split()
file = open('/home/pi/Desktop/Proyecto_de_Grado/Features/Texture_2_Etapa.csv', 'w', newline='')

with file:
    writer = csv.writer(file)
    writer.writerow(header)
labels = 'Incorrect_Mask With_Mask'.split()

for l in labels:
    for filename in os.listdir(f'/home/pi/Desktop/Proyecto_de_Grado/Base_de_Datos_2/2_Etapa_Clasificacion/{l}'):
        file = f'/home/pi/Desktop/Proyecto_de_Grado/Base_de_Datos_2/2_Etapa_Clasificacion/{l}/{filename}'
        face = cv2.imread(file)
        h, w, c = face.shape
        if h > 0 and w > 0:
            
            X = Feature_Extractor_2_Etapa(face, "Texture")
            
            to_append = f'{X[0][0]}'
            
            for i in range(1, X.shape[1]):
                to_append = to_append + ' ' + f'{X[0][i]}'
                
            to_append = to_append + ' ' + f'{h}' + ' ' + f'{w}' + ' ' + f'{h/w}' + ' ' + f'{l}'
                
            file = open('/home/pi/Desktop/Proyecto_de_Grado/Features/Texture_2_Etapa.csv', 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())