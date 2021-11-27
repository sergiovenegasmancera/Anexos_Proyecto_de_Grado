import cv2
import numpy as np
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


def Feature_Extractor_1_Etapa(face, Type_features):
    
    h, w, c = face.shape
    h1 = int(h*0.398524875)

    upper_part = face[0:h1, 0:w]
    lower_part = face[h1+1:h, 0:w]
    
    if Type_features == "Color_quotient":
        
        # Separación de canales - Upper Part
        (B, G, R) = cv2.split(upper_part)
        mBu, mGu, mRu = np.mean(B), np.mean(G), np.mean(R)

        # Separación de canales - Lower Part
        (B, G, R) = cv2.split(lower_part)
        mBl, mGl, mRl = np.mean(B), np.mean(G), np.mean(R)
        
        rgl = mRl/mGl
        rgu = mRu/mGu

        rgq = rgl/rgu

        rbl = mRl/mBl
        rbu = mRu/mBu

        rbq = rbl/rbu

        X = np.array([[rgl, rgu, rbl, rbu, rgq, rbq]])
    
    else:
       
       ##################################################
        M_b_0, M_g_0, M_r_0 = get_moments(upper_part)
        contrast_0, dissimilarity_0, homogeneity_0, energy_0, correlation_0, asm_0 = get_texture_features(upper_part)
        
        ##################################################
        M_b_1, M_g_1, M_r_1 = get_moments(lower_part)
        contrast_1, dissimilarity_1, homogeneity_1, energy_1, correlation_1, asm_1 = get_texture_features(lower_part)
        
        ###################################################
            
        M_b = np.divide(M_b_0, M_b_1)
        M_g = np.divide(M_g_0, M_g_1)
        M_r = np.divide(M_r_0, M_r_1)
        contrast = arr1_divide_by_arr2(contrast_0, contrast_1)
        dissimilarity = arr1_divide_by_arr2(dissimilarity_0, dissimilarity_1)
        homogeneity = arr1_divide_by_arr2(homogeneity_0, homogeneity_1)
        energy = arr1_divide_by_arr2(energy_0, energy_1)
        correlation = arr1_divide_by_arr2(correlation_0, correlation_1)
        asm = arr1_divide_by_arr2(asm_0, asm_1)
       
        X = np.array([[contrast[0], contrast[1], contrast[2], contrast[3], dissimilarity[0],dissimilarity[1], dissimilarity[2], dissimilarity[3], homogeneity[0], homogeneity[1], homogeneity[2], homogeneity[3], energy[0], energy[1], energy[2], energy[3], correlation[0], correlation[1], correlation[2], correlation[3], asm[0], asm[1], asm[2], asm[3], M_b[0], M_b[1], M_b[2], M_g[0], M_g[1], M_g[2], M_r[0], M_r[1], M_r[2]]])
       
    return X


def Feature_Extractor_2_Etapa(face, Type_features):
    
    
    h, w, c = face.shape
    h1 = int(h*0.398524875)
    h2 = int(h*0.655057318)

    upper_part = face[0:h1, 0:w]
    half_part =  face[h1+1:h2, 0:w]
    lower_part = face[h2+1:h, 0:w]
    
    if Type_features == "Color_quotient":
        
        # Separación de canales - Upper Part
        (B, G, R) = cv2.split(upper_part)
        mBu, mGu, mRu = np.mean(B), np.mean(G), np.mean(R)
        
        # Separación de canales - Half Part
        (B, G, R) = cv2.split(half_part)
        mBh, mGh, mRh = np.mean(B), np.mean(G), np.mean(R)

        # Separación de canales - Lower Part
        (B, G, R) = cv2.split(lower_part)
        mBl, mGl, mRl = np.mean(B), np.mean(G), np.mean(R)
        
        rgl1 = mRh/mGh
        rgu1 = mRu/mGu

        rgq1 = rgl1/rgu1

        rbl1 = mRh/mBh
        rbu1 = mRu/mBu

        rbq1 = rbl1/rbu1
        
        
        rgl2 = mRl/mGl
        rgu2 = mRu/mGu

        rgq2 = rgl2/rgu2

        rbl2 = mRl/mBl
        rbu2 = mRu/mBu

        rbq2 = rbl2/rbu2
        
        
        X = np.array([[rgl1, rgu1, rbl1, rbu1, rgq1, rbq1, rgl2, rgu2, rbl2, rbu2, rgq2, rbq2]])
    
    else:
    
        M_b_0, M_g_0, M_r_0 = get_moments(upper_part)
        contrast_0, dissimilarity_0, homogeneity_0, energy_0, correlation_0, asm_0 = get_texture_features(upper_part)

        ##################################################
        M_b_1, M_g_1, M_r_1 = get_moments(half_part)
        contrast_1, dissimilarity_1, homogeneity_1, energy_1, correlation_1, asm_1 = get_texture_features(half_part)

        ##################################################
        M_b_2, M_g_2, M_r_2 = get_moments(lower_part)
        contrast_2, dissimilarity_2, homogeneity_2, energy_2, correlation_2, asm_2 = get_texture_features(lower_part)

        #######################################################

        M_b0 = np.divide(M_b_0, M_b_1)
        M_g0 = np.divide(M_g_0, M_g_1)
        M_r0 = np.divide(M_r_0, M_r_1)
        contrast0 = arr1_divide_by_arr2(contrast_0, contrast_1)
        dissimilarity0 = arr1_divide_by_arr2(dissimilarity_0, dissimilarity_1)
        homogeneity0 = arr1_divide_by_arr2(homogeneity_0, homogeneity_1)
        energy0 = arr1_divide_by_arr2(energy_0, energy_1)
        correlation0 = arr1_divide_by_arr2(correlation_0, correlation_1)
        asm0 = arr1_divide_by_arr2(asm_0, asm_1)

        #######################################################

        M_b1 = np.divide(M_b_0, M_b_2)
        M_g1 = np.divide(M_g_0, M_g_2)
        M_r1 = np.divide(M_r_0, M_r_2)
        contrast1 = arr1_divide_by_arr2(contrast_0, contrast_2)
        dissimilarity1 = arr1_divide_by_arr2(dissimilarity_0, dissimilarity_2)
        homogeneity1 = arr1_divide_by_arr2(homogeneity_0, homogeneity_2)
        energy1 = arr1_divide_by_arr2(energy_0, energy_2)
        correlation1 = arr1_divide_by_arr2(correlation_0, correlation_2)
        asm1 = arr1_divide_by_arr2(asm_0, asm_2)

        #######################################################

        M_b2 = np.divide(M_b_1, M_b_2)
        M_g2 = np.divide(M_g_1, M_g_2)
        M_r2 = np.divide(M_r_1, M_r_2)
        contrast2 = arr1_divide_by_arr2(contrast_1, contrast_2)
        dissimilarity2 = arr1_divide_by_arr2(dissimilarity_1, dissimilarity_2)
        homogeneity2 = arr1_divide_by_arr2(homogeneity_1, homogeneity_2)
        energy2 = arr1_divide_by_arr2(energy_1, energy_2)
        correlation2 = arr1_divide_by_arr2(correlation_1, correlation_2)
        asm2 = arr1_divide_by_arr2(asm_1, asm_2)


        X = np.array([[contrast0[0], contrast0[1], contrast0[2], contrast0[3], dissimilarity0[0],dissimilarity0[1], dissimilarity0[2], dissimilarity0[3], homogeneity0[0], homogeneity0[1], homogeneity0[2], homogeneity0[3], energy0[0], energy0[1], energy0[2], energy0[3], correlation0[0], correlation0[1], correlation0[2], correlation0[3], asm0[0], asm0[1], asm0[2], asm0[3], M_b0[0], M_b0[1], M_b0[2], M_g0[0], M_g0[1], M_g0[2], M_r0[0], M_r0[1], M_r0[2], contrast1[0], contrast1[1], contrast1[2], contrast1[3], dissimilarity1[0],dissimilarity1[1], dissimilarity1[2], dissimilarity1[3], homogeneity1[0], homogeneity1[1], homogeneity1[2], homogeneity1[3], energy1[0], energy1[1], energy1[2], energy1[3], correlation1[0], correlation1[1], correlation1[2], correlation1[3], asm1[0], asm1[1], asm1[2], asm1[3], M_b1[0], M_b1[1], M_b1[2], M_g1[0], M_g1[1], M_g1[2], M_r1[0], M_r1[1], M_r1[2], contrast2[0], contrast2[1], contrast2[2], contrast2[3], dissimilarity2[0],dissimilarity2[1], dissimilarity2[2], dissimilarity2[3], homogeneity2[0], homogeneity2[1], homogeneity2[2], homogeneity2[3], energy2[0], energy2[1], energy2[2], energy2[3], correlation2[0], correlation2[1], correlation2[2], correlation2[3], asm2[0], asm2[1], asm2[2], asm2[3], M_b2[0], M_b2[1], M_b2[2], M_g2[0], M_g2[1], M_g2[2], M_r2[0], M_r2[1], M_r2[2]]])

    return X
    