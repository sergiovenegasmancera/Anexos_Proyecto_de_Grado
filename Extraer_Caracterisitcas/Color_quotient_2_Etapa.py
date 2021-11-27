from Feature_Extractor import *
import numpy as np
import cv2
import csv
import os


header = 'rgl1 rgu1 rbl1 rbu1 rgq1 rbq1 rgl2 rgu2 rbl2 rbu2 rgq2 rbq2 h w h/w label'.split()
file = open('/home/pi/Desktop/Proyecto_de_Grado/Features/Color_quotient_2_Etapa.csv', 'w', newline='')

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
            
            X = Feature_Extractor_2_Etapa(face, "Color_quotient")
            
            to_append = f'{X[0][0]} {X[0][1]} {X[0][2]} {X[0][3]} {X[0][4]} {X[0][5]} {X[0][6]} {X[0][7]} {X[0][8]} {X[0][9]} {X[0][10]} {X[0][11]} {h} {w} {h/w} {l}'
            
            file = open('/home/pi/Desktop/Proyecto_de_Grado/Features/Color_quotient_2_Etapa.csv', 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())