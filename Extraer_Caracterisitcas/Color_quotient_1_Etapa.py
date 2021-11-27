from Feature_Extractor import *
import numpy as np
import cv2
import csv
import os


header = 'rgl rgu rbl rbu rgq rbq  label'.split()
file = open('/home/pi/Desktop/Proyecto_de_Grado/Features/Color_quotient_1_Etapa.csv', 'w', newline='')


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
            
            X = Feature_Extractor_1_Etapa(face, "Color_quotient")

            to_append = f'{X[0][0]} {X[0][1]} {X[0][2]} {X[0][3]} {X[0][4]} {X[0][5]} {l}'
            file = open('/home/pi/Desktop/Proyecto_de_Grado/Features/Color_quotient_1_Etapa.csv', 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())