import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import *
from sklearn.preprocessing import label_binarize
from sklearn.utils import shuffle
from sklearn import svm
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import DistanceMetric
import math
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from os import system
from keras.models import Sequential
from keras.layers import Dense
from sklearn import preprocessing
from tensorflow import keras


############################################################################################################
dataframe = pd.read_csv(r'/home/pi/Desktop/Proyecto_de_Grado/Features/Color_quotient_1_Etapa.csv')
############################################################################################################
y = dataframe['label']
X = dataframe.loc[:, dataframe.columns != 'label'] #select all columns but not the labels

#### PCA 2 COMPONENTS ####
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

# concatenate with target label
finalDf = pd.concat([principalDf, y], axis = 1)

pca.explained_variance_ratio_

plt.figure(figsize = (16, 9))
sns.scatterplot(x = "principal component 1", y = "principal component 2", data = finalDf, hue = "label", alpha = 0.7, s = 100);

plt.title('PCA on Genres', fontsize = 25)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 10);
plt.xlabel("Principal Component 1", fontsize = 15)
plt.ylabel("Principal Component 2", fontsize = 15)
plt.savefig("PCA Scattert.jpg")
############################################################################################################

X, y = shuffle(X, y, random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=0)
#scaler = StandardScaler()
scaler = MinMaxScaler()
scaler.fit(X_train)# el fit de los datos solo se hace con el conjunto de entrenamiento!
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
#joblib.dump(scaler, '/home/pi/Desktop/Proyecto_de_Grado/Modelos/Scalers/Scaler_Color_quotient_1_Etapa.pkl') # Guardo el modelo.
############################################################################################################


# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=6, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


le = preprocessing.LabelEncoder()
le.fit(['Without_Mask', 'With_Mask'])
y_train_enc = le.transform(y_train)


# fit the keras model on the dataset
model.fit(X_train, y_train_enc, epochs=500, batch_size=10)

# evaluate the keras model
_, accuracy = model.evaluate(X_test, le.transform(y_test))
print('Accuracy: %.2f' % (accuracy*100))
# Guardar el Modelo
model.save('Color_quotient_1_Etapa_MPL.h5')

