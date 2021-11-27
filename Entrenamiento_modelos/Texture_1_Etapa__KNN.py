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
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score

############################################################################################################
dataframe = pd.read_csv(r'/home/pi/Desktop/Proyecto_de_Grado/Features/Texture_1_Etapa.csv')
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
#joblib.dump(scaler, '/home/pi/Desktop/Proyecto_de_Grado/Modelos/Scalers/Scaler_Texture_1_Etapa.pkl') # Guardo el modelo.
############################################################################################################

x = int(math.sqrt(X_train.shape[0]))
print(x)
porcentaje = int(0.10*x)
param_grid = [
        {
            'n_neighbors' : range(x-porcentaje, x+porcentaje),
            'metric' : ['minkowski', 'manhattan', 'euclidean', 'chebyshev'],
            'weights' : ['distance'],
            'algorithm' : ['auto']
        }
       ]


classifier = GridSearchCV(KNeighborsClassifier(), param_grid, scoring='accuracy')
classifier.fit(X_train, y_train)

print("Best parameters set found on development set:")
print(classifier.best_params_)

classifier = KNeighborsClassifier(algorithm = 'auto', metric = 'minkowski', n_neighbors =  66, weights = 'distance')
classifier.fit(X_train, y_train)
y_test_predicted = classifier.predict(X_test)
y_test_scores = classifier.predict_proba(X_test)
y_test_scores = y_test_scores[:,1]
MCC = matthews_corrcoef(y_test, y_test_predicted)
print("matthews_corrcoef", MCC)
ACC = accuracy_score(y_test, y_test_predicted)
print("Accuracy", ACC)

joblib.dump(classifier, '/home/pi/Desktop/Proyecto_de_Grado/Modelos/Classifiers/KNN/Clasificador_Texture_1_Etapa.pkl') # Guardo el modelo.


clf = KNeighborsClassifier(algorithm = 'auto', metric = 'minkowski', n_neighbors =  66, weights = 'distance')
scores = cross_val_score(clf, X_train, y_train, cv = 5)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
print('Best:', max(scores))

