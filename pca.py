import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

from sklearn.linear_model import LogisticRegression # para comparar PCA y Incremental PCA

from sklearn.preprocessing import StandardScaler # escalar datos entre 0 y 1
from sklearn.model_selection import train_test_split # módulo separar datos

if __name__ == "__main__":
    dt_heart = pd.read_csv('./data/heart.csv') #guarda df en dt_heart

    print(dt_heart.head(5)) #imprimimos primeras 5 filas del df

    dt_features = dt_heart.drop(['target'], axis=1) #guardamos solo var features
    dt_target = dt_heart['target']  #guardamos solo var target

    dt_features = StandardScaler().fit_transform(dt_features)#normalizamos datos var features

    X_train, X_test, y_train, y_test = train_test_split(dt_features, dt_target, test_size=0.30, random_state=42) #partimos conjunto de datos
    
