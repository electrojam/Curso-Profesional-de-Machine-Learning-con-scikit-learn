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
    
    print(X_train.shape) # verificamos éxito datos partidos con forma de var
    print(y_train.shape)

    pca = PCA(n_components=3) #configuramos algorit PCA
    pca.fit(X_train)    #entrenamos

    ipca = IncrementalPCA(n_components=3, batch_size=10) #configuramos algorit IPCA, manda entrena datos por lotes
    ipca.fit(X_train)   #entrenamos
    
    plt.plot(range(len(pca.explained_variance_)), pca.explained_variance_ratio_)
    plt.show()
    
    logistic = LogisticRegression(solver='lbfgs') #definimos var con función regresión logística

    dt_train = pca.transform(X_train) #aplicamos PCA a conjuntos de entrenamiento y de prueba
    dt_test = pca.transform(X_test)
    logistic.fit(dt_train, y_train) #aplicamos regresión logística 

    print("SCORE PCA:", logistic.score(dt_test, y_test)) #medimos efectividad del modelo, accuracy

    dt_train = ipca.transform(X_train) #aplicamos IPCA a datos de prueba y entrenamiento
    dt_test = ipca.transform(X_test)
    logistic.fit(dt_train, y_train) #aplicamos regresión logística 

    print("SCORE IPCA:", logistic.score(dt_test, y_test))
    