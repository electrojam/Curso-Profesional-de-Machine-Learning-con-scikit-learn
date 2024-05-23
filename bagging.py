import pandas as pd

from sklearn.neighbors import KNeighborsClassifier #modelo para utilizar en el clasificador
from sklearn.ensemble import BaggingClassifier #método de ensamble BaggingClassifier

from sklearn.model_selection import train_test_split #herramienta para particionar datos
from sklearn.metrics import accuracy_score #herramienta para medir accuracy, precisión, éxito del modelo

if __name__ == '__main__':  #definicmos este como archivo principal a ejecutar

    dt_heart = pd.read_csv('./data/heart.csv')  #cargue de df
    print(dt_heart['target'].describe())    #revisamos solo columna, variable target, y la describimos

    X = dt_heart.drop(['target'], axis=1) #guardamos en X df excepto variable target
    y = dt_heart['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.35, random_state=42)

    knn_class = KNeighborsClassifier().fit(X_train, y_train)    #definimos clasificador knn
    knn_pred = knn_class.predict(X_test)    #realizamos las predicciones
    print("="*64)
    print("Knn classifier accuracy score:", accuracy_score(knn_pred, y_test))

    bag_class = BaggingClassifier(estimator=KNeighborsClassifier(), n_estimators=50).fit(X_train, y_train)    #definimos clasificador assamble, y entrenamos
    bag_pred = bag_class.predict(X_test)    #realizamos predicción
    print("="*64)
    print("Bag classifier accuracy score:", accuracy_score(bag_pred, y_test))