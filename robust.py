import pandas as pd
 
from sklearn.linear_model import (  #importamos nuestros dos modelos 
    RANSACRegressor, HuberRegressor
)

from  sklearn.svm import SVR #modelo de máquina s de soporte vectorial

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':
    dataset = pd.read_csv('./data/felicidad_corrupt.csv')

    print(dataset.head(5))

    X = dataset.drop(['country', 'score'], axis=1) #definimos nuestros features, haciendo drop de la scolumnas que no necesitamos
    y = dataset[['score']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) # particionamos los datos

    estimadores = { #creamos diccionario con funciones 3 regularizadoras
        'SVR' : SVR(gamma = 'auto', C=1.0, epsilon=0.1),
        'RANSAC': RANSACRegressor(),
        'HUBER' : HuberRegressor(epsilon=1.35)
    }

    for name, estimador in estimadores.items(): #automatizamos implementación de nuestros 3 modelos
        estimador.fit(X_train, y_train)
        predictions = estimador.predict(X_test)
        print("="*64)
        print(name)
        print("MSE (mean squared error): ", mean_squared_error(y_test, predictions))


