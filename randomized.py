import pandas as pd

from sklearn.model_selection import RandomizedSearchCV  # importamos herramienta de optimización
from sklearn.ensemble import RandomForestRegressor # método de ensamble a utilizar tipo regressión

if __name__ == '__main__':

    dataset = pd.read_csv('./data/felicidad.csv')
    print(dataset)

    X = dataset.drop(['country', 'rank', 'score'], axis=1)
    y = dataset['score']

    reg = RandomForestRegressor() # definimos regresor a utilizar, no configuramos parámetros porque es lo que vamos a configurar a través de randomized
    parametros = {  # definimos la grilla de parámetros, diccionario, que va a utilizar nuestro optimizador
        'n_estimators' : range(4, 15), # entre 4 y 15 estimadores o árboles
        'criterion' : ['squared_error', 'absolute_error'], # medida de calidad de los secript que hace mi árbol, que tan bueno fue
        'max_depth' : range(2, 11) # limitamos que tan profundo es nuetro árbol
    } 

    rand_est = RandomizedSearchCV(reg, parametros, n_iter=10, cv=3, scoring='neg_mean_absolute_error').fit(X, y) # optimizamos nuestro estimador

    print(rand_est.best_estimator_) # arroja el regresor con los mejores parámetros de los que se habían definido
    print(rand_est.best_params_)    # arroja el regresor con los mejores parámetros de los que se habían definido
    print(rand_est.predict(X.loc[[0]])) # hacemos predicción sobre el dato del primer país, primera fila con el regresor con mejores parámetros,

    #El resultado arroja una predicción de feature score = 7.52720008 y
    # en el dataset felicidad, en el primer país el score = 7.594444821 