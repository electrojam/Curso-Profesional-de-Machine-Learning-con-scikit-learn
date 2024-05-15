import pandas as pd
import sklearn

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso  #modelo de regularizacion
from sklearn.linear_model import Ridge  # otro modelo de regularizacion

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':
    dataset = pd.read_csv('./data/whr2017.csv')

    print(dataset.head(5))
    print(dataset.describe())

    #Vamos a dividir df en features y targets

    X = dataset[['gdp', 'family', 'lifexp', 'freedom', 'corruption', 'generosity', 'dystopia']] # definios nuestra variable que contendrá los features para realizar nuestra predicción
    y = dataset['score']    #definimos nuestro target

    print(X.shape)
    print(y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42) #divid datos 25% test, 75% train

    #Definimos 3 modelos para luego comparar 

    modelLinear = LinearRegression().fit(X_train, y_train) #definimos nuestro regresor lineal
    y_predict_linear = modelLinear.predict(X_test)

    modelLasso = Lasso(alpha=0.02).fit(X_train, y_train) #deficinimos nuestro regresor Lasso, entre mayor alpha mayor penalización para los features
    y_predict_lasso = modelLasso.predict(X_test) # realizamos predicción para comprobar si se mejor o peor

    modelRidge = Ridge(alpha=1).fit(X_train, y_train) #definimos nuestra regresión Ridge
    y_predict_ridge = modelRidge.predict (X_test)

    #Comparamos modelos

    linear_loss = mean_squared_error(y_test, y_predict_linear)
    print("Linear Loss:", linear_loss)

    lasso_loss = mean_squared_error(y_test, y_predict_lasso)
    print("Lasso Loss:", lasso_loss)

    ridge_loss = mean_squared_error(y_test, y_predict_ridge)
    print("Ridge Loss:", ridge_loss)

    '''
    -- Entre menor sea el Loss menor pérdida, menor equivocacón entre lo esperado y lo predicho
    El resultado anterior arroja que ganó el modelo Ridge porque tiene menor valor de Loss
    '''

    print("="*32)
    print("Coef LASSO")
    print(modelLasso.coef_)

    print("="*32)
    print("Coef RIDGE")
    print(modelRidge.coef_)

    '''
    -- El resultado arroja un arreglo de 7 columnas, que corresponden a los features del df X que
    partimos al principio
    -- en el arreglo de 7 columnas, los números más grandes quiere decir que esa columna o feature tiene 
    más peso en el modelo entrenado, para este caso el feature 'gdp' tiene mucho más peso en
    -- Por otro lado en el modelo Lasso quitó el feature de corrupción 0. porue no lo encontró
    determinante para desarrollar el modelo.
    -- en modelo Ridge ninguno de los coeficientes de los features fue a 0.
    '''