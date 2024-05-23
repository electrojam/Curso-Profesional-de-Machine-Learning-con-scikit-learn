import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor  #libreria para predecir variable continúa mediante tree decision

from sklearn.model_selection import (
    cross_val_score, 
    KFold
)

if __name__ == "__main__":

    dataset = pd.read_csv('./data/felicidad.csv') #cargamos dataset

    X = dataset.drop(['country', 'score'], axis=1) #guardamos ds sin feature country ni score
    y = dataset['score']    #guardamos feature score en y


    model = DecisionTreeRegressor() #definimos modelo DecisiionTreeRegressor
    score = cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error') #llamamos función para validación rápida
    print(score)
    print("="*64)
    print(np.abs(np.mean(score))) #solamente calculamos un score

    #Miraremos como funciona cross validation de fondo

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    for train, test in kf.split(dataset):
        print(train)
        print(test)