import pandas as pd

from sklearn.cluster import MeanShift

if __name__ == "__main__":

    dataset = pd.read_csv('./data/candy.csv')
    print(dataset.head(5))

    X = dataset.drop('competitorname', axis=1) #eliminamos columna competitorname por que es categórica

    meanshift = MeanShift().fit(X) #configuramos algoritmo meanshift
    print(meanshift.labels_)    #miramos los labels que creó para clasificar los datos
    print("="*64)
    print(max(meanshift.labels_))   #miramos cuantas etiquetas creó para clasificar los datos
    print("="*64)
    print(meanshift.cluster_centers_) #miramos los centros de los grups de datos.  Mostrará un arreglo por cada centro

    dataset['meanshift'] = meanshift.labels_    #agregamos las etiquetas creadas al dataset original
    print("="*64)
    print(dataset)