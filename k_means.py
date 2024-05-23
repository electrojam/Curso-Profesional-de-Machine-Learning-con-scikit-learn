import pandas as pd

from sklearn.cluster import MiniBatchKMeans #libreria de clustering para pc de bajos recursos

if __name__ == "__main__":
    
    dataset = pd.read_csv("./data/candy.csv")
    print(dataset.head(10))

    # como es problema de aprendizaje no supervisado no se necesita train_test_split, tampoco separar features de target

    X = dataset.drop('competitorname', axis=1)

    #Caso en que sepamos el kmeans=4
    kmeans = MiniBatchKMeans(n_clusters=4, batch_size=8).fit(X)  #definimos algoritmo kmeans, k=4 y batch=grupos 
    print("Total de centros: ", len(kmeans.cluster_centers_))
    print("="*64)
    print(kmeans.predict(X)) #hacemos predicción con qué etiquetas él va a poner a cada uno de los datos 

    dataset['group'] = kmeans.predict(X) # agregamos nueva columna a df X con los datos resultantes de la predicción
    
    print(dataset)
    