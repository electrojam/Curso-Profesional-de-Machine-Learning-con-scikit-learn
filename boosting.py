import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == "__main__":

    dt_heart = pd.read_csv("./data/heart.csv")
    print(dt_heart['target'].describe())

    X = dt_heart.drop(['target'], axis=1)
    y = dt_heart['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.35, random_state=42)

    boost = GradientBoostingClassifier(n_estimators=50).fit(X_train, y_train) #definimos función boosting mediante gradient y entrenamos
    boost_pred = boost.predict(X_test) #realizamos las predicciones
    print("="*64)
    print("Gradient Boosting accuracy:", accuracy_score(boost_pred, y_test))