import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report, confusion_matrix

pd.set_option('display.max_columns', None)
"pd.set_option('display.max_rows', None) "  # Display all rows
pd.set_option('display.width', 1000)  # Set the overall width of the display
pd.set_option('display.max_colwidth', None)

if __name__ == '__main__':

    df = pd.read_csv("C:/Users/Korisnik/Desktop/Refactored_Py_DS_ML_Bootcamp-master/15-Decision-Trees-and-Random-Forests/kyphosis.csv")

    print(df.head())
    print(df.info())

    # sns.pairplot(df, hue="Kyphosis")
    # plt.show()

    from sklearn.model_selection import train_test_split

    X = df.drop("Kyphosis", axis=1)
    y = df["Kyphosis"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

    # Decision tree
    from sklearn.tree import DecisionTreeClassifier
    dtree = DecisionTreeClassifier()
    dtree.fit(X_train, y_train)
    
    predictions = dtree.predict(X_test)
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))

    # Random forest
    from sklearn.ensemble import RandomForestClassifier
    rfc = RandomForestClassifier(n_estimators=250)
    rfc.fit(X_train,y_train)

    rfc_pred = rfc.predict(X_test)
    print(confusion_matrix(y_test, rfc_pred))
    print(classification_report(y_test, rfc_pred))

    # Tree visualization
