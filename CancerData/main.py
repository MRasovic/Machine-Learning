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

    from sklearn.datasets import load_breast_cancer
    cancer = load_breast_cancer()
    print(cancer.keys())

    # turning the data into a pandas DataFrame
    df = pd.DataFrame(cancer["data"], columns=cancer["feature_names"])
    print(df.head())

    # Splitting data
    X = df
    y = cancer["target"]
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

    # Using SVM
    from sklearn.svm import SVC
    sup_model = SVC()
    sup_model.fit(X_train, y_train)

    predictions = sup_model.predict(X_test)
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))
