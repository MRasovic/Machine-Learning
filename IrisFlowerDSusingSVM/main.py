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
    iris = sns.load_dataset("iris")
    print(iris.keys())
    print(iris.head())
    print(iris.shape)

    #
    # Exploratory Data Analysis
    #

    # plt.figure(figsize=(10, 8))
    # sns.pairplot(data=iris, hue="species")

    # plt.figure(figsize=(10, 8))
    # sns.kdeplot(iris, y="sepal_length", x="sepal_width", fill=True)

    # plt.show()

    #
    # Splitting training and evaluating the model
    #
    X = iris.drop("species", axis=1)
    y = iris["species"]

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

    from sklearn.svm import SVC

    model = SVC()
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    print(classification_report(y_test, pred))
    print(confusion_matrix(y_test, pred))

    #
    # Gridsearch Practice
    #

    from sklearn.model_selection import GridSearchCV

    param_grid = {"C": [0.1, 1, 10, 100], "gamma": [1, 0.1, 0.01, 0.001]}  # Defining gridsearch parameters in a dictionary

    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
    grid.fit(X_train, y_train)

    grid_predictions = grid.predict(X_test)
    print(confusion_matrix(y_test, grid_predictions))
    print(classification_report(y_test, grid_predictions))