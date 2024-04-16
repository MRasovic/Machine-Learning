import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report

pd.set_option('display.max_columns', None)
"pd.set_option('display.max_rows', None) "  # Display all rows
pd.set_option('display.width', 1000)  # Set the overall width of the display
pd.set_option('display.max_colwidth', None)


if __name__ == '__main__':

    ad_data = pd.read_csv("C:/Users/Korisnik/Desktop/Refactored_Py_DS_ML_Bootcamp-master/13-Logistic-Regression/advertising.csv")

    #
    # Exploratory Data Analysis
    #

    plt.figure()
    sns.histplot(data=ad_data, x="Age",bins=30)

    plt.figure()
    sns.jointplot(data=ad_data, x="Age", y="Area Income")

    plt.figure()
    sns.jointplot(data=ad_data, x="Age", y="Area Income", kind="kde", shade=True)

    plt.figure()
    sns.pairplot(data=ad_data)

    # plt.close('all')
    plt.show()



    ad_data.info(verbose = True)
    print(ad_data.head())
    print(ad_data.describe())

