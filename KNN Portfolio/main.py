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
    df = pd.read_csv(
        "C:/Users/Korisnik/Desktop/Refactored_Py_DS_ML_Bootcamp-master/14-K-Nearest-Neighbors/KNN_Project_Data")

    print(df.head())
    print(df.shape[0])

    # plt.figure()
    # sns.pairplot(data=df, hue="TARGET CLASS", palette='coolwarm')
    # plt.show()

    from sklearn.preprocessing import StandardScaler

    # Instancing the scaler
    scaler = StandardScaler()
    # Fitting the scaler to the target column
    scaler.fit(df.drop("TARGET CLASS", axis=1))
    # transforming the other columns to the target class scale
    scaler_feature = scaler.transform(df.drop("TARGET CLASS", axis=1))
    column_names = df.columns[:-1]
    print(column_names)
    df_feat = pd.DataFrame(scaler_feature, columns=df.columns[:-1])

    print(df_feat.head())
    print(df_feat.shape[0])

    # Splitting the data
    X = df_feat
    y = df["TARGET CLASS"]
    print(X.shape[0])
    print(y.shape[0])

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

    from sklearn.neighbors import KNeighborsClassifier
    # n neighbors defines the K parameter
    knn = KNeighborsClassifier(n_neighbors=1)

    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)

    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred))

    # Assessing the optimal value for K
    error_rate = []

    for i in range(1, 40):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        error_rate.append(pred_i != y_test)
    
    plt.figure(figsize=(10, 6))
    plt.plot(error_rate, range(1, 40),  color='blue', linestyle='dashed', marker='o',
             markerfacecolor='red', markersize=6)
    plt.title('Error Rate vs. K Value')
    plt.xlabel('K')
    plt.ylabel('Error Rate')
    plt.show()
