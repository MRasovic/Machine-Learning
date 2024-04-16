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
        "C:/Users/Korisnik/Desktop/Refactored_Py_DS_ML_Bootcamp-master/15-Decision-Trees-and-Random-Forests/loan_data.csv")

    print(df.head())
    print(df.info())
    print(df.describe())

    # Data Analysis

    # FICO score and credit policy relation
    plt.figure(figsize=(10, 6))  # Taking into account the large overlapping histograms
    df[df['credit.policy'] == 1]['fico'].hist(alpha=0.5, color='blue',
                                              bins=30,
                                              label='Credit.Policy=1')  # Using pandas to separate into 2 histograms
    df[df['credit.policy'] == 0]['fico'].hist(alpha=0.5, color='red',
                                              bins=30, label='Credit.Policy=0')  # Alpha being transparency
    plt.legend()
    plt.xlabel('FICO')

    # FICO score and paid loans
    plt.figure(figsize=(10, 6))
    df[df["not.fully.paid"] == 1]["fico"].hist(alpha=0.7, label="Fully paid", bins=35, color="green")
    df[df["not.fully.paid"] == 0]["fico"].hist(alpha=0.5, label="Not Fully paid", bins=35)

    plt.legend()
    plt.xlabel("FICO")

    # Count plot
    plt.figure(figsize=(10, 6))
    grouped_up = df.groupby("purpose").count()
    sns.countplot(grouped_up, x=df["purpose"], hue=df["not.fully.paid"])

    # Joint pot
    plt.figure(figsize=(10, 6))
    sns.jointplot(data=df, x="fico", y="int.rate")

    # plt.show()

    # Adding dummy values
    """
    purp = pd.get_dummies(df["purpose"])
    df = pd.concat([df, purp], axis=1)
    """
    cat_feats = ["purpose"]
    purp = pd.get_dummies(df, columns=cat_feats, drop_first=True)

    print(df.head())
    print(purp.info())

    # Train test split
    from sklearn.model_selection import train_test_split

    X = purp.drop("not.fully.paid", axis=1)
    y = purp["not.fully.paid"]
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

    ran_forest = RandomForestClassifier(n_estimators=200)
    ran_forest.fit(X_train, y_train)

    predict_forest = ran_forest.predict(X_test)
    print(confusion_matrix(y_test, predict_forest))
    print(classification_report(y_test, predict_forest))
