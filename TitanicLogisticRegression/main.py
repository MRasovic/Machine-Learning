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
    train_df = pd.read_csv(
        "C:/Users/Korisnik/Desktop/Refactored_Py_DS_ML_Bootcamp-master/13-Logistic-Regression/titanic_train.csv")

    # Exploratory analysis

    plt.figure()
    sns.heatmap(train_df.isnull(), yticklabels=False, cbar=False, cmap="viridis")

    sns.set_style('whitegrid')

    plt.figure()
    sns.countplot(x='Survived', hue='Sex', data=train_df, palette='RdBu_r')

    plt.figure()
    sns.countplot(x='Survived', hue='Pclass', data=train_df)

    plt.figure()
    sns.countplot(x="SibSp", data=train_df)

    plt.figure()
    train_df["Fare"].hist(bins=40)

    plt.figure()
    sns.boxplot(x="Pclass", y="Age", data=train_df)

    #
    # Cleaning the data
    #

    # Removing Nan values
    average_age = train_df.groupby("Pclass")["Age"].mean()
    print(average_age)


    def impute_age(cols):
        Age = cols[0]
        Pclass = cols[1]

        if pd.isnull(Age):

            if Pclass == 1:
                return 38

            elif Pclass == 2:
                return 30

            else:
                return 25

        else:
            return Age


    train_df["Age"] = train_df[["Age", "Pclass"]].apply(impute_age, axis=1)
    train_df.drop("Cabin", axis=1, inplace=True)

    # Adding dummy values
    sex = pd.get_dummies(train_df["Sex"], drop_first=True)
    embark = pd.get_dummies(train_df["Embarked"], drop_first=True)
    train_df = pd.concat([train_df, sex, embark], axis=1)  # add the dummy columns to the dataframe

    # Dropping unnecessary columns
    train_df.drop(["Sex", "Embarked", "Name", "Ticket"], axis=1, inplace=True)
    train_df.drop("PassengerId", axis=1, inplace=True)

    #
    # Logistic Regression
    #

    X = train_df.drop("Survived", axis=1)
    y = train_df["Survived"]

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.3, random_state=101)

    from sklearn.linear_model import LogisticRegression

    logmodel = LogisticRegression(max_iter=1000)
    logmodel.fit(X_train, y_train)

    predictions = logmodel.predict(X_test)

    print(classification_report(y_test, predictions))

    plt.close('all')
    plt.show()

    # train_df.info(verbose=True)
    # print(train_df.shape[0])
    # print(train_df.head())
