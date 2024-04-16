import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
"pd.set_option('display.max_rows', None) "  # Display all rows
pd.set_option('display.width', 1000)  # Set the overall width of the display
pd.set_option('display.max_colwidth', None)
palette = sns.color_palette("muted")

if __name__ == '__main__':
    df = pd.read_csv(
        'C:/Users/Korisnik/Desktop/Refactored_Py_DS_ML_Bootcamp-master/11-Linear-Regression/Ecommerce Customers')

    # Sifting through the data

    # print(df.head())
    # print(df.info())
    # print(df.describe())

    # sns.jointplot(df, x=df["Time on Website"], y=df["Yearly Amount Spent"])
    # sns.jointplot(df, x=df["Time on App"], y=df["Yearly Amount Spent"])
    # sns.jointplot(df, x=df["Time on App"], y=df["Length of Membership"], kind="hex")
    # sns.pairplot(df)

    # plt.show()

    #
    #
    # Linear regression
    #
    #

    print(df.columns)
    X = df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
    Y = df['Yearly Amount Spent']

    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=101)

    from sklearn.linear_model import LinearRegression
    lm = LinearRegression()

    print(X.shape[0])

    lm.fit(X_train, Y_train)
    predictions = lm.predict(X_test)

    sales_df = pd.DataFrame({"Feature": X.columns, "Coefficient": lm.coef_})
    print(sales_df)

