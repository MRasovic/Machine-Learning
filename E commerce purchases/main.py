import pandas as pd
import re


def average_purchase_price():
    print(ecom["Purchase Price"].mean())


def highest_lowest_purchase_price():
    print(ecom["Purchase Price"].min())
    print(ecom["Purchase Price"].max())


def language_of_choice():
    print(ecom[ecom["Language"] == "en"].count())


def num_lawyers():
    print(ecom[ecom["Job"] == "Lawyer"].count())


def am_or_pm():
    ecom["AM or PM"].value_counts()


def get_purchase_price():
    a = ecom[ecom["Lot"] == "90 WT"]["Purchase Price"]
    print(a)


def hillaries_emails():
    print(ecom[ecom["Credit Card"] == 4926535242672853]["Email"])


def big_nibba_buyers():
    print(ecom[(ecom["CC Provider"] == "American Express") & (ecom["Purchase Price"] > 95)].count())


def pending_expiry():
    a = ecom["CC Exp Date"].str.findall("/25").value_counts()

    print(a)


if __name__ == '__main__':
    ecom = pd.read_csv(
        r"C:\Users\Korisnik\Desktop\Refactored_Py_DS_ML_Bootcamp-master\04-Pandas-Exercises\Ecommerce Purchases")

    # print(ecom.info())
    # average_purchase_price()
    # highest_lowest_purchase_price()
    # language_of_choice()
    # num_lawyers()
    # am_or_pm()
    # get_purchase_price()
    # hillaries_emails()
    # big_nibba_buyers()
    pending_expiry()
