import pandas_datareader as data
import pandas_datareader.data as web
import pandas as pd
import numpy as np
import datetime


start = datetime.datetime(2006, 1, 1)
end = datetime.datetime(2016, 1, 1)

if __name__ == '__main__':

    BAC = web.DataReader("BAC", 'google', start, end)
