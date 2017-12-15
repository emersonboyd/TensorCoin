import pandas_datareader.data as web
import pandas as pd
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt
import matplotlib
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn import svm
from sklearn import isotonic
from sklearn import kernel_ridge
from sklearn import cross_decomposition
from sklearn import gaussian_process
from sklearn import metrics
import math


FINAL_START_DATE = datetime.datetime(2017, 1, 1)
FINAL_END_DATE = datetime.datetime.now()


def read_data():
    csv_folder = '../bitcoin-historical-data/'
    csv_name = 'coinbaseUSD_1-min_data_2014-12-01_to_2017-10-20.csv.csv'
    df = pd.read_csv(''.join([csv_folder, csv_name]))
    df['date'] = pd.to_datetime(df['Timestamp'], unit='s').dt.date
    prices_by_date = df
    prices_by_date = df.groupby('date')['Weighted_Price'].mean()
    # prices_by_date = prices_by_date.set_index('date')
    return prices_by_date


def split_df(df, percent_train):
    num_train = int(percent_train * len(df))
    return df[:num_train], df[num_train:]


def drop_df_fraction(df, percent_drop):
    start_df, end_df = split_df(df, percent_drop)
    return end_df


def fill_missing_dates(df):
    start_date = df.iloc[0].name
    end_date = df.iloc[-1].name

    df = df.reindex(pd.date_range(start_date, end_date), method='ffill')

    return df


def get_average_over_time_period(df, num_days, method):
    start_date = df.iloc[0].name
    end_date = df.iloc[-1].name

    start_period = start_date
    end_period = start_period + datetime.timedelta(days=num_days)

    average_df = pd.DataFrame()

    while end_period <= end_date:
        df_date_period = df[start_period : end_period]
        if method == 'mean':
            df_date_period = df_date_period.mean()
        if method == 'median':
            df_date_period = df_date_period.median()
        df_date_period.name = end_period
        average_df = average_df.append(df_date_period)

        start_period = end_period
        end_period = start_period + datetime.timedelta(days=num_days)

    return average_df


def get_percent_change(df):
    return df.pct_change()


def date_to_number(date):
    return time.mktime(date.timetuple())


def number_to_date(num):
    return datetime.datetime.fromtimestamp(num).date()


def get_model(series, label):
    model = linear_model.LinearRegression()
    # model = ensemble.RandomForestRegressor()

    dates_to_train = series.index.values
    prices_to_train = series.as_matrix()

    # fit the x and y to n_samples x 1
    dates_to_train = np.reshape(dates_to_train, (len(dates_to_train), 1))
    prices_to_train = np.reshape(prices_to_train, (len(prices_to_train), 1))

    prices_to_train = prices_to_train.ravel().astype(int)

    model.fit(dates_to_train, prices_to_train) #fitting the data points in the model
    return model


def predict_prices(df, label, dates_to_predict):
    model = get_model(df, label)

    predicted_prices = model.predict(dates_to_predict)
    return model, predicted_prices


def plot_model(series_train, series_test, label):
    series_train.index = series_train.index.map(date_to_number)
    series_test.index = series_test.index.map(date_to_number)

    dates_to_predict = series_test.index.values
    prices_to_test_against = series_test.as_matrix()

    # fit the x and y to n_samples x 1
    dates_to_predict = np.reshape(dates_to_predict, (len(dates_to_predict), 1))
    prices_to_test_against = np.reshape(prices_to_test_against, (len(prices_to_test_against), 1))

    model, predicted_prices = predict_prices(series_train, label, dates_to_predict)

    dates_to_train = series_train.index.values
    prices_to_train = series_train.as_matrix()

    # fit the x and y to n_samples x 1
    dates_to_train = np.reshape(dates_to_train, (len(dates_to_train), 1))
    prices_to_train = np.reshape(prices_to_train, (len(prices_to_train), 1))

    mse = metrics.mean_squared_error(prices_to_test_against, model.predict(dates_to_predict))
    rmse = math.sqrt(mse)
    print 'Root Mean Square Error', rmse

    plt.figure(figsize=(15, 15), dpi=80, facecolor='w', edgecolor='k')
    plt.title('Linear Regression Model', fontsize=40)

    map_func = np.vectorize(number_to_date)
    dates_to_predict_as_date = map_func(dates_to_predict)
    dates_to_train_as_date = map_func(dates_to_train)

    print dates_to_predict_as_date
    print dates_to_train_as_date

    # plt.scatter(dates_to_train, prices_to_train, color='green')  # plotting the initial datapoints
    plt.scatter(dates_to_predict_as_date, prices_to_test_against, color='red', label='Real BTC Price')  # plotting the test datapoints
    plt.plot(dates_to_predict_as_date, model.predict(dates_to_predict), color='blue', linewidth=2, label='Predicted BTC Price')  # plotting the line made by linear regression

    plt.xlabel('Dates', fontsize=40)
    plt.ylabel('BTC Price(USD)', fontsize=40)
    plt.legend(loc=2, prop={'size': 25})

    plt.show()

    return


# start = FINAL_START_DATE
# end = FINAL_END_DATE
# ndaq_df = web.DataReader("NDAQ", 'yahoo', start, end)
#
# ndaq_df = fill_missing_dates(ndaq_df)
#
# ndaq_df_mean_20 = get_average_over_time_period(ndaq_df, 20, 'mean')
# ndaq_df_median_20 = get_average_over_time_period(ndaq_df, 20, 'median')
#
# ndaq_df_pct_change = get_percent_change(ndaq_df)
#
# ndaq_df.index = ndaq_df.index.map(date_to_number)
# ndaq_df_mean_20.index = ndaq_df_mean_20.index.map(date_to_number)
#
# print predict_prices(ndaq_df_mean_20, 'Open', date_to_number(FINAL_END_DATE))
#
# plot_model(ndaq_df_mean_20, 'Open')

series_coin = read_data()

series_coin = drop_df_fraction(series_coin, .8)

series_coin_train, series_coin_test = split_df(series_coin, .7)

plot_model(series_coin_train, series_coin_test, 'Weighted_Price')

