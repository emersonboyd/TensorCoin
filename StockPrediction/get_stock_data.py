import pandas_datareader.data as web
import pandas as pd
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt
import matplotlib
from sklearn import linear_model


FINAL_START_DATE = datetime.datetime(2010, 1, 1)
FINAL_END_DATE = datetime.datetime.now()


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
    return datetime.datetime.fromtimestamp(num)


def get_model(df, label):
    model = linear_model.LinearRegression() #defining the linear regression model

    dates_to_train = df.index.values
    prices_to_train = df.loc[:, label].as_matrix()

    # fit the x and y to n_samples x 1
    dates_to_train = np.reshape(dates_to_train, (len(dates_to_train), 1))
    prices_to_train = np.reshape(prices_to_train, (len(prices_to_train), 1))

    model.fit(dates_to_train, prices_to_train) #fitting the data points in the model

    return model


def predict_prices(df, label, dates_to_predict):
    model = get_model(df, label)

    predicted_price = model.predict(dates_to_predict)
    return predicted_price


start = FINAL_START_DATE
end = FINAL_END_DATE
ndaq_df = web.DataReader("NDAQ", 'yahoo', start, end)

ndaq_df = fill_missing_dates(ndaq_df)

ndaq_df_mean_20 = get_average_over_time_period(ndaq_df, 20, 'mean')
ndaq_df_median_20 = get_average_over_time_period(ndaq_df, 20, 'median')

ndaq_df_pct_change = get_percent_change(ndaq_df)

ndaq_df.index = ndaq_df.index.map(date_to_number)
ndaq_df_mean_20.index = ndaq_df_mean_20.index.map(date_to_number)

print predict_prices(ndaq_df_mean_20, 'Open', date_to_number(FINAL_END_DATE))
#
# ndaq_df.plot(y='Open')
# plt.show()