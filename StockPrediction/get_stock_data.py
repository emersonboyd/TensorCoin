import pandas_datareader.data as web
import pandas as pd
import datetime

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


start = FINAL_START_DATE
end = FINAL_END_DATE
ndaq_df = web.DataReader("NDAQ", 'yahoo', start, end)

ndaq_df = fill_missing_dates(ndaq_df)

ndaq_df_mean_20 = get_average_over_time_period(ndaq_df, 20, 'mean')
ndaq_df_median_20 = get_average_over_time_period(ndaq_df, 20, 'median')

ndaq_df_pct_change = get_percent_change(ndaq_df)

ndaq_df_mean_20.plot()