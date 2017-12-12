import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model
import pdb

MODEL_NAME = 'bitcoin_model.h5'

# Get the Bitcoin data from the csv file
def read_data():
    df = pd.read_csv('./bitcoin_historical-data/coinbaseUSD_1-min_data_2014-12-01_to_2017-10-20.csv.csv')
    df['day'] = pd.to_datetime(df['Timestamp'], unit='s').dt.date
    df['hour'] = pd.to_datetime(df['Timestamp'], unit='s').dt.hour
    df['minute'] = pd.to_datetime(df['Timestamp'], unit='s').dt.minute
    return df

# Get the range of training and testing data e.g.
# Get the data from 20 days ago to 2 days ago, for every minute
# segment_data('df,days',20,2,'minute') Can do 'day' 'hour' and 'minute'
def segment_data(df, rangeUnit, startUnitsBack, endUnitsBack, segmentUnits):
    if segmentUnits == 'day':
        prices = df.groupby('day')['Weighted_Price'].mean()
    elif segmentUnits == 'hour':
        prices = df.groupby(['day','hour'])['Weighted_Price'].mean()
    elif segmentUnits == 'minute':
        prices = df.groupby(['day', 'hour', 'minute'])['Weighted_Price'].mean()
    if rangeUnit == segmentUnits:
        startIndex = len(prices) - startUnitsBack
        endIndex = len(prices) - endUnitsBack
    if rangeUnit == 'day':
        if segmentUnits == 'hour':
            startIndex = len(prices) - 24*startUnitsBack
            endIndex = len(prices) - 24*endUnitsBack
        else:
            startIndex = len(prices) - 24 * 60 * startUnitsBack
            endIndex = len(prices) - 24 * 60 * endUnitsBack
    if rangeUnit == 'hour':
        startIndex = len(prices) - 60 * startUnitsBack
        endIndex = len(prices) - 60 * endUnitsBack
    training = prices[startIndex:endIndex]
    testing =  prices[endIndex:]
    return training, testing


def train_data(input, output):
    regressor = Sequential()
    regressor.add(LSTM(units=4, activation='sigmoid', input_shape=(None, 1)))
    regressor.add(Dense(units=1))
    regressor.compile(optimizer='adam', loss='mean_squared_error')
    regressor.fit(input, output, batch_size=5, epochs=100)
    regressor.save(MODEL_NAME)
    return regressor


def draw_data(df_test, predicted_price,unit):
    plt.figure(figsize=(15, 15), dpi=80, facecolor='w', edgecolor='k')
    plt.title('BTC Price Prediction', fontsize=40)
    plt.plot(df_test.values, color='red', label='Real BTC Price')
    plt.plot(predicted_price, color='blue', label='Predicted BTC Price')
    df_test = df_test.reset_index()
    plt.xticks(df_test.index, df_test[unit], rotation='vertical')
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(10)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(18)
    plt.xlabel('Time', fontsize=40)
    plt.ylabel('BTC Price(USD)', fontsize=40)
    plt.legend(loc=2, prop={'size': 25})
    plt.show()


def run():
    df = read_data()
    segmentUnit = 'minute'
    df_train, df_test = segment_data(df, 'day',30,1,segmentUnit)

    sc = MinMaxScaler()

    training_set = np.reshape(df_train.values, (len(df_train.values), 1))
    training_set = sc.fit_transform(training_set)
    train_input, train_output = training_set[0:len(training_set) - 1], training_set[1:len(training_set)]
    train_data(np.reshape(train_input, (len(train_input), 1, 1)), train_output)

    regressor = load_model(MODEL_NAME)

    test_input = np.reshape(df_test.values, (len(df_test.values), 1))
    test_input = sc.transform(test_input)
    test_input = np.reshape(test_input, (len(test_input), 1, 1))
    predicted_price = sc.inverse_transform(regressor.predict(test_input))

    draw_data(df_test, predicted_price,segmentUnit)

if __name__ == "__main__":
    run()
