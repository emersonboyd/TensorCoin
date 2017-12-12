import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model

MODEL_NAME = 'bitcoin_model.h5'


def read_data():
    df = pd.read_csv('./bitcoin-historical-data/coinbaseUSD_1-min_data_2014-12-01_to_2017-10-20.csv.csv')
    df['date'] = pd.to_datetime(df['Timestamp'], unit='s').dt.date
    return df


def segment_data(df, percent_train):
    prices_by_date = df.groupby('date')['Weighted_Price'].mean()
    num_train = int(percent_train * len(prices_by_date))
    return prices_by_date[:num_train], prices_by_date[num_train:]


def train_data(input, output):
    regressor = Sequential()
    regressor.add(LSTM(units=4, activation='sigmoid', input_shape=(None, 1)))
    regressor.add(Dense(units=1))
    regressor.compile(optimizer='adam', loss='mean_squared_error')
    regressor.fit(input, output, batch_size=5, epochs=100)
    regressor.save(MODEL_NAME)
    return regressor


def draw_data(df_test, predicted_price):
    plt.figure(figsize=(25, 15), dpi=80, facecolor='w', edgecolor='k')
    plt.title('BTC Price Prediction', fontsize=40)
    plt.plot(df_test.values, color='red', label='Real BTC Price')
    plt.plot(predicted_price, color='blue', label='Predicted BTC Price')
    df_test = df_test.reset_index()
    plt.xticks(df_test.index, df_test['date'], rotation='vertical')
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
    df_train, df_test = segment_data(df, 0.9)

    sc = MinMaxScaler()

    training_set = np.reshape(df_train.values, (len(df_train.values), 1))
    training_set = sc.fit_transform(training_set)
    # train_input, train_output = training_set[0:len(training_set) - 1], training_set[1:len(training_set)]
    # train_data(np.reshape(train_input, (len(train_input), 1, 1)), train_output)

    regressor = load_model(MODEL_NAME)

    test_input = np.reshape(df_test.values, (len(df_test.values), 1))
    test_input = sc.transform(test_input)
    test_input = np.reshape(test_input, (len(test_input), 1, 1))
    predicted_price = sc.inverse_transform(regressor.predict(test_input))

    draw_data(df_test, predicted_price)

if __name__ == "__main__":
    run()
