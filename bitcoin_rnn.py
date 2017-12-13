import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import SimpleRNN
from keras.layers import GRU
from keras.models import load_model
from keras.layers.merge import concatenate
import pdb


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
def segment_data(df, range_unit, start_units_back, end_units_back, segment_units):
    if segment_units == 'day':
        prices = df.groupby('day')['Weighted_Price'].mean()
    elif segment_units == 'hour':
        prices = df.groupby(['day', 'hour'])['Weighted_Price'].mean()
    elif segment_units == 'minute':
        prices = df.groupby(['day', 'hour', 'minute'])['Weighted_Price'].mean()
    if range_unit == segment_units:
        start_index = len(prices) - start_units_back
        end_index = len(prices) - end_units_back
    if range_unit == 'day':
        if segment_units == 'hour':
            start_index = len(prices) - 24 * start_units_back
            end_index = len(prices) - 24 * end_units_back
        else:
            start_index = len(prices) - 24 * 60 * start_units_back
            end_index = len(prices) - 24 * 60 * end_units_back
    if range_unit == 'hour':
        start_index = len(prices) - 60 * start_units_back
        end_index = len(prices) - 60 * end_units_back
    training = prices[start_index:end_index]
    testing = prices[end_index:]
    testVal = prices[end_index]
    prev_week = prices[end_index-(24*7):end_index]
    return training, testing,testVal,prev_week


def train_data(input, output, MODEL_NAME):
    # Original inputs
    # regressor = Sequential()
    # regressor.add(LSTM(units=4, activation='sigmoid', input_shape=(None, 1)))
    # regressor.add(Dense(units=1))
    # regressor.compile(optimizer='adam', loss='mean_squared_error')
    # regressor.fit(input, output, batch_size=5, epochs=100)
    # regressor.save(MODEL_NAME)
    # return regressor

    # Testing Inputs
    regressor = Sequential()

    # ADD Layers to the RNN
    # LSTM: dropout, units,
    # regressor.add(LSTM(units=8, activation='sigmoid', input_shape=(None, 1)))
    # regressor.add(LSTM(units=16, activation='sigmoid', input_shape=(None, 1)))
    # regressor.add(LSTM(units=4, activation='sigmoid', input_shape=(None, 1), dropout=.2))
    # regressor.add(LSTM(units=4, activation='sigmoid', input_shape=(None, 1),dropout=.05))
    # regressor.add(SimpleRNN(units=4, activation='sigmoid', input_shape=(None, 1)))
    regressor.add(LSTM(units=4, activation='sigmoid', input_shape=(None, 1)))
    regressor.add(Dense(1))

    # Compile RNN, optomizers can be RMSprop,SGD,Adam
    # regressor.compile(optimizer='adam', loss='mean_squared_error')
    # regressor.compile(optimizer='sgd', loss='mean_squared_error')
    regressor.compile(optimizer='sgd', loss='mean_squared_error')

    # Fit Model. Change Batch size and epochs
    regressor.fit(input, output, batch_size=20, epochs=100)

    # Save the model to file for future use
    regressor.save(MODEL_NAME)
    return regressor


# Display the results
def draw_data(df_test, predicted_price, unit, xlabel, title):
    pdb.set_trace()
    # df_test_vals = np.ndarray.tolist(df_test.values)
    # predictions = []
    # pdb.set_trace()
    # for item in predicted_price:
    #     predictions.append(item[0])
    # difference = np.subtract(df_test_vals, predictions)
    # avg_off = sum(difference) / float(len(difference))
    # print avg_off
    plt.figure(figsize=(15, 15), dpi=80, facecolor='w', edgecolor='k')
    plt.title(title, fontsize=40)
    plt.plot(df_test.values, color='red', label='Real BTC Price')
    plt.plot(predicted_price, color='blue', label='Predicted BTC Price')
    # plt.plot(difference,color='green',label='Difference')
    df_test = df_test.reset_index()
    plt.xticks(df_test.index, df_test[unit], rotation='vertical')
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks(10):
        tick.label1.set_fontsize(10)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(18)
    plt.xlabel(xlabel, fontsize=40)
    plt.ylabel('BTC Price(USD)', fontsize=40)
    plt.legend(loc=2, prop={'size': 25})
    plt.show()


def run():
    MODELS = ['365_days_to_1_day_in_hours.h5',  # Original 4 LSTMS
              '365_days_to_1_day_in_hours_2_new_layers_16_lstms.h5',  # 16 LSTMS
              '365_days_to_1_day_in_hours_2_new_layers_dropout_05.h5',  # 5% Dropout
              '365_days_to_1_day_in_hours_2_new_layers_dropout_2.h5',  # %20% Dropout
              '365_days_to_1_day_in_hours_8_lstms.h5',  # 8 LSTMS
              '365_days_to_1_day_in_hours_8_tanh.h5',  # 8 TANH activation LSTM
              '365_days_to_1_day_in_hours_rnn.h5',  # SimpleRNN
              '365_days_to_1_day_in_hours_tanh.h5',  # 4 TANH activations LSTMs
              '365_days_to_1_day_in_hours_RMSprop.h5',  # RMSprop optimation
              '365_days_to_1_day_in_hours_SGD.h5',
              '365_days_to_1_day_in_hours_batch_10.h5',
              '365_days_to_1_day_in_hours_batch_20_200_epochs.h5']  # SGD optimation
    df = read_data()
    create_model = True
    run_through_all = False
    segment_unit = 'hour'
    MODEL_NAME = 'models/50_days_sigmoid_sgd.h5'
    PULL_MODEL = 'standard.h5'
    df_train, df_test,test_val,prev = segment_data(df, 'day', 50, 1, segment_unit)

    sc = MinMaxScaler()
    # Reshape the data to create a model
    training_set = np.reshape(df_train.values, (len(df_train.values), 1))
    training_set = sc.fit_transform(training_set)

    # Get the regression model
    if create_model:
        train_input, train_output = training_set[0:len(training_set) - 1], training_set[1:len(training_set)]
        # Create a regressor and save it to MODEL_NAME
        regressor = train_data(np.reshape(train_input, (len(train_input), 1, 1)), train_output, MODEL_NAME)
        # load the regressor
    else:
        regressor = load_model(PULL_MODEL)

    if not run_through_all:
        # Get the data to predict


        test_input = np.reshape(df_test.values, (len(df_test.values), 1))
        test_input = sc.transform(test_input)
        test_input = np.reshape(test_input, (len(test_input), 1, 1))
        predicted_prices = sc.inverse_transform(regressor.predict(test_input, batch_size=1))

        a = np.full(1,test_val)
        test_input = np.reshape(a, (len(a), 1))
        test_input = sc.transform(test_input)
        test_input = np.reshape(test_input, (len(test_input), 1, 1))
        prices = list()
        actual = list()
        actual.append(test_val)
        prices.append(test_val)
        prev = test_input
        for i in range(0,len(df_test.values)-1):
            predicted_price = sc.inverse_transform(regressor.predict(prev, batch_size=1))
            predicted_price = predicted_price.item(i)
            prices.append(predicted_price)
            actual.append(df_test.values.item(i+1))
            prev = np.array(actual)
            prev = np.reshape(prev, (len(prev), 1))
            prev = sc.transform(prev)
            prev = np.reshape(prev, (len(prev), 1, 1))
        print prices

        draw_data(df_test, np.array(prices[1:]), 'hour', 'Time', 'Baseline Model 364 days of hourly training')
    else:
        for file in MODELS:
            regressor = load_model(file)
            test_input = np.reshape(df_test.values, (len(df_test.values), 1))
            test_input = sc.transform(test_input)
            test_input = np.reshape(test_input, (len(test_input), 1, 1))
            predicted_price = sc.inverse_transform(regressor.predict(test_input, batch_size=1))

            draw_data(df_test, predicted_price, 'hour', 'Time', file)


if __name__ == "__main__":
    run()
