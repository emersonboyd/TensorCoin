import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import RNN
from keras.layers import GRU
from keras.models import load_model
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
        prices = df.groupby(['day','hour'])['Weighted_Price'].mean()
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
    testing =  prices[end_index:]
    return training, testing


def train_data(input, output,MODEL_NAME):
    #Original inputs
    # regressor = Sequential()
    # regressor.add(LSTM(units=4, activation='sigmoid', input_shape=(None, 1)))
    # regressor.add(Dense(units=1))
    # regressor.compile(optimizer='adam', loss='mean_squared_error')
    # regressor.fit(input, output, batch_size=5, epochs=100)
    # regressor.save(MODEL_NAME)
    # return regressor

    #Testing Inputs
    regressor = Sequential()


    #ADD Layers to the RNN
    #LSTM: dropout, units,
    regressor.add(LSTM(units=8, activation='sigmoid', input_shape=(None, 1)))
    #Dense
    regressor.add(Dense(units=1))
    #RNN
    #regressor.add(RNN(units=1))

    #Compile RNN, optomizers can be RMSprop
    regressor.compile(optimizer='adam', loss='mean_squared_error')

    #Fit Model. Change Batch size and epochs
    regressor.fit(input, output, batch_size=5, epochs=100)

    #Save the model to file for future use
    regressor.save(MODEL_NAME)
    return regressor


#Display the results
def draw_data(df_test, predicted_price,unit,xlabel,title):
    df_test_vals = np.ndarray.tolist(df_test.values)
    predicted_price = np.ndarray.tolist(predicted_price)
    predictions = []
    for item in predicted_price:
        predictions.append(item[0])
    difference = np.subtract(df_test_vals, predictions)
    avg_off = sum(difference)/float(len(difference));
    print avg_off
    plt.figure(figsize=(15, 15), dpi=80, facecolor='w', edgecolor='k')
    plt.title(title, fontsize=40)
    plt.plot(df_test.values, color='red', label='Real BTC Price')
    plt.plot(predicted_price, color='blue', label='Predicted BTC Price')
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
    pdb.set_trace()


def run():
    df = read_data()
    create_model = True
    segment_unit = 'hour'
    MODEL_NAME = '365_days_to_1_day_in_hours_2_new_layers.h5'
    PULL_MODEL = '365_days_to_1_day_in_hours.h5'
    df_train, df_test = segment_data(df, 'day', 365, 1, segment_unit)

    sc = MinMaxScaler()
    # Reshape the data to create a model
    training_set = np.reshape(df_train.values, (len(df_train.values), 1))
    training_set = sc.fit_transform(training_set)

    #Get the regression model
    if create_model:
        train_input, train_output = training_set[0:len(training_set) - 1], training_set[1:len(training_set)]
        # Create a regressor and save it to MODEL_NAME
        regressor = train_data(np.reshape(train_input, (len(train_input), 1, 1)), train_output,MODEL_NAME)
        # load the regressor
    else:
        regressor = load_model(PULL_MODEL)

    #Get the data to predict
    test_input = np.reshape(df_test.values, (len(df_test.values), 1))
    test_input = sc.transform(test_input)
    test_input = np.reshape(test_input, (len(test_input), 1, 1))
    predicted_price = sc.inverse_transform(regressor.predict(test_input,batch_size=1))

    draw_data(df_test, predicted_price,'hour','Time','Baseline Model 364 days of hourly training')

if __name__ == "__main__":
    run()
