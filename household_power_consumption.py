# Machine learning using data: https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption
# https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/
# https://machinelearningmastery.com/how-to-develop-convolutional-neural-networks-for-multi-step-time-series-forecasting/

import math
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as ks
import matplotlib.pyplot as pyplot
import sklearn.metrics as skl_met

class Scaler:
    def __init__(self):
        self.mean = 0.0
        self.scale = 1.0

    def fit(self, data, stds=6):
        self.mean = np.mean(data, axis=0)
        self.scale = np.std(data, axis=0) * stds

    def transform(self, data, inplace=False):
        if inplace:
            data -= self.mean
            data /= self.scale
            return data
        else:
            return (data - self.mean) / self.scale

    def inverse_transform(self, data):
        return (data * self.scale) + self.mean


base_file = 'data/household_power_consumption'

def load_data(base_file, initial=True, full=False):
    if initial:
        # Load, prepare and save data
        dataset = pd.read_csv(base_file + '.txt', sep=';', header=0, low_memory=False, infer_datetime_format=True,
                              na_values='?', dtype=np.float32, parse_dates={'datetime': [0,1]}, index_col=['datetime']) 
        print('Done read_csv')
        values = dataset.values
        one_day = 60 * 24
        nans = np.argwhere(np.isnan(values))
        print('NaNs', len(nans))
        for a in nans:
            values[a[0], a[1]] = values[a[0] - one_day, a[1]]
        #nans = np.argwhere(np.isnan(values))
        #print('NaNs', len(nans))
        #for row in range(values.shape[0]): # slow
        #    for col in range(values.shape[1]):
        #        if np.isnan(values[row, col]):
        #            values[row, col] = values[row - one_day, col]
        print('Add a derived metering:') # subtract the sum of three defined sub-metering variables from the total active energy
        dataset['Sub_metering_4'] = (values[:,0] * 1000 / 60) - (values[:,4] + values[:,5] + values[:,6])    
        dataset.to_pickle(base_file + '.pkl')

        # Resample data to daily
        daily_groups = dataset.resample('D')
        daily_data = daily_groups.sum()
        # summarize
        print(daily_data.shape)
        print(daily_data.head())
        # save
        daily_data.to_pickle(base_file + '_days.pkl')
        data = dataset if full else daily_data
    else:
        # Load the prepared dataset
        #dataset = pd.read_pickle(base_file + '.pkl')
        data = pd.read_pickle(base_file + '.pkl') if full else pd.read_pickle(base_file + '_days.pkl')
    return data


def plot_variables(dataset):
    pyplot.figure()
    for i in range(len(dataset.columns)):
        # create subplot
        pyplot.subplot(len(dataset.columns), 1, i+1)
        # get variable name
        name = dataset.columns[i]
        # plot data
        pyplot.plot(dataset[name])
        # set title
        pyplot.title(name, y=0)
        # turn off ticks to remove clutter
        pyplot.yticks([])
        pyplot.xticks([])
    pyplot.show()    


def plot_active_power(dataset):
    years = ['2007', '2008', '2009', '2010']
    pyplot.figure()
    for i in range(len(years)):
        # prepare subplot
        ax = pyplot.subplot(len(years), 1, i+1)
        # determine the year to plot
        year = years[i]
        # get all observations for the year
        result = dataset[str(year)]
        # plot the active power for the year
        pyplot.plot(result['Global_active_power'])
        # add a title to the subplot
        pyplot.title(str(year), y=0, loc='left')
        # turn off ticks to remove clutter
        pyplot.yticks([])
        pyplot.xticks([])        
    pyplot.show()    


# plot training history
def plot_history(history):
    # plot loss
    pyplot.subplot(2, 1, 1)
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.title('loss', y=0, loc='center')
    pyplot.legend()
    # plot rmse
    pyplot.subplot(2, 1, 2)
    pyplot.plot(history.history['rmse'], label='train')
    pyplot.plot(history.history['val_rmse'], label='test')
    pyplot.title('rmse', y=0, loc='center')
    pyplot.legend()
    pyplot.show()


# split a univariate dataset into train/test sets
def split_dataset(data):
    # split into standard weeks
    train, test = data[1:-328], data[-328:-6]
    # restructure into windows of weekly data
    train = np.array(np.split(train, len(train)/7))
    test = np.array(np.split(test, len(test)/7))
    return train, test


# evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(actual, predicted):
    scores = list()
    # calculate an RMSE score for each day
    for i in range(actual.shape[1]):
        # calculate mse
        mse = skl_met.mean_squared_error(actual[:, i], predicted[:, i])
        # calculate rmse
        rmse = math.sqrt(mse)
        # store
        scores.append(rmse)
    # calculate overall RMSE
    s = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col])**2
    score = math.sqrt(s / (actual.shape[0] * actual.shape[1]))
    return score, scores

# summarize scores
def summarize_scores(name, score, scores):
    s_scores = ', '.join(['%.3f' % s for s in scores])
    print('%s: [%.3f] %s' % (name, score, s_scores))


# convert history into inputs and outputs
# NB: For all with univariate Input
def to_supervised_univar_inp(train, n_input, n_out=7):
    # flatten data
    data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
    X, y = list(), list()
    in_start = 0
    # step over the entire history one time step at a time
    for _ in range(len(data)):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_out
        # ensure we have enough data for this instance
        if out_end <= len(data):
            x_input = data[in_start:in_end, 0]
            x_input = x_input.reshape((len(x_input), 1))
            X.append(x_input)
            y.append(data[in_end:out_end, 0])
        # move along one time step
        in_start += 1
    return np.array(X), np.array(y)

# NB: For all with multivariate Input
def to_supervised_multivar_inp(train, n_input, n_out=7):
    # flatten data
    data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
    X, y = list(), list()
    in_start = 0
    # step over the entire history one time step at a time
    for _ in range(len(data)):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_out
        # ensure we have enough data for this instance
        if out_end <= len(data):
            X.append(data[in_start:in_end, :]) # All parameters!
            y.append(data[in_end:out_end, 0])
        # move along one time step
        in_start += 1
    return array(X), array(y)


# LSTM Model With Univariate Input and Vector Output
# -- Reads in a sequence of days of total daily power consumption and predicts a vector output
#    of the next standard week of daily power consumption
def build_model_rnn1(rebuild, train, n_input):
    if rebuild:
        # prepare data
        train_x, train_y = to_supervised_univar_inp(train, n_input)
        # define parameters
        verbose, epochs, batch_size = 1, 70, 16
        n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
        # define model
        model = ks.models.Sequential()
        model.add(ks.layers.LSTM(200, activation='tanh', input_shape=(n_timesteps, n_features)))
        model.add(ks.layers.Dense(100, activation='relu'))
        model.add(ks.layers.Dense(n_outputs))
        model.compile(loss='mse', optimizer='adam')
        # fit network
        model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
        model.save(base_file + '_rnn1.h5')
    else:
        model = ks.models.load_model(base_file + '_rnn1.h5')
    return model


# Encoder-Decoder LSTM Model With Univariate Input
# -- The model will be comprised of two sub models, the encoder to read and encode the input sequence,
#    and the decoder that will read the encoded input sequence and make a one-step prediction for each
#    element in the output sequence.
def build_model_rnn2(rebuild, train, n_input):
    if rebuild:
        # prepare data
        train_x, train_y = to_supervised_univar_inp(train, n_input)
        # define parameters
        verbose, epochs, batch_size = 0, 20, 16
        n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
        # reshape output into [samples, timesteps, features]
        train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
        # define model
        model = ks.models.Sequential()
        model.add(ks.layers.LSTM(200, activation='tanh', input_shape=(n_timesteps, n_features)))
        model.add(ks.layers.RepeatVector(n_outputs))
        model.add(ks.layers.LSTM(200, activation='tanh', return_sequences=True))
        model.add(ks.layers.TimeDistributed(ks.layers.Dense(100, activation='relu')))
        model.add(ks.layers.TimeDistributed(ks.layers.Dense(1)))
        model.compile(loss='mse', optimizer='adam')
        # fit network
        model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
        model.save(base_file + '_rnn2.h5')
    else:
        model = ks.models.load_model(base_file + '_rnn2.h5')
    return model

# Encoder-Decoder LSTM Model With Multivariate Input
# -- Use each of the eight time series variables to predict the
#    next standard week of daily total power consumption.
def build_model_rnn3(rebuild, train, n_input):  # identical to above, except for parameters...
    if rebuild:
        # prepare data
        train_x, train_y = to_supervised_multivar_inp(train, n_input)
        # define parameters
        verbose, epochs, batch_size = 0, 50, 16
        n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
        # reshape output into [samples, timesteps, features]
        train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
        # define model
        model = ks.models.Sequential()
        model.add(ks.layers.LSTM(200, activation='tanh', input_shape=(n_timesteps, n_features)))
        model.add(ks.layers.RepeatVector(n_outputs))
        model.add(ks.layers.LSTM(200, activation='tanh', return_sequences=True))
        model.add(ks.layers.TimeDistributed(ks.layers.Dense(100, activation='relu')))
        model.add(ks.layers.TimeDistributed(ks.layers.Dense(1)))
        model.compile(loss='mse', optimizer='adam')
        # fit network
        model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
        model.save(base_file + '_rnn3.h5')
    else:
        model = ks.models.load_model(base_file + '_rnn3.h5')
    return model


# CNN-LSTM Encoder-Decoder Model With Univariate Input
# -- The CNN does not directly support sequence input; instead, a 1D CNN is capable of reading across sequence input
#    and automatically learning the salient features. These can then be interpreted by an LSTM decoder as per normal.
#    We refer to hybrid models that use a CNN and LSTM as CNN-LSTM models, and in this case we are using them together
#    in an encoder-decoder architecture.
def build_model_rnn4(rebuild, train, n_input):
    if rebuild:
        # prepare data
        train_x, train_y = to_supervised_univar_inp(train, n_input)
        # define parameters
        verbose, epochs, batch_size = 1, 20, 16
        n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
        # reshape output into [samples, timesteps, features]
        train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
        # define model
        model = ks.models.Sequential()
        model.add(ks.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
        model.add(ks.layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
        model.add(ks.layers.MaxPool1D(pool_size=2))
        model.add(ks.layers.Flatten())
        model.add(ks.layers.RepeatVector(n_outputs))
        model.add(ks.layers.LSTM(200, activation='tanh', return_sequences=True))
        model.add(ks.layers.TimeDistributed(ks.layers.Dense(100, activation='relu')))
        model.add(ks.layers.TimeDistributed(ks.layers.Dense(1)))
        model.compile(loss='mse', optimizer='adam')
        # fit network
        model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
        model.save(base_file + '_rnn4.h5')
    else:
        model = ks.models.load_model(base_file + '_rnn4.h5')
    return model


# ConvLSTM Encoder-Decoder Model With Univariate Input
# -- Extension of the CNN-LSTM approach is to perform the convolutions of the CNN as part of the LSTM for each time step.
# -- Unlike an LSTM that reads the data in directly in order to calculate internal state and state transitions, and
#    unlike the CNN-LSTM that is interpreting the output from CNN models, the ConvLSTM is using convolutions directly 
#    as part of reading input into the LSTM units themselves.
def build_model_rnn5(rebuild, train, n_steps, n_length, n_input):
    if rebuild:
        # prepare data
        train_x, train_y = to_supervised_univar_inp(train, n_input)
        # define parameters
        verbose, epochs, batch_size = 0, 20, 16
        n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
        # reshape into subsequences [samples, time steps, rows, cols, channels]
        train_x = train_x.reshape((train_x.shape[0], n_steps, 1, n_length, n_features))
        # reshape output into [samples, timesteps, features]
        train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
        # define model
        model = ks.models.Sequential()
        model.add(ks.layers.ConvLSTM2D(filters=64, kernel_size=(1,3), activation='relu', input_shape=(n_steps, 1, n_length, n_features)))
        model.add(ks.layers.Flatten())
        model.add(ks.layers.RepeatVector(n_outputs))
        model.add(ks.layers.LSTM(200, activation='tanh', return_sequences=True))
        model.add(ks.layers.TimeDistributed(ks.layers.Dense(100, activation='relu')))
        model.add(ks.layers.TimeDistributed(ks.layers.Dense(1)))
        model.compile(loss='mse', optimizer='adam')
        # fit network
        model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
        model.save(base_file + '_rnn5.h5')
    else:
        model = ks.models.load_model(base_file + '_rnn5.h5')
    return model

# Multi-step Time Series Forecasting With a Univariate CNN
# -- Given some number of prior days of total daily power consumption,
#    predict the next standard week of daily power consumption.
def build_model_cnn1(rebuild, train, n_input):
    if rebuild:
        # prepare data
        train_x, train_y = to_supervised_univar_inp(train, n_input)
        # define parameters
        verbose, epochs, batch_size = 1, 20, 4
        n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
        # define model
        model = ks.models.Sequential()
        model.add(ks.layers.Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
        model.add(ks.layers.MaxPool1D(pool_size=2))
        model.add(ks.layers.Flatten())
        model.add(ks.layers.Dense(10, activation='relu'))
        model.add(ks.layers.Dense(n_outputs))
        model.compile(loss='mse', optimizer='adam')
        # fit network
        model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
        model.save(base_file + '_cnn1.h5')
    else:
        model = ks.models.load_model(base_file + '_cnn1.h5')
    return model


# Multi-step Time Series Forecasting With a Multichannel CNN:
# -- Use each of the eight time series variables to predict the next standard week of daily total power consumption,
#    providing each one-dimensional time series to the model as a separate channel of input.
def build_model_cnn2(rebuild, train, n_input):
    if rebuild:
        # prepare data
        train_x, train_y = to_supervised_multivar_inp(train, n_input)
        # define parameters
        verbose, epochs, batch_size = 0, 70, 16
        n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
        # reshape output into [samples, timesteps, features]
        train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
        # define model
        model = ks.models.Sequential()
        model.add(ks.layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
        model.add(ks.layers.Conv1D(filters=32, kernel_size=3, activation='relu'))
        model.add(ks.layers.MaxPool1D(pool_size=2))
        model.add(ks.layers.Conv1D(filters=16, kernel_size=3, activation='relu'))
        model.add(ks.layers.MaxPool1D(pool_size=2))
        model.add(ks.layers.Flatten())
        model.add(ks.layers.Dense(100, activation='relu'))
        model.add(ks.layers.Dense(n_output))
        model.compile(loss='mse', optimizer='adam')
        # fit network
        model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
        model.save(base_file + '_cnn2.h5')
    else:
        model = ks.models.load_model(base_file + '_cnn2.h5')
    return model

# Multi-step Time Series Forecasting With a Multihead CNN
# -- Extend the CNN model to have a separate sub-CNN model or head for each input variable.
# -- This requires a modification to the preparation of the model, and in turn, modification
#    to the preparation of the training and test datasets.
Starting with the model, we must define a separate CNN model for each of the eight input variables.
def build_model_cnn3(train, n_input):
    # prepare data
    train_x, train_y = to_supervised_multivar_inp(train, n_input)
    # define parameters
    verbose, epochs, batch_size = 0, 25, 16
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    # create a channel for each variable
    in_layers, out_layers = list(), list()
    for i in range(n_features):
        inputs = ks.layers.Input(shape=(n_timesteps,1))
        conv1 = ks.layers.Conv1D(filters=32, kernel_size=3, activation='relu')(inputs)
        conv2 = ks.layers.Conv1D(filters=32, kernel_size=3, activation='relu')(conv1)
        pool1 = ks.layers.MaxPool1D(pool_size=2)(conv2)
        flat = ks.layers.Flatten()(pool1)
        # store layers
        in_layers.append(inputs)
        out_layers.append(flat)
    # merge heads
    merged = ks.layers.merge.concatenate(out_layers)
    # interpretation
    dense1 = ks.layers.Dense(200, activation='relu')(merged)
    dense2 = ks.layers.Dense(100, activation='relu')(dense1)
    outputs = ks.layers.Dense(n_outputs)(dense2)
    model = ks.models.Model(inputs=in_layers, outputs=outputs)
    # compile model
    model.compile(loss='mse', optimizer='adam')
    # plot the model
    plot_model(model, show_shapes=True, to_file='multiheaded_cnn.png')
    # fit network
    input_data = [train_x[:,:,i].reshape((train_x.shape[0],n_timesteps,1)) for i in range(n_features)]
    model.fit(input_data, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model


# ConvLSTM Encoder-Decoder Model With Univariate Input
def build_model_crnn(rebuild, train, n_steps, n_length, n_input):
    if rebuild:
        # prepare data
        train_x, train_y = to_supervised_univar_inp(train, n_input)
        # define parameters
        verbose, epochs, batch_size = 1, 20, 16
        n_features, n_outputs = train_x.shape[2], train_y.shape[1]
        # reshape into subsequences [samples, timesteps, rows, cols, channels]
        train_x = train_x.reshape((train_x.shape[0], n_steps, 1, n_length, n_features))
        # reshape output into [samples, timesteps, features]
        train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
        # define model
        model = ks.models.Sequential()
        model.add(ks.layers.ConvLSTM2D(filters=64, kernel_size=(1,3), activation='tanh',
                                       input_shape=(n_steps, 1, n_length, n_features)))
        model.add(ks.layers.Flatten())
        model.add(ks.layers.RepeatVector(n_outputs))
        model.add(ks.layers.LSTM(200, activation='tanh', return_sequences=True))
        model.add(ks.layers.TimeDistributed(ks.layers.Dense(100, activation='relu')))
        model.add(ks.layers.TimeDistributed(ks.layers.Dense(1)))
        model.compile(loss='mse', optimizer='adam')
        # fit network
        model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
        model.save(base_file + '_crnn.h5')
    else:
        model = ks.models.load_model(base_file + '_crnn.h5')
    return model


# make a forecast
def forecast_1(model, history, n_input):
    # flatten data
    data = np.array(history)
    data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
    # retrieve last observations for input data
    input_x = data[-n_input:, 0]
    # reshape into [1, n_input, 1]
    input_x = input_x.reshape((1, len(input_x), 1))
    # forecast the next week
    yhat = model.predict(input_x, verbose=0)
    # we only want the vector forecast
    yhat = yhat[0]
    return yhat

# Forecast for "Multi-step Time Series Forecasting With a Multichannel CNN"
def forecast_cnn2(model, history, n_input):
    # flatten data
    data = array(history)
    data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
    # retrieve last observations for input data
    input_x = data[-n_input:, :]
    # reshape into [1, n_input, n]
    input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
    # forecast the next week
    yhat = model.predict(input_x, verbose=0)
    # we only want the vector forecast
    yhat = yhat[0]
    return yhat

# Forcast for "Multi-step Time Series Forecasting With a Multihead CNN"
def forecast_cnn3(model, history, n_input):
    # flatten data
    data = array(history)
    data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
    # retrieve last observations for input data
    input_x = data[-n_input:, :]
    # reshape into n input arrays
    input_x = [input_x[:,i].reshape((1,input_x.shape[0],1)) for i in range(input_x.shape[1])]
    # forecast the next week
    yhat = model.predict(input_x, verbose=0)
    # we only want the vector forecast
    yhat = yhat[0]
    return yhat

# make a forecast
def forecast_3(model, history, n_steps, n_length, n_input):
    # flatten data
    data = np.array(history)
    data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
    # retrieve last observations for input data
    input_x = data[-n_input:, 0]
    # reshape into [samples, timesteps, rows, cols, channels]
    input_x = input_x.reshape((1, n_steps, 1, n_length, 1))
    # forecast the next week
    yhat = model.predict(input_x, verbose=0)
    # we only want the vector forecast
    yhat = yhat[0]
    return yhat


# evaluate a single model
def evaluate_model_1(model, train, test, n_input):
    # history is a list of weekly data
    history = [x for x in train]
    # walk-forward validation over each week
    predictions = list()
    for i in range(len(test)):
        # predict the week
        yhat_sequence = forecast_1(model, history, n_input)
        # store the predictions
        predictions.append(yhat_sequence)
        # get real observation and add to history for predicting the next week
        history.append(test[i, :])
    # evaluate predictions days for each week
    predictions = np.array(predictions)
    score, scores = evaluate_forecasts(test[:, :, 0], predictions)
    return score, scores


# evaluate a single model
def evaluate_model_3(model, train, test, n_steps, n_length, n_input):
    # history is a list of weekly data
    history = [x for x in train]
    # walk-forward validation over each week
    predictions = list()
    for i in range(len(test)):
        # predict the week
        yhat_sequence = forecast_3(model, history, n_steps, n_length, n_input)
        # store the predictions
        predictions.append(yhat_sequence)
        # get real observation and add to history for predicting the next week
        history.append(test[i, :])
    # evaluate predictions days for each week
    predictions = np.array(predictions)
    score, scores = evaluate_forecasts(test[:, :, 0], predictions)
    return score, scores


##### MAIN #####

if __name__ == '__main__':
    np.set_printoptions(suppress=True) # no sci np array printing
    dataset = load_data(base_file, False)

    plot_variables(dataset)

    scaler = Scaler()
    scaler.fit(dataset.values)
    print(scaler)

    normalized = scaler.transform(dataset.values)
    train, test = split_dataset(normalized)
    print(train.shape, test.shape)
    print(scaler.inverse_transform(train[1,1,:]))
   
    # evaluate model and get scores
    model_type = 'cnn1'
    rebuild_model = True
    if model_type == 'cnn1:
        # define the total days to use as input
        n_input = 7
        model = build_model_cnn1(rebuild_model, train, n_input)
        score, scores = evaluate_model_1(model, train, test, n_input)
    elif model_type == 2:
        # define the total days to use as input
        n_input = 7
        model = build_model_2(rebuild_model, train, n_input)
        score, scores = evaluate_model_1(model, train, test, n_input)
    elif model_type == 3:
        n_steps, n_length = 2, 7
        # define the total days to use as input
        n_input = n_length * n_steps
        model = build_model_3(rebuild_model, train, n_steps, n_length, n_input)
        score, scores = evaluate_model_3(model, train, test, n_steps, n_length, n_input)

    # summarize scores
    summarize_scores('lstm', score, scores)
    # plot scores
    days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
    pyplot.plot(days, scores, marker='o', label='lstm')
    pyplot.show()
