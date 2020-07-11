# wget https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip
# From book "Deep Learning with Python, FRANÇOIS CHOLLET"

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Tuple

import tensorflow.keras as ks
#from tensorflow.keras.preprocessing import sequence

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


def load_data(fname):
    dtparser = lambda x: datetime.strptime(x, '%d.%m.%Y %H:%M:%S')
    dates = pd.read_csv(fname, parse_dates=['Date Time'], date_parser=dtparser)
    header = dates.columns[1:].tolist()
    float_data = dates.iloc[:, 1:].values
    dates = dates.iloc[:, 0]
    return header, dates, float_data


def generator(data: np.ndarray, lookback: int, delay: int, min_index: int, max_index:int,
              shuffle:bool=False, batch_size:int=128, step:int=6) -> Tuple[np.ndarray, np.ndarray]:
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, _ in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets


def build_model(type, do_load=False):
    act='tanh'
    #act='relu'
    if do_load:
        model = ks.models.load_model(base_name + '_m' + str(type) + '.h5')
        return model, None
    elif type == 1:
        model = ks.models.Sequential([
            ks.layers.Flatten(input_shape=(lookback // step, float_data.shape[-1])),
            ks.layers.Dense(32, activation=act),
            ks.layers.Dense(1)
        ])
    elif type == 2:
        model = ks.models.Sequential([
            ks.layers.GRU(32, dropout=0.2, input_shape=(None, float_data.shape[-1])),  # + recurrent_dropout=0.2 -> not cuDNN comp.
            ks.layers.Dense(1)
        ])
    elif type == 3: # Model combining a 1D convolutional base and a GRU layer
        model = ks.models.Sequential([
            ks.layers.Conv1D(32, 5, activation=act, input_shape=(None, float_data.shape[-1])),
            ks.layers.MaxPool1D(3),
            ks.layers.Conv1D(32, 5, activation=act),
            ks.layers.GRU(32, dropout=0.1, input_shape=(None, float_data.shape[-1])),  # + recurrent_dropout=0.5 -> not cuDNN comp.
            ks.layers.Dense(1)
        ])

    print("Fit model", type)
    model.compile(optimizer=ks.optimizers.RMSprop(), loss='mae')
    history = model.fit(x=train_gen, steps_per_epoch=500, epochs=20,
                        validation_data=val_gen,
                        validation_steps=val_steps,
                        verbose=1)
    model.save(base_name + '_m' + str(type) + '.h5')
    return model, history



############ start

base_name = 'data/jena_climate_2009_2016'
model_type = 2
do_load = False
do_plot = False
do_evaluate = True
step = 3                 # The period, in timesteps, at which you sample data. Set to 6 order to draw one data point every hour. (sampled every 10 minute)
lookback = 5*(step*24)   # How many timesteps back the input data should go: Observations will go back 5 days.
delay = 1*(step*24)      # How many timesteps in the future the target should be: 24 hours in the future.
batch_size = 128         # The number of samples per batch
training_timesteps = 200000

header, dates, float_data = load_data(base_name + '.csv')

if do_plot:
    print(header)
    print(dates)
    print(float_data.shape)

    temp = float_data[:, 1]  # temperature (in degrees celsius)
    plt.figure()
    plt.plot(range(len(temp)), temp)
    # first 10 days (temp is recorded every 10 minutes))
    plt.figure()
    plt.plot(range(1440), temp[:1440])
    # Show it
    plt.show()

# Normalize data
train = float_data[:training_timesteps]
#mean, std = train.mean(axis=0), train.std(axis=0)
#dmin, dmax = train.min(axis=0), train.max(axis=0)
#fac = 8
#for i in range(len(mean)):
#    print(i, "rng:", round(dmin[i], 2), round(dmax[i], 2))
#    print(i, "std:", round(mean[i] - std[i]*(fac/2), 2), round(mean[i] + std[i]*(fac/2), 2))
#float_data -= mean
#float_data /= std * fac
std = train.std(axis=0)
scaler = Scaler()
scaler.fit(train)
scaler.transform(float_data, True)


train_gen = generator(float_data, lookback=lookback, delay=delay, min_index=0, max_index=200000,
                      shuffle=True, step=step, batch_size=batch_size)
val_gen = generator(float_data, lookback=lookback, delay=delay, min_index=200001, max_index=300000,
                      shuffle=True, step=step, batch_size=batch_size)
test_gen = generator(float_data, lookback=lookback, delay=delay, min_index=300001, max_index=None,
                      shuffle=True, step=step, batch_size=batch_size)


# these are the forecasted parts that we will want to look at
val_steps = (300000 - 200001 - lookback) // batch_size   # how many steps to draw from val_gen in order to see
                                                         # the entire validation set
test_steps = (len(float_data) - 300001 - lookback) // batch_size     # " " for test

if do_evaluate: # evaluate naive method
    batch_maes = []
    for s in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    print('naive MAE:', np.mean(batch_maes))
    celsius_mae = 0.29 * std[1] 
    print('celsius_mae:', celsius_mae)

model, history = build_model(model_type, do_load)
model.summary()

if history: # plot
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
