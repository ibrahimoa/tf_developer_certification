import os
import tensorflow as tf
import pandas as pd

DATASETS_PATH = "datasets/time_series"
BITCOIN_DATA_PATH = os.path.join(DATASETS_PATH, "BTC_USD_2013-10-01_2021-08-02-CoinDesk.csv")

df = pd.read_csv(BITCOIN_DATA_PATH, parse_dates=["Date"], index_col=["Date"])  # tell pandas column 1 is a datetime.
print(df.head())
print(df.info())

bitcoin_prices = pd.DataFrame(df["Closing Price (USD)"]).rename(
    columns={"Closing Price (USD)": "Price"})
print(bitcoin_prices.head())

import matplotlib.pyplot as plt

bitcoin_prices.plot(figsize=(10, 7))
plt.ylabel("BTC Price in $")
plt.title("Price of Bitcoin from 1 Oct 2013 to 2 Aug 2021", fontsize=16)
plt.legend(fontsize=14)

# Importing time series data with Python's CSV module
import csv
from datetime import datetime

timesteps = []
btc_price = []
with open(BITCOIN_DATA_PATH, "r") as f:
    csv_reader = csv.reader(f, delimiter=",")
    next(csv_reader)  # skip first column
    for line in csv_reader:
        timesteps.append(datetime.strptime(line[1], "%Y-%m-%d"))
        btc_price.append(float(line[2]))
print(timesteps[:10], btc_price[:10])

# plot from csv
import numpy as np

plt.figure(figsize=(10, 7))
plt.plot(timesteps, btc_price)
plt.title("Price of Bitcoin from 1 Oct 2013 to 2 Aug 2021", fontsize=16)
plt.xlabel("Date")
plt.ylabel("BTC Price in $")
plt.legend(fontsize=14)

# creating train and test sets
# Get bitcoin date array
timesteps = bitcoin_prices.index.to_numpy()
prices = bitcoin_prices["Price"].to_numpy()
split_size: int = int(0.8 * len(prices))  # 80% train, 20% test
x_train, y_train = timesteps[:split_size], prices[:split_size]
x_test, y_test = timesteps[split_size:], prices[split_size:]

# plot correctly made splits
plt.figure(figsize=(10, 7))
plt.scatter(x_train, y_train, s=5, label="Train data")
plt.scatter(x_test, y_test, s=5, label="Test data")
plt.xlabel("Date")
plt.ylabel("BTC Price in $")
plt.legend(fontsize=14)


# Create a function to plot time series data
def plot_time_series(timesteps, values, format='.', start=0, end=None, label=None):
    """
    Plots a timesteps (a series of points in time) against values (a series of values across timesteps).

    :param timesteps : array of timesteps
    :param values : array of values across time
    :param format : style of plot, default "."
    :param start : where to start the plot (setting a value will index from start of timesteps & values)
    :param end : where to end the plot (setting a value will index from end of timesteps & values)
    :param label : label to show on plot of values
    """
    # Plot the series
    plt.plot(timesteps[start:end], values[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("BTC Price")
    if label:
        plt.legend(fontsize=14)  # make label bigger
    plt.grid(True)


# naive forecast
naive_forecast = y_test[:-1]  # naive forecast equals every value excluding the
# last one.

plt.figure(figsize=(10, 7))
offset = 300  # offset the values by 300 timesteps
plot_time_series(timesteps=x_test, values=y_test, start=offset, label="Test data")
plot_time_series(timesteps=x_test[1:], values=naive_forecast, format="-", start=offset, label="Naive forecast");


def mean_absolute_scaled_error(y_true, y_pred):
    """
    Implement MASE (assuming no seasonality of data).
    MASE equals one for the naive forecast (or very close to one).
    A forecast which performs better than the naïve should get < 1 MASE.
    """
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))

    # Find MAE of naive forecast (no seasonality)
    mae_naive_no_season = tf.reduce_mean(
        tf.abs(y_true[1:] - y_true[:-1]))  # our seasonality is 1 day (hence the shifting of 1 day)

    return mae / mae_naive_no_season


def evaluate_preds(y_true, y_pred):
    # Make sure float32 (for metric calculations)
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    # Calculate various metrics
    mae = tf.keras.metrics.mean_absolute_error(y_true, y_pred)
    mse = tf.keras.metrics.mean_squared_error(y_true, y_pred)  # puts and emphasis on outliers (all errors get squared)
    rmse = tf.sqrt(mse)
    mape = tf.keras.metrics.mean_absolute_percentage_error(y_true, y_pred)
    mase = mean_absolute_scaled_error(y_true, y_pred)

    if mae.ndim > 0:  # if mae isn't already a scalar, reduce it to one by aggregating tensors to mean
        mae = tf.reduce_mean(mae)
        mse = tf.reduce_mean(mse)
        rmse = tf.reduce_mean(rmse)
        mape = tf.reduce_mean(mape)
        mase = tf.reduce_mean(mase)

    return {"mae": mae.numpy(),
            "mse": mse.numpy(),
            "rmse": rmse.numpy(),
            "mape": mape.numpy(),
            "mase": mase.numpy()}


naive_results = evaluate_preds(y_true=y_test[1:], y_pred=naive_forecast)
print(naive_results)

# Windowing is a method to turn a time series dataset into supervised learning problem.
# For example for a univariate time series, windowing for one week (window=7)
# to predict the next single value (horizon=1) might look like:

# Window for one week (univariate time series)

# [0, 1, 2, 3, 4, 5, 6] -> [7]
# [1, 2, 3, 4, 5, 6, 7] -> [8]
# [2, 3, 4, 5, 6, 7, 8] -> [9]

HORIZON: int = 1  # predict 1 tep at a time
WINDOW_SIZE: int = 7  # use a week worth of timesteps to predict the horizon.


def get_labelled_windows(x, horizon=1):
    """
    Creates labels for windowed dataset.

    E.g. if horizon=1 (default)
    Input: [1, 2, 3, 4, 5, 6] -> Output: ([1, 2, 3, 4, 5], [6])
    """
    return x[:, :-horizon], x[:, -horizon:]


# Create function to view NumPy arrays as windows
def make_windows(x, window_size=7, horizon=1):
    """
    Turns a 1D array into a 2D array of sequential windows of window_size.
    """
    # 1. Create a window of specific window_size (add the horizon on the end for later labelling)
    window_step = np.expand_dims(np.arange(window_size + horizon), axis=0)
    # 2. Create a 2D array of multiple window steps (minus 1 to account for 0 indexing)
    window_indexes = window_step + np.expand_dims(np.arange(len(x) - (window_size + horizon - 1)),
                                                  axis=0).T  # create 2D array of windows of size window_size
    # 3. Index on the target array (time series) with 2D array of multiple window steps
    windowed_array = x[window_indexes]
    # 4. Get the labelled windows
    windows, labels = get_labelled_windows(windowed_array, horizon=horizon)

    return windows, labels


full_windows, full_labels = make_windows(x=prices, window_size=WINDOW_SIZE, horizon=HORIZON)
print(f"Number of windows: {len(full_windows)}\nNumber of labels: {len(full_labels)}")

# view the last 3 windows/labels
for i in range(3):
    print(f"Window: {full_windows[i - 3]} -> Label: {full_labels[i - 3]}")


# Note: You can find a function which achieves similar results to the ones we implemented above at:
# tf.keras.preprocessing.timeseries_dataset_from_array().
# Just like ours, it takes in an array and returns a windowed dataset. It has the benefit of
# returning data in the form of a tf.data.Dataset instance (we'll see how to do this with our
# own data later).

# turning windows into training and test sets

def make_train_test_splits(windows, labels, test_split=0.2):
    """
    Splits matching pairs of windows and labels into train and test splits.
    """
    if len(windows) != len(labels):
        raise ValueError(f"Length of windows ({len(windows)}) parameters and labels ({len(labels)}) don't match...")
    split_size: int = int(len(windows) * (1 - test_split))
    train_windows, train_labels = windows[:split_size], labels[:split_size]
    test_windows, test_labels = windows[split_size:], labels[split_size:]
    return train_windows, test_windows, train_labels, test_labels


train_w, test_w, train_l, test_l = make_train_test_splits(full_windows, full_labels)

import os


def create_model_checkpoint(model_name, save_path="model_experiments"):
    return tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(save_path, model_name), verbose=0,
                                              save_best_only=True)


def make_preds(model, input_data):
    """
    Uses model to make predictions on input_data.

    Parameters
    ----------
    model: trained model

    input_data: windowed input data (same kind of data model was trained on)

    Returns model predictions on input_data.
    """
    forecast = model.predict(input_data)
    return tf.squeeze(forecast)  # return 1D array of predictions


from tensorflow.keras import layers

tf.random.set_seed(42)

# MODEL 1: DENSE (WINDOW=7, HORIZON=1)
model_1 = tf.keras.Sequential([
    layers.Dense(128, activation="relu"),
    layers.Dense(HORIZON, activation="linear")  # linear activation is the same as having no activation.
], name="model_1_dense")

model_1.compile(loss="mae",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["mae"])

# model_1.fit(x=train_w,
#             y=train_l,
#             epochs=100,
#             verbose=0,
#             validation_data=(test_w, test_l),
#             callbacks=[create_model_checkpoint(model_name=model_1._name)])
#
# model_1.evaluate(test_w, test_l)
model_1 = tf.keras.models.load_model("model_experiments/model_1_dense")
model_1.evaluate(test_w, test_l)
model_1_preds = make_preds(model_1, test_w)
model_1_results = evaluate_preds(y_true=tf.squeeze(test_l), y_pred=model_1_preds)

# print model_1 results:
# offset: int = 300
# plt.figure(figsize=(10, 7))
# plot_time_series(timesteps=x_test[-len(test_w):], values=test_l, start=offset, label="Test data")
# plot_time_series(timesteps=x_test[-len(test_w):], values=model_1_preds, start=offset, format="-",
#                  label="Model 1 predictions")

# MODEL 2: DENSE (WINDOW=30, HORIZON=1)
HORIZON = 1
WINDOW_SIZE = 30
full_windows, full_labels = make_windows(prices, window_size=WINDOW_SIZE, horizon=HORIZON)
train_w, test_w, train_l, test_l = make_train_test_splits(windows=full_windows, labels=full_labels)
model_2 = tf.keras.Sequential([
    layers.Dense(128, activation="relu"),
    layers.Dense(HORIZON)
], name="model_2_dense")
model_2.compile(loss="mae", optimizer=tf.keras.optimizers.Adam())
# model_2.fit(train_w,
#             train_l,
#             epochs=100,
#             batch_size=128,
#             verbose=0,
#             validation_data=(test_w, test_l),
#             callbacks=[create_model_checkpoint(model_name=model_2._name)])
# model_2.evaluate(test_w, test_l)
model_2 = tf.keras.models.load_model("model_experiments/model_2_dense/")
model_2.evaluate(test_w, test_l)
model_2_preds = make_preds(model_2, input_data=test_w)
model_2_results = evaluate_preds(y_true=tf.squeeze(test_l), y_pred=model_2_preds)

# MODEL 3: DENSE (WINDOW=30, HORIZON=7)
HORIZON = 7
WINDOW_SIZE = 30
full_windows, full_labels = make_windows(prices, window_size=WINDOW_SIZE, horizon=HORIZON)
train_w, test_w, train_l, test_l = make_train_test_splits(windows=full_windows, labels=full_labels)
model_3 = tf.keras.Sequential([
    layers.Dense(128, activation="relu"),
    layers.Dense(HORIZON)
], name="model_3_dense")
model_3.compile(loss="mae", optimizer=tf.keras.optimizers.Adam())
model_3.fit(train_w,
            train_l,
            epochs=100,
            batch_size=128,
            verbose=0,
            validation_data=(test_w, test_l),
            callbacks=[create_model_checkpoint(model_name=model_3._name)])
model_3.evaluate(test_w, test_l)
model_3 = tf.keras.models.load_model("model_experiments/model_3_dense/")
model_3.evaluate(test_w, test_l)
model_3_preds = make_preds(model_3, input_data=test_w)
model_3_results = evaluate_preds(y_true=tf.squeeze(test_l), y_pred=model_3_preds)

offset = 300
plt.figure(figsize=(10, 7))
# Plot model_3_preds by aggregating them (note: this condenses information so the preds will look fruther ahead than the test data)
plot_time_series(timesteps=x_test[-len(test_w):],
                 values=test_l[:, 0],
                 start=offset,
                 label="Test_data")
plot_time_series(timesteps=x_test[-len(test_w):],
                 values=tf.reduce_mean(model_3_preds, axis=1),
                 format="-",
                 start=offset,
                 label="Model 3 predictions")

pd.DataFrame(
    {"naive": naive_results["mae"],
     "horizon_1_window_7": model_1_results["mae"],
     "horizon_1_window_30": model_2_results["mae"],
     "horizon_7_window_30": model_3_results["mae"]},
    index=["mae"]).plot(figsize=(10, 7), kind="bar");

# MODEL 4: CONV1D
tf.random.set_seed(42)
HORIZON = 1  # predict next day
WINDOW_SIZE = 7  # use previous week worth of data
# Create windowed dataset
full_windows, full_labels = make_windows(prices, window_size=WINDOW_SIZE, horizon=HORIZON)
train_w, test_w, train_l, test_l = make_train_test_splits(windows=full_windows, labels=full_labels)

# Before we pass our data to the Conv1D layer, we have to reshape it in order to make sure it works
x = tf.constant(train_w[0])
expand_dims_layer = layers.Lambda(lambda x: tf.expand_dims(x, axis=1))  # add an extra dimension for timesteps
print(f"Original shape: {x.shape}")  # (WINDOW_SIZE)
print(f"Expanded shape: {expand_dims_layer(x).shape}")  # (WINDOW_SIZE, input_dim)
print(f"Original values with expanded shape:\n {expand_dims_layer(x)}")

# Create model
model_4 = tf.keras.Sequential([
    # Create Lambda layer to reshape inputs, without this layer, the model will error
    layers.Lambda(lambda x: tf.expand_dims(x, axis=1)),
    # resize the inputs to adjust for window size / Conv1D 3D input requirements
    layers.Conv1D(filters=128, kernel_size=5, padding="causal", activation="relu"),
    layers.Dense(HORIZON)
], name="model_4_conv1D")

# Compile model
model_4.compile(loss="mae",
                optimizer=tf.keras.optimizers.Adam())

# Fit model
# model_4.fit(train_w,
#             train_l,
#             batch_size=128,
#             epochs=100,
#             verbose=0,
#             validation_data=(test_w, test_l),
#             callbacks=[create_model_checkpoint(model_name=model_4._name)])
# print(model_4.summary())
model_4 = tf.keras.models.load_model("model_experiments/model_4_conv1D")
model_4.evaluate(test_w, test_l)
model_4_preds = make_preds(model_4, test_w)
model_4_results = evaluate_preds(y_true=tf.squeeze(test_l), y_pred=model_4_preds)

print(f"Naive results: {naive_results}")
print(f"model_1 results: {model_1_results}")
print(f"model_2 results: {model_2_results}")
print(f"model_3 results: {model_3_results}")
print(f"model_4 results: {model_4_results}")