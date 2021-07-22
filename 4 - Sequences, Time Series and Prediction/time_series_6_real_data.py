from tensorflow.keras.layers import Dense, Lambda, LSTM, Conv1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from time_series_1 import *

import csv

time_step = []
sunspots = []
with open("Sunspots.csv") as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    next(reader)  # Ignore first line since it's information and not data
    for row in reader:
        time_step.append(int(row[0]))
        sunspots.append(float(row[2]))

time = np.array(time_step)
series = np.array(sunspots)
plot_series(time, series)

split_time = 3000  # Total data: 3500. 3000 for training and 500 for validation
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

# Try numbers that are multiple of 3000 (training_data)
window_size = 60
batch_size = 150
shuffle_buffer_size = 1000


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast


dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
model = Sequential([
    Conv1D(filters=60, kernel_size=5, strides=1, padding='causal', activation='relu', input_shape=[None, 1]),
    LSTM(60, return_sequences=True),
    LSTM(60),
    Dense(30, activation='relu'),
    Dense(10, activation='relu'),
    Dense(1),
    Lambda(lambda x: x * 400)
])

lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10 ** (epoch / 20))
optimizer = SGD(learning_rate=1e-8, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer, metrics=['mae'])
history = model.fit(dataset, epochs=100, callbacks=[lr_schedule])

# loss per epoch vs lr per epoch:
lrs = 1e-8 * (10 ** (np.arange(100) / 20))
plt.figure(figsize=(10, 6))
plt.semilogx(lrs, history.history["loss"])
plt.axis([1e-8, 1e-3, 0, 30])  # Lowest value in the curve is approx 1e-5 -> Ideal lr
plt.show()

############################################### TRAIN WITH IDEAL LEARNING RATE ###############################################
tf.keras.backend.clear_session()
dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
model = Sequential([
    Conv1D(filters=32, kernel_size=5, strides=1, padding='causal', activation='relu', input_shape=[None, 1]),
    LSTM(32, return_sequences=True),
    LSTM(32),
    Dense(30, activation='relu'),
    Dense(10, activation='relu'),
    Dense(1),
    Lambda(lambda x: x * 400)
])
optimizer = SGD(learning_rate=1e-5, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer, metrics=['mae'])
history = model.fit(dataset, epochs=500)

rnn_forecast = model_forecast(model, series[..., np.newaxis], window_size)
rnn_forecast = rnn_forecast[split_time - window_size:-1, -1]
plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid, plot=False, figure=False)
plot_series(time_valid, rnn_forecast, plot=False, figure=False)
plt.show()
print(f'Mean absolute error: {tf.keras.metrics.mean_absolute_error(x_valid, rnn_forecast).numpy()}')

mae = history.history['mae']
loss = history.history['loss']
epochs = range(len(loss))  # Get number of epochs
# ------------------------------------------------
# Plot MAE and Loss
# ------------------------------------------------
plt.figure()
plt.plot(epochs, mae, 'r')
plt.plot(epochs, loss, 'b')
plt.title('MAE and Loss')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["MAE", "Loss"])
plt.show()
# ------------------------------------------------
# Plot Zoomed MAE and Loss
# ------------------------------------------------
epochs_zoom = epochs[200:]
mae_zoom = mae[200:]
loss_zoom = loss[200:]
plt.figure()
plt.plot(epochs_zoom, mae_zoom, 'r')
plt.plot(epochs_zoom, loss_zoom, 'b')
plt.title('MAE and Loss')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["MAE", "Loss"])
plt.show()
