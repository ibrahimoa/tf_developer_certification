import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from time_series import *

dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True) # drop_remainder -> Only give us windows with 5 items
dataset = dataset.flat_map(lambda window: window.batch(5))
dataset = dataset.map(lambda window: (window[:-1], window[-1:])) # Create x's and y's
# Sequence bias is when the order of things can impact the selection of things
dataset = dataset.shuffle(buffer_size=10) # 10 because we have 10 data items
dataset = dataset.batch(2).prefetch(1) # In batches of two
for x,y in dataset:
    print("x = ", x.numpy())
    print("y = ", y.numpy())

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset

time = np.arange(4 * 365 + 1, dtype="float32")
baseline = 10
amplitude = 40
slope = 0.05
noise_level = 5
split_time = 1000
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude) + white_noise(time, noise_level, seed=42)
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

window_size = 20
batch_size = 32
shuffle_buffer_size = 1000

lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10 ** (epoch / 20))
optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)
dataset = windowed_dataset(series, window_size, batch_size, shuffle_buffer_size)

###################################################### LINEAR REGRESSION ###################################################

l0 = layers.Dense(1, input_shape=[window_size]) # Simple linear regression
model = tf.keras.models.Sequential([l0])
model.compile(loss="mse", optimizer=optimizer)
model.fit(dataset,
          epochs=100,
          verbose=1)
print("Layer weights {}".format(l0.get_weights()))

forecast = []
for times in range(len(series) - window_size):
    forecast.append(model.predict(series[times:times + window_size][np.newaxis]))
forecast = forecast[split_time-window_size:]
results = np.array(forecast)[:, 0, 0]
plt.figure(figsize=(10,6))
plot_series(time_valid, x_valid, plot=False, figure=False)
plot_series(time_valid, results, plot=False, figure=False)
plt.show()
print(tf.keras.metrics.mean_absolute_error(x_valid, results).numpy())

########################################################### DNN ########################################################

model = tf.keras.models.Sequential([
    layers.Dense(10, input_shape=[window_size], activation='relu'),
    layers.Dense(10, activation='relu'),
    layers.Dense(1)
])
model.compile(loss="mse", optimizer=optimizer)
history = model.fit(dataset,
          epochs=100,
          verbose=1,
          callbacks=[lr_schedule])

# loss per epoch vs lr per epoch:
lrs = 1e-8 * (10 ** (np.arange(100) / 20))
plt.figure(figsize=(10,6))
plt.semilogx(lrs, history.history["loss"])
plt.axis([1e-8, 1e-3, 0, 300]) # Lowest value in the curve is approx 7e-6 -> Ideal lr
plt.show()

# Train with the ideal lr:
model = tf.keras.models.Sequential([
    layers.Dense(10, input_shape=[window_size], activation='relu'),
    layers.Dense(10, activation='relu'),
    layers.Dense(1)
])
optimizer = tf.keras.optimizers.SGD(lr=7e-6, momentum=0.9)
model.compile(loss="mse", optimizer=optimizer)
history = model.fit(dataset,
          epochs=500,
          verbose=1)
loss = history.history["loss"]
epochs = range(len(loss))
plt.figure(figsize=(10,6))
plt.plot(epochs, loss, 'b', label='Training Loss')
plt.show()
forecast = []
for times in range(len(series) - window_size):
    forecast.append(model.predict(series[times:times + window_size][np.newaxis]))
forecast = forecast[split_time-window_size:]
results = np.array(forecast)[:, 0, 0]
plt.figure(figsize=(10,6))
plot_series(time_valid, x_valid, plot=False, figure=False)
plot_series(time_valid, results, plot=False, figure=False)
plt.show()
print(tf.keras.metrics.mean_absolute_error(x_valid, results).numpy())