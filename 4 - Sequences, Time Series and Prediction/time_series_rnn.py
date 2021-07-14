import numpy as np
import tensorflow as tf
# Lambda -> Arbitrary operations to improve out model
from tensorflow.keras.layers import SimpleRNN, Dense, Lambda
from tensorflow.keras.models import Sequential
from time_series import *

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
train_set = windowed_dataset(x_train, window_size, batch_size=128, shuffle_buffer=shuffle_buffer_size)
optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)
# lr goes from 1e-8 to 1e-3:
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10 ** (epoch/20))

model = Sequential([
    Lambda(lambda  x: tf.expand_dims(x, axis=-1), input_shape=[None]),
    SimpleRNN(40, return_sequences=True),
    SimpleRNN(40),
    Dense(1),
    Lambda(lambda x: x * 100.0)
])

model.compile(loss=tf.keras.losses.Huber(), # Less sensitive to outliers (valores atípicos)
              optimizer=optimizer,
              metrics=["mae"])
history = model.fit(train_set,
          epochs=100,
          callbacks=[lr_schedule])

# loss per epoch vs lr per epoch:
lrs = 1e-8 * (10 ** (np.arange(100) / 20))
plt.figure(figsize=(10,6))
plt.semilogx(lrs, history.history["loss"])
plt.axis([1e-8, 1e-3, 0, 30]) # Lowest value in the curve is approx 7e-6 -> Ideal lr
plt.show()

# The ideal learning rate is between 10e-6 and 10e-5. Let's set it at 5e-5
tf.keras.backend.clear_session() # Free memory
tf.random.set_seed(51)
np.random.seed(51)
training_data = windowed_dataset(x_train, window_size, batch_size=128, shuffle_buffer=shuffle_buffer_size)
model = Sequential([
    Lambda(lambda  x: tf.expand_dims(x, axis=-1), input_shape=[None]),
    SimpleRNN(40, return_sequences=True),
    SimpleRNN(40),
    Dense(1),
    Lambda(lambda x: x * 100.0)
])

optimizer = tf.keras.optimizers.SGD(lr=5e-5, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(), # Less sensitive to outliers (valores atípicos)
              optimizer=optimizer,
              metrics=["mae"])
history = model.fit(training_data, epochs=400)

mae=history.history['mae']
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

#------------------------------------------------
# Plot MAE and Loss
#------------------------------------------------
plt.figure()
plt.plot(epochs, mae, 'r')
plt.plot(epochs, loss, 'b')
plt.title('MAE and Loss')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["MAE", "Loss"])
plt.show()
#------------------------------------------------
# Plot Zoomed MAE and Loss
#------------------------------------------------
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