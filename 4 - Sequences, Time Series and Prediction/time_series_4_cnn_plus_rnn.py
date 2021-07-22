# Lambda -> Arbitrary operations to improve our model
from tensorflow.keras.layers import Dense, Lambda, Bidirectional, LSTM, Conv1D
from tensorflow.keras.models import Sequential
from time_series_1 import *


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast


time = np.arange(4 * 365 + 1, dtype="float32")
baseline = 10
amplitude = 40
slope = 0.05
noise_level = 5
split_time = 1000
window_size = 20
batch_size = 32  # The impact of this variable is high (Check the Machine Learning course ... )
shuffle_buffer_size = 1000

series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude) + white_noise(time, noise_level, seed=42)
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]
train_set = windowed_dataset(x_train, window_size, batch_size=batch_size, shuffle_buffer=shuffle_buffer_size)

model = Sequential([
    Conv1D(filters=32, kernel_size=5, strides=1, padding="causal", activation="relu", input_shape=[None, 1]),
    LSTM(32, return_sequences=True),
    LSTM(32),
    Dense(1),
    Lambda(lambda x: x * 200.0)
])

optimizer = tf.keras.optimizers.SGD(learning_rate=1e-5, momentum=0.9)  # We can get this optimum lr as in previous examples
model.compile(loss=tf.keras.losses.Huber(),  # Less sensitive to outliers
              optimizer=optimizer,
              metrics=["mae"])
history = model.fit(train_set, epochs=500)

rnn_forecast = model_forecast(model, series[..., np.newaxis], window_size)
rnn_forecast = rnn_forecast[split_time - window_size:-1, -1]
plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid, plot=False, figure=False)
plot_series(time_valid, rnn_forecast, plot=False, figure=False)
plt.show()
print('Performance:')
print(f'Mean absolute error: {tf.keras.metrics.mean_absolute_error(x_valid, rnn_forecast).numpy()}')

# -----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
# -----------------------------------------------------------
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
