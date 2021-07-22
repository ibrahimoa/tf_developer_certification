from tensorflow.keras.layers import Dense, Lambda, Bidirectional, LSTM
from tensorflow.keras.models import Sequential
from time_series_1 import *


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
window_size = 20
batch_size = 32
shuffle_buffer_size = 1000

series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude) + white_noise(time, noise_level, seed=42)
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]
train_set = windowed_dataset(x_train, window_size, batch_size=batch_size, shuffle_buffer=shuffle_buffer_size)

model = Sequential([
    Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]),
    Bidirectional(LSTM(32, return_sequences=True)),
    Bidirectional(LSTM(32)),
    Dense(1),
    Lambda(lambda x: x * 100.0)
])

optimizer = tf.keras.optimizers.SGD(learning_rate=1e-5, momentum=0.9)  # We can get this optimum lr as in previous examples
model.compile(loss="mse",
              optimizer=optimizer,
              metrics=["mae"])
history = model.fit(train_set, epochs=500)

forecast = []
for times in range(len(series) - window_size):
    forecast.append(model.predict(series[times:times + window_size][np.newaxis]))
forecast = forecast[split_time - window_size:]
results = np.array(forecast)[:, 0, 0]
plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid, plot=False, figure=False)
plot_series(time_valid, results, plot=False, figure=False)
plt.show()
print(tf.keras.metrics.mean_absolute_error(x_valid, results).numpy())
