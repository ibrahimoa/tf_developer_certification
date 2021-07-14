import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Data affected by time usally have trend, seasonality, autocorrelation and noise (white/pink) -> Patterns

def plot_series(time, series, format="-", plot=True, figure=True):
    if figure:
        plt.figure(figsize=(10, 6))
    plt.plot(time, series, format)
    plt.xlabel("time")
    plt.ylabel("value")
    plt.grid(True)
    if plot:
        plt.show()

def trend(time, slope=0.0):
    return slope * time

def seasonal_pattern(season_time):
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),  # Value when the condition season_time < 0.4 is true
                    1 / np.exp(3 * season_time))  # Value when season_time >= 0.4

def seasonality(time, period, amplitude=1, phase=0):
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

def white_noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

def autocorrelation_1(time, amplitude, seed=None):  # Autocorrelation have delay included
    rnd = np.random.RandomState(seed)
    phi_1 = 0.5
    phi_2 = -0.1
    ar = rnd.randn(len(time) + 50)
    ar[:50] = 100
    for step in range(50, len(time) + 50):
        ar[step] += phi_1 * ar[step - 50]
        ar[step] += phi_2 * ar[step - 33]
    return ar[50:] * amplitude

def impulses(time, num_impulses, amplitude=1, seed=None):
    rnd = np.random.RandomState(seed)
    impulse_indices = rnd.randint(len(time), size=num_impulses)
    series = np.zeros(len(time))
    for index in impulse_indices:
        series[index] += rnd.rand() * amplitude
    return series

def autocorrelation_2(source, phi_s):
    ar = source.copy()
    max_lag = len(phi_s)
    for step, value in enumerate(source):
        for lag, phi in phi_s.items():
            if step - lag > 0:
                ar[step] += phi * ar[step - lag]
    return ar

def moving_average_forecast(series, window_size):
    """Forecasts the mean if the last few values.
       If window_size=1, then this is equivalent to naive forecast"""
    forecast = []
    for time in range(len(series) - window_size):
        forecast.append(series[time:time + window_size].mean())
    return np.array(forecast)


# In time series, we usually split our system in a fixed partitioning:
# Training period
# Validation period
# Test period

# Metrics:
# errors = forecasts - actual
# mse (mean square error) = np.square(errors).mean()
# rmse = np.sqrt(mse)
# mae (mean absolute error/deviation) = np.abs(errors).mean()
# mape (mean absolute percentage error) = np.abs(errors / x_valid).mean()
# keras.metrics.mean_absolute_error(x_valid, forecast).numpy()

# Moving average : eliminates the noise and indicates the general trend
# Differencing : Instead of studying the time series itself, we study the difference between the value at
# time T and the value at an  earlier period.
# DifferencingInverse(Moving average(Differencing)) -> Little better than naive forecast
# If we also do 'Moving average' on the final forecast we reduce the noise and have better results
# (Smoothing both past and present values)

# Moving averages using centered windows can be more accurate than using trailing windows (past and future)
# But we can't use centered windows to smooth present values since we don't know future values.
# However, to smooth past values we can afford to use centered windows.

if __name__ == "__main__":

    time = np.arange(4 * 365 + 1, dtype="float32")
    baseline = 10
    amplitude = 40
    slope = 0.05
    noise_level = 5

    #series = trend(time, 0.1)
    #plot_series(time, series)

    #series = seasonality(time, period=365, amplitude=amplitude)
    #plot_series(time, series)

    #series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
    #plot_series(time, series)

    #noise = white_noise(time, noise_level, seed=42)
    #plot_series(time, noise)

    #series = autocorrelation_1(time, 10, seed=42)
    #plot_series(time[:200], series[:200])

    #signal = impulses(time, 10, seed=42)
    #plot_series(time, signal)

    #series = autocorrelation_2(signal, {1: 0.99})
    #plot_series(time, series, plot=False)
    #plt.plot(time, signal, "-k")
    #plt.show()

    #signal = impulses(time, 10, seed=42)
    #series = autocorrelation_2(signal, {1: 0.70, 50: 0.2})
    #plot_series(time, series, plot=False)
    #plt.plot(time, signal, "-k")
    #plt.show()

    #from pandas.plotting import autocorrelation_plot

    #plt.figure(figsize=(10, 6))
    #autocorrelation_plot(series)
    #plt.show()

    series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude) + white_noise(time, noise_level, seed=42)
    split_time = 1000
    time_train = time[:split_time]
    x_train = series[:split_time]
    time_valid = time[split_time:]
    x_valid = series[split_time:]
    plot_series(time_train, x_train)
    plot_series(time_valid, x_valid)

    # Naive forecast:
    naive_forecast = series[split_time - 1: -1]
    plt.figure(figsize=(10, 6))
    plot_series(time_valid, x_valid, plot=False, figure=False)
    plot_series(time_valid, naive_forecast, plot=False, figure=False)
    plt.show()
    print(tf.keras.metrics.mean_squared_error(x_valid, naive_forecast).numpy())
    print(tf.keras.metrics.mean_absolute_error(x_valid, naive_forecast).numpy())

    # Moving series
    moving_average = moving_average_forecast(series, 30)[split_time - 30:]
    plt.figure(figsize=(10, 6))
    plot_series(time_valid, x_valid, plot=False, figure=False)
    plot_series(time_valid, moving_average, plot=False, figure=False)
    plt.show()
    print('\nMoving average:')
    print(tf.keras.metrics.mean_squared_error(x_valid, moving_average).numpy())
    print(tf.keras.metrics.mean_absolute_error(x_valid, moving_average).numpy())

    # Differencing series:
    diff_series = (series[365:] - series[:-365])
    diff_time = time[365:]
    #plot_series(diff_time, diff_series)

    # Moving average + differencing:
    diff_moving_avg = moving_average_forecast(diff_series, 50)[split_time - 365 - 50:]
    plt.figure(figsize=(10, 6))
    plot_series(time_valid, diff_series[split_time - 365:], plot=False, figure=False)
    plot_series(time_valid, diff_moving_avg, plot=False, figure=False)
    plt.show()

    # Now let's bring back the tren and seasonality by adding past values from t - 365
    diff_moving_avg_plus_past = series[split_time - 365: -365] + diff_moving_avg
    plt.figure(figsize=(10, 6))
    plot_series(time_valid, x_valid, plot=False, figure=False)
    plot_series(time_valid, diff_moving_avg_plus_past, plot=False, figure=False)
    plt.show()
    print('\nDifferencing moving average with past values:')
    print(tf.keras.metrics.mean_squared_error(x_valid, diff_moving_avg_plus_past).numpy())
    print(tf.keras.metrics.mean_absolute_error(x_valid, diff_moving_avg_plus_past).numpy())

    # Use moving average on past values to reduce the noise:
    diff_moving_avg_plus_smooth_past = moving_average_forecast(series[split_time - 370: -360], 10) + diff_moving_avg
    plt.figure(figsize=(10, 6))
    plot_series(time_valid, x_valid, plot=False, figure=False)
    plot_series(time_valid, diff_moving_avg_plus_smooth_past, plot=False, figure=False)
    plt.show()
    print('\nDifferencing moving average with smooth past values:')
    print(tf.keras.metrics.mean_squared_error(x_valid, diff_moving_avg_plus_smooth_past).numpy())
    print(tf.keras.metrics.mean_absolute_error(x_valid, diff_moving_avg_plus_smooth_past).numpy())