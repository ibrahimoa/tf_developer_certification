import keras
import numpy as np
from keras import layers

model = keras.Sequential([layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')  # stochastic gradient descent
# y = 2x - 1
xs = np.array([-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-9.0, -7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
model.fit(xs, ys, epochs=250)  # 500 times trough the training loop
print(model.predict([-5.0, 5.0]))
