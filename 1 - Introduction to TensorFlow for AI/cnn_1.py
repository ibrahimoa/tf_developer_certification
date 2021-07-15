import tensorflow as tf
from tensorflow import keras

print(tf.test.is_built_with_cuda())
print(tf.test.is_gpu_available())

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # print(logs)
        if logs is None:
            logs = {}
        if (logs.get('loss') < 0.15):  # In previous tf version you may need to use 'acc' instead of 'accuracy'
            print('\nLoss is below 15% so cancelling training')
            self.model.stop_training = True


callback3 = myCallback()

mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images = training_images.reshape(60000, 28, 28, 1)
training_images = training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images / 255.0

model = keras.models.Sequential([
    # 64 3x3 filters. relu -> negative values are throwing away 28x28x1 -> single byte for color depth (grayscale)
    # first filters are random.
    keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    # Maxpooling -> We are going to keep the maximum value (and there is the name)
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(16, (2, 2), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Flatten(),
    keras.layers.Dense(200, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print(model.summary())
model.fit(training_images, training_labels, epochs=15, callbacks=[callback3])
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test loss = {test_loss}\nTest acduracy = {test_acc}')
