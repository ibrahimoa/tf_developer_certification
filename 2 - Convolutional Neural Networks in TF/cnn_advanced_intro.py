import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import os

base_dir = 'dogs_vs_cats'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

print('Training:')
print('\t- cat images: ', len(os.listdir(train_cats_dir)))
print('\t- dog images: ', len(os.listdir(train_dogs_dir)))
print('\nValidation:')
print('\t- cat images: ', len(os.listdir(validation_cats_dir)))
print('\t- dog images: ', len(os.listdir(validation_dogs_dir)), '\n')

train_batch_size: int = 32
valid_batch_size: int = 16
train_steps: int = int((len(os.listdir(train_cats_dir)) + len(os.listdir(train_dogs_dir))) / train_batch_size)
valid_steps: int = int((len(os.listdir(validation_cats_dir)) + len(os.listdir(validation_dogs_dir))) / valid_batch_size)

# Rescale all images by 1./255
train_datagen = ImageDataGenerator(rescale=1.0 / 255.)
test_datagen = ImageDataGenerator(rescale=1.0 / 255.)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=train_batch_size,
                                                    class_mode='binary',
                                                    target_size=(300, 300))
validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        batch_size=valid_batch_size,
                                                        class_mode='binary',
                                                        target_size=(300, 300))

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('cats') and 1 for the other ('dogs')
    tf.keras.layers.Dense(1, activation='sigmoid')
])

print(model.summary())

model.compile(optimizer=RMSprop(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_generator,
                    validation_data=validation_generator,
                    steps_per_epoch=train_steps,
                    epochs=15,
                    validation_steps=valid_steps,
                    verbose=2)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))  # Get number of epochs

plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Training and validation accuracy')
plt.figure()

plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('Training and validation loss')

plt.show()
