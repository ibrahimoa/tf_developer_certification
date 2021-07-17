import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3

local_weights_file = 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model = InceptionV3(input_shape=(150, 150, 3),
                                include_top=False,
                                weights=None)

pre_trained_model.load_weights(local_weights_file)
for layer in pre_trained_model.layers:
    layer.trainable = False  # Lock them: They are not trainable (we don't want to change them)

print(pre_trained_model.summary())
last_layer = pre_trained_model.get_layer('mixed7')
last_output = last_layer.output

x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)  # Between 0 and 1 and it's the fraction of units to drop.
x = layers.Dense(1, activation='sigmoid')(x)

# By dropping some out, we achieve the effect of neighbors not affecting each other too much and potentially
# removing overfitting.

model = Model(pre_trained_model.input, x)
model.compile(optimizer=RMSprop(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Generate the data:

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

train_datagen = ImageDataGenerator(rescale=1.0 / 255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

valid_datagen = ImageDataGenerator(rescale=1.0 / 255.)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=train_batch_size,
                                                    class_mode='binary',
                                                    target_size=(150, 150))
validation_generator = valid_datagen.flow_from_directory(validation_dir,
                                                         batch_size=valid_batch_size,
                                                         class_mode='binary',
                                                         target_size=(150, 150))

# Fit our pre-trained model to the data (transfer learning):

history = model.fit(train_generator,
                    validation_data=validation_generator,
                    steps_per_epoch=train_steps,
                    epochs=25,
                    validation_steps=valid_steps,
                    verbose=2)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Training and validation accuracy')
plt.figure()

plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('Training and validation loss')

plt.show()
