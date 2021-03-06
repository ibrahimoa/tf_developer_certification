# With ImageGenerator we can use images with different sizes and also do 'auto.labeling' by setting a default
# directory and then making two folders: Training and Validation (for example), in each sub-directory we can
# create another sub-directory with name 'Horses' and store there every horse image we want. All that images
# are going to be labeled as Horses automatically.

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop

train_dir = 'horse-or-human\\train'
validation_dir = 'horse-or-human\\validation'

train_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(300, 300),  # images need to be resized to make them consistent
    batch_size=32,  # experiment to see the impact on the performance
    class_mode='binary'  # classifies between two classes/types
)

validation_generator = train_datagen.flow_from_directory(
    validation_dir,
    target_size=(300, 300),  # images need to be resized to make them consistent
    batch_size=32,  # experiment to see the impact on the performance
    class_mode='binary'  # classifies between two classes/types
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),  # 3 bytes per pixel -> Coloured images
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # sigmoid is great to binary classification (1 one class, 0 the other one)
])
print(model.summary())

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(learning_rate=0.001),
              metrics=['accuracy'])  # or 'acc'

model.fit(
    train_generator,
    steps_per_epoch=32,  # 128*8 = 1024 images (what we have)
    epochs=10,
    validation_data=validation_generator,
    validation_steps=8,  # 8*32 = 256 images
    verbose=1)  # how much to display while training is going on. With verbose set to 2, we'll get a little
# less animation hiding the epoch progress.
