import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

base_dir = 'rock_paper_scissors'
train_dir = os.path.join(base_dir, 'rps')
validation_dir = os.path.join(base_dir, 'rps-test-set')
evaluation_dir = os.path.join(base_dir, 'rps-validation')

train_rock_dir = os.path.join(train_dir, 'rock')
train_paper_dir = os.path.join(train_dir, 'paper')
train_scissors_dir = os.path.join(train_dir, 'scissors')
validation_rock_dir = os.path.join(validation_dir, 'rock')
validation_paper_dir = os.path.join(validation_dir, 'paper')
validation_scissors_dir = os.path.join(validation_dir, 'scissors')

print('Training:')
print('\t- rock images: ', len(os.listdir(train_rock_dir)))
print('\t- paper images: ', len(os.listdir(train_paper_dir)))
print('\t- scissors images: ', len(os.listdir(train_scissors_dir)))
print('\nValidation:')
print('\t- rock images: ', len(os.listdir(validation_rock_dir)))
print('\t- paper images: ', len(os.listdir(validation_paper_dir)))
print('\t- scissors images: ', len(os.listdir(validation_scissors_dir)))
print('\nEvaluation:')
print('\t- unlabeled images: ', len(os.listdir(evaluation_dir)), '\n')

train_batch_size: int = 48
valid_batch_size: int = 24
train_steps: int = int(
    (len(os.listdir(train_rock_dir)) + len(os.listdir(train_paper_dir)) + len(os.listdir(train_scissors_dir))) / train_batch_size)
valid_steps: int = int((len(os.listdir(validation_rock_dir)) + len(os.listdir(validation_paper_dir)) + len(
    os.listdir(validation_scissors_dir))) / valid_batch_size)

# Rescale all images by 1./255

train_datagen = ImageDataGenerator(rescale=1.0 / 255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1.0 / 255.)
evaluation_datagen = ImageDataGenerator(rescale=1.0 / 255.)

# class_mode='categorical' for multiclass classification
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=train_batch_size,
                                                    target_size=(300, 300),
                                                    class_mode='categorical',
                                                    classes=['rock', 'paper', 'scissors'])

validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        batch_size=valid_batch_size,
                                                        target_size=(300, 300),
                                                        class_mode='categorical',
                                                        classes=['rock', 'paper', 'scissors'])

evaluation_generator = evaluation_datagen.flow_from_directory(evaluation_dir,
                                                              batch_size=len(os.listdir(evaluation_dir)),
                                                              target_size=(300, 300),
                                                              class_mode='categorical',
                                                              classes=['rock', 'paper', 'scissors'])
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(4, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(8, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(8, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(256, activation='relu'),
    # It's not a binary classification anymore, so we need a neuron per each class.
    tf.keras.layers.Dense(3, activation='softmax')
])

print(model.summary())

model.compile(optimizer=RMSprop(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_generator,
                    validation_data=validation_generator,
                    steps_per_epoch=train_steps,
                    epochs=25,
                    validation_steps=valid_steps,
                    verbose=2)

test_loss, test_acc = model.evaluate(evaluation_generator)
print(f'Test performance:\n\tTest Loss: {test_loss}\n\tTest Accuracy: {test_acc}')
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
