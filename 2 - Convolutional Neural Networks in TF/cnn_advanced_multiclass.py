import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

base_dir = 'G:\Personal\Cursos\Convolutional Neural Networks in TF\\rock_paper_scissors'
train_dir = os.path.join(base_dir, 'rps')
validation_dir = os.path.join(base_dir, 'rps-test-set')
evaluation_dir = os.path.join(base_dir, 'rps-validation')

train_rock_dir = os.path.join(train_dir, 'rock')
train_paper_dir = os.path.join(train_dir, 'paper')
train_scissors_dir = os.path.join(train_dir, 'scissors')
validation_rock_dir = os.path.join(validation_dir, 'rock')
validation_paper_dir = os.path.join(validation_dir, 'paper')
validation_scissors_dir = os.path.join(validation_dir, 'scissors')


print('total training rock images: ', len(os.listdir(train_rock_dir)))
print('total training paper images: ', len(os.listdir(train_paper_dir)))
print('total training scissors images: ', len(os.listdir(train_scissors_dir)))
print('total validation rock images: ', len(os.listdir(validation_rock_dir)))
print('total validation paper images: ', len(os.listdir(validation_paper_dir)))
print('total validation scissors images: ', len(os.listdir(validation_scissors_dir)))


#Rescale all images by 1./255

train_datagen = ImageDataGenerator(rescale=1.0/255,
                                   rotation_range=40, #Range from 0 to 180 degrees to randomly rotate images. In this case it's going to rotate between 0 and 40 degrees
                                   width_shift_range=0.2, #Move image in this fram (20%)
                                   height_shift_range=0.2,
                                   shear_range=0.2, #Girar la imagen un 20%
                                   zoom_range=0.2, #Zoom up-to 20%
                                   horizontal_flip=True, #Efecto cámara: girar la imagen con respecto al eje vertical
                                   fill_mode='nearest') #Ckeck other options

test_datagen = ImageDataGenerator(rescale=1.0/255.)
evaluation_datagen = ImageDataGenerator(rescale=1.0/255.)

#class_mode='categorical' for multiclass classification
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=64,
                                                    target_size=(300, 300),
                                                    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        batch_size=64,
                                                        class_mode='categorical',
                                                        target_size=(300, 300))

evaluation_generator = evaluation_datagen.flow_from_directory(evaluation_dir,
                                                              batch_size=33,
                                                              class_mode='categorical',
                                                              target_size=(300, 300))
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

print(model.summary())

model.compile(optimizer=RMSprop(lr=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit_generator(train_generator,
                    validation_data=validation_generator,
                    steps_per_epoch=40,
                    epochs=25,
                    validation_steps=6,
                    verbose=1)

test_loss, test_acc = model.evaluate(evaluation_generator)

acc      = history.history['accuracy']
val_acc  = history.history['val_accuracy']
loss     = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc)) #Get number of epochs

plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Training and validation accuracy')
plt.figure()

plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('Training and validation loss')

plt.show()