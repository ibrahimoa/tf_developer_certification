import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os


print(tf.__version__)
base_dir = 'G:\Personal\Cursos\Convolutional Neural Networks in TF\Dogs_vs_cats'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

print('total training cat images: ', len(os.listdir(train_cats_dir)))
print('total training dog images: ', len(os.listdir(train_dogs_dir)))
print('total validation cat images: ', len(os.listdir(validation_cats_dir)))
print('total validation dog images: ', len(os.listdir(validation_dogs_dir)))


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

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=16,
                                                    class_mode='binary',
                                                    target_size=(300, 300))
validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        batch_size=16,
                                                        class_mode='binary',
                                                        target_size=(300, 300))


model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('cats') and 1 for the other ('dogs')
    tf.keras.layers.Dense(1, activation='sigmoid')
])

print(model.summary())

model.compile(optimizer=RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit_generator(train_generator,
                    validation_data=validation_generator,
                    steps_per_epoch=375, #16*375 = 6000
                    epochs=50,
                    validation_steps=63, #16*63 = 1000
                    verbose=1)

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
