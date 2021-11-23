import os, shutil
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from imutils import paths
import numpy as np
import random


# LOADING IMAGES

# here base_dir will be 
test = []
glass_dir = 'C:/Users/zehra/Desktop/UMRAM/ABIDE_pcp/Data/glass_brain_images'
#stat_dir = 'C:/Users/zehra/Desktop/UMRAM/ABIDE_pcp/Data/stat_images'

datalist = os.listdir(glass_dir)
size = len(datalist)
train = random.sample(datalist, int(size*0.8))
for i in datalist:
    if not i in train:
        test.append(i)

# dataliststat = os.listdir(glass_dir)
# size = len(dataliststat)
# train = random.sample(dataliststat, int(size*0.8))
# for i in dataliststat:
#     if not i in train:
#         teststat.append(i)

#imagelari split etmek icin 
def load_imgs(imagePaths, inp_dim):
    data = []
    for imagePath in imagePaths:
        for subject in os.listdir(os.path.join(glass_dir,imagePath)):
            print(subject)
            subject = os.path.join(os.path.join(glass_dir,imagePath), subject)
            image = load_img(subject, target_size=(inp_dim, inp_dim, 3))
            image = img_to_array(image)
            data.append(image)
    data = np.array(data, dtype="float32")
    return data

# BIG ERROR
# notice here we might mix images from different subjects since imagePaths is somehow ill defined
# we need to keep different imagePaths for different subjects
train_data = load_imgs(train, 150)
test_data= load_imgs(test, 150)

from keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
    train_data,
    target_size=(150, 150, 3),
    batch_size=32,
    class_mode='binary')

test_generator = test_datagen.flow_from_directory(
    test_data,
    target_size=(150, 150, 3),
    batch_size=32,
    class_mode='binary')


# MODEL DEFINITION

from tensorflow import keras
from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

print(model.summary())

from tensorflow.keras import optimizers

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])



history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=300,
    validation_data=test_generator,
    validation_steps=50)




# PLOTTING WHAT HAPPENED DURING TRAINING

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()



#Model Evaluation
#Simdilik test kismini cikardim, datayi bolerken zorluk yasadim cunku :')
# well we will definetely need this part, 
"""
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')


test_loss, test_acc = model.evaluate_generator((X_test, y_test), steps=10)

print('test loss: ', test_loss)
print('test acc: ', test_acc)
"""