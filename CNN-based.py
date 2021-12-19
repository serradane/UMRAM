import os, shutil
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from imutils import paths
import numpy as np
import random
import matplotlib.pyplot as plt
from nilearn import image
from nilearn.image import iter_img



# LOADING IMAGES

# here base_dir will be 
testglass = []
teststat = []
testglass_asd = []
testglass_control = []
teststat_asd = []
teststat_control = []
trainglass_control = []
trainglass_asd = []
trainstat_control = []
trainstat_asd = []
datalist = []
train = []
test = []
labels_test = []
labels_train = []
test_data_stat = []
#file isimleri asd icin
glass_dir_asd = 'C:/Users/zehra/Desktop/UMRAM/ABIDE_pcp/Data/glass_brain_images'
stat_dir_asd = 'C:/Users/zehra/Desktop/UMRAM/ABIDE_pcp/Data/stat_images'
glass_dir_control = 'C:/Users/zehra/Desktop/UMRAM/ABIDE_pcp/Data/glass_brain_control'
stat_dir_control = 'C:/Users/zehra/Desktop/UMRAM/ABIDE_pcp/Data/stat_control'

#TODO:bu kismin aynisini kontrol icin de yap ve array tut 0-1 label 0 control icin olacak okurken takibini yap ya da image generate ederken isminde yaz listedeki isimlerden
#tekrar arraye labellari doldurmak.

np.random.seed(1234)

datalistglassasd = os.listdir(glass_dir_asd)
datalistglasscontrol = os.listdir(glass_dir_control)
dataliststatasd = os.listdir(stat_dir_asd)
dataliststatcontrol = os.listdir(stat_dir_control)

# datalistglass = datalistglasscontrol + datalistglassasd
# dataliststat = dataliststatasd + dataliststatcontrol

size = len(datalistglassasd)
trainglass_asd = random.sample(datalistglassasd, int(size*0.8))
for i in datalistglassasd:
    if not i in trainglass_asd:
        testglass_asd.append(i)

size = len(dataliststatasd)
trainstat_asd = random.sample(dataliststatasd, int(size*0.8))
for i in dataliststatasd:
    if not i in trainstat_asd:
        teststat_asd.append(i)

size = len(datalistglasscontrol)
trainglass_control = random.sample(datalistglasscontrol, int(size*0.8))
for i in datalistglasscontrol:
    if not i in trainglass_control:
        testglass_control.append(i)

size = len(dataliststatcontrol)
trainstat_control = random.sample(dataliststatcontrol, int(size*0.8))
for i in dataliststatcontrol:
    if not i in trainstat_control:
        teststat_control.append(i)


#imagelari split etmek icin 
def load_imgs(imagePaths, inp_dim, inp_dir):
    data = []
    for imagePath in imagePaths:
        for subject in os.listdir(os.path.join(inp_dir,imagePath)):
            subject = os.path.join(os.path.join(inp_dir,imagePath), subject)
            image = load_img(subject, target_size=(inp_dim, inp_dim, 3))
            image = img_to_array(image)
            image /= 255
            data.append(image)
    data = np.array(data, dtype="float32")
    return data

# BIG ERROR
# notice here we might mix images from different subjects since imagePaths is somehow ill defined
# we need to keep different imagePaths for different subjects
train_data_glass_asd = load_imgs(trainglass_asd, 150, glass_dir_asd)
test_data_glass_asd = load_imgs(testglass_asd, 150, glass_dir_asd)

train_data_glass_c = load_imgs(trainglass_control, 150, glass_dir_control)
test_data_glass_c = load_imgs(testglass_control, 150, glass_dir_control)

train_data_glass = np.concatenate((train_data_glass_asd, train_data_glass_c), axis = 0)
for i in enumerate(train_data_glass_asd):
    labels_train.append(1)
for i in enumerate(train_data_glass_c):
    labels_train.append(0)

test_data_glass = np.concatenate((test_data_glass_asd, test_data_glass_c), axis = 0)
for i in enumerate(test_data_glass_asd):
    labels_test.append(1)
for i in enumerate(test_data_glass_c):
    labels_test.append(0)

glass_train_converted = tf.image.rgb_to_grayscale(train_data_glass)
glass_test_converted = tf.image.rgb_to_grayscale(test_data_glass)

train_data_stat_asd = load_imgs(trainstat_asd, 150, stat_dir_asd)
test_data_stat_asd = load_imgs(teststat_asd, 150, stat_dir_asd)

train_data_stat_control = load_imgs(trainstat_control, 150, stat_dir_control)
test_data_stat_control = load_imgs(teststat_control, 150, stat_dir_control)

print(np.shape(train_data_stat_asd))
print(np.shape(train_data_stat_control))
train_data_stat = np.concatenate((train_data_stat_asd, train_data_stat_control), axis = 0)
test_data_stat = np.concatenate((test_data_stat_asd, test_data_stat_control), axis = 0)

print(np.shape(train_data_glass))
for i in range (len(train_data_glass)):
    train.append(np.concatenate((glass_train_converted[i], train_data_stat[i]), axis=2))

for i in range (len(test_data_glass)):
    test.append(np.concatenate((glass_test_converted[i], test_data_stat[i]), axis=2))

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


train = np.array(train)
test = np.array(test)
print(np.shape(train))
print(np.shape(test))

print(labels_train)
print(labels_test)
print(len(labels_train))
print(len(labels_test))



train_generator = train_datagen.flow(
    train,
    labels_train,
    #target_size=(150, 150, 3),
    batch_size=32,
    #class_mode='binary'
    )

test_generator = test_datagen.flow(
    test,
    labels_test,
    #target_size=(150, 150, 3),
    batch_size=32,
    #class_mode='binary'
    )

#plt.imshow(image[:,:,:3]) ve plt.imshow(image[:,:,3]) train_generator'da olusan image'a benzer seyler cikiyor mu?


print(train_generator)
print(test_generator)

# MODEL DEFINITION

from tensorflow import keras
from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 4)))
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