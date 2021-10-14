import os, shutil
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np




# LOADING IMAGES

# here base_dir will be 

glass_dir = 'C:/Users/zehra/Desktop/UMRAM/ABIDE_pcp/Data/glass_brain_images'
#stat_dir = 'C:/Users/zehra/Desktop/UMRAM/ABIDE_pcp/Data/stat_images'
imagePaths     = list(paths.list_images(glass_dir))

#imagelari split etmek icin 
def load_imgs(imagePaths, inp_dim):
	data = []
	labels = []
	# loop over the image paths
	for imagePath in imagePaths:
		# extract the class label from the filename
		label = imagePath.split(os.path.sep)[-2]

		image = load_img(imagePath, target_size=(inp_dim, inp_dim))
		image = img_to_array(image)

		data.append(image)
		labels.append(label)
		
	# convert the data and labels to NumPy arrays
	data = np.array(data, dtype="float32")
	labels = np.array(labels)
	LB = LabelBinarizer()
	labels = LB.fit_transform(labels)
	#labels = to_categorical(labels)

	return data, labels

# BIG ERROR
# notice here we might mix images from different subjects since imagePaths is somehow ill defined
# we need to keep different imagePaths for different subjects
data, labels = load_imgs(imagePaths, 150)
X_train, X_rem, y_train, y_rem = train_test_split(data, labels, train_size=0.8)
X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, test_size=0.5)




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





# Data Preprocessing

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

"""
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')
"""

history = model.fit_generator(
    train_datagen.flow(X_train, y_train, batch_size=32),
    steps_per_epoch=100,
    epochs=300,
    validation_data=(X_valid, y_valid),
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
"""

test_loss, test_acc = model.evaluate_generator((X_test, y_test), steps=10)

print('test loss: ', test_loss)
print('test acc: ', test_acc)
