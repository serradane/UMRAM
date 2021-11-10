import os, shutil, random
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np

dirpath = 'C:/Users/zehra/Desktop/METU'
destDirectory = 'C:/Users/zehra/Desktop/UMRAM/ABIDE_pcp/Data/glass_brain_images2'

filenames = random.sample(os.listdir(dirpath), 2)
for fname in filenames:
    srcpath = os.path.join(dirpath, fname)
    shutil.copyfile(srcpath, destDirectory)