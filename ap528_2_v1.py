# This Python 3 environment comes with many helpful analytics libraries installed

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt 
import plotly.express as px
import os
import cv2
from PIL import Image
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB0, Xception
from tensorflow.keras.optimizers import Adam
import os

input_dir = "../input/cassava-leaf-disease-classification"
input_model_dir = "../input/cassavacompetition"


TARGET_SIZE = 512

model_1 = keras.models.load_model('../input/efficientnet-trained-models/EffNetB0_512_8-1.h5')
model_2 = keras.models.load_model('../input/efficientnet-trained-models/model2_Marcellus_efnB3.h5')

submission_file = pd.read_csv(os.path.join('../input/cassava-leaf-disease-classification/sample_submission.csv'))

preds = []

for image_id in submission_file.image_id:
    image = Image.open(os.path.join(f'../input/cassava-leaf-disease-classification/test_images/{image_id}'))
    image = image.resize((TARGET_SIZE, TARGET_SIZE))
    image = np.expand_dims(image, axis = 0)

    pred = np.argmax(model_1.predict(image))
    if (pred == 0):
      pred = 3
    else:
      pred = np.argmax(model_2.predict(image))
      if (pred == 3):
         pred = 4

    preds.append(pred)
   
submission_file['label'] = preds
submission_file

submission_file.to_csv('submission.csv', index = False)