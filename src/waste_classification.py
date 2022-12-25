########################
# Importing
########################
#!pip install ipython
#!pip install pandas
#!pip install opencv-python
#!pip install visualkeras
#!pip install pillow
#!pip install matplotlib
#!pip install scikit-image --user


import tensorflow as tf
import pandas as pd
import numpy as np
import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.client import device_lib
from skimage.io import imread, imshow
from skimage.transform import resize
import glob

#########################
# GPU check (Opsiyonel)
#########################
print(tf.__version__)

tf.test.is_gpu_available()


device_lib.list_local_devices()

##############################
# Dosyaları Yükleme -1
##############################
print(os.getcwd())

data_path = os.path.join(os.getcwd(), "src\\DATASET")
train_path = os.path.join(data_path, "TRAIN")
valid_path = os.path.join(data_path, "VALID")
test_path = os.path.join(data_path, "TEST")

generator = ImageDataGenerator(rescale=1. / 255)

train_batches = generator.flow_from_directory(directory=train_path,
                                              target_size=(224, 224),
                                              shuffle=True,
                                              batch_size=16)

valid_batches = generator.flow_from_directory(directory=valid_path,
                                             target_size=(224, 224),
                                             shuffle=True,
                                             batch_size=16)

test_batches = generator.flow_from_directory(directory=test_path,
                                             target_size=(224, 224),
                                             batch_size=8)

print(train_batches.class_indices, valid_batches.class_indices)



# 1- Dengeli olup olmadığını kontrol edelim

train_r = os.listdir(os.path.join(train_path, "R"))
train_o = os.listdir(os.path.join(train_path, "O"))
valid_r = os.listdir(os.path.join(test_path, "R"))
valid_o = os.listdir(os.path.join(test_path, "O"))

print("Train R: ", len(train_r))
print("Train O: ", len(train_o))
print("Valid R: ", len(valid_r))
print("Valid O: ", len(valid_o))

# 2-  1. / 255 neler yapıyor bakalım?

img_path = "src/DATASET/TRAIN/O/O_1.jpg"

img = cv2.imread(img_path, cv2.IMREAD_COLOR)
cv2.imshow("deneme", img)
cv2.waitKey(0)

img_scale = img / 255
cv2.imshow("deneme", img_scale)
cv2.waitKey(0)


#####################################
# Model Kurulumu
#####################################
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D

model = Sequential()

model.add(Conv2D(filters=16, kernel_size=(3, 3), input_shape=(224, 224, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=8, kernel_size=(3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=2, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


# Görselleştirelim
import visualkeras
from PIL import ImageFont
visualkeras.layered_view(model, legend=True).show()


# Overfit engeller.
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)

#checkpoint = ModelCheckpoint("my_best_model.h5",monitor='val_loss',verbose=1,mode='min',save_best_only=True,save_weights_only=False,period=1)
#checkpoint = ModelCheckpoint(filepath=filepath, monitor="val_loss", mode="min", save_best_only=True, verbose=1)

###################################################
# Train
####################################################

history = model.fit(train_batches,
                       epochs=10,
                       validation_data=valid_batches,
                       verbose=1,
                       callbacks=[early_stopping])


##################################################
# Sonuçlar / Test
##################################################
def get_history_plot(history, m1="accuracy", m2="val_accuracy"):
    plt.figure(figsize=(12, 5))
    plt.plot(history.history[m1], color="r")
    plt.plot(history.history[m2], color="b")
    plt.title(m1)
    plt.ylabel(m1)
    plt.xlabel("Epochs")
    plt.legend(["train", "val"])

get_history_plot(history)
get_history_plot(history, "loss", "val_loss")

model.evaluate(test_batches)
model.save('saved_model/my_model')
#pickle


new_model = tf.keras.models.load_model('saved_model/my_model')
new_model.evaluate(test_batches)


def predict_img(path, model):
    img = load_img(path, target_size=(224, 224))
    img = img_to_array(img)
    img = img / 255
    imshow(img)
    img = np.expand_dims(img, axis=0)
    pred = model.predict_proba(img)
    pred = np.argmax(pred)
    if pred == 0:
        print("Organic")
    else:
        print("Recyclable")


predict_img('src/DATASET/TEST/O/O_13907.jpg', new_model)
predict_img('src/DATASET/TEST/R/R_11002.jpg', new_model)

###################################################################################
# BOLUM 2 / Transfer Learning
###################################################################################
from tensorflow.keras.applications.vgg19 import VGG19
vgg = VGG19(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
vgg.summary()
visualkeras.layered_view(vgg, legend=True).show()

vgg2 = VGG19(input_shape=(224, 224, 3), weights='imagenet')
vgg2.summary()
visualkeras.layered_view(vgg2, legend=True).show()



for layer in vgg.layers:
    layer.trainable = False
vgg.summary()


tf_model = Sequential(layers=vgg.layers)
tf_model.add(Dropout(0.2))
tf_model.add(Flatten())
tf_model.add(Dense(512, activation='relu'))
tf_model.add(Dropout(0.2))
tf_model.add(Dense(256, activation='relu'))
tf_model.add(Dense(128, activation='relu'))
tf_model.add(Dense(2, activation='softmax'))
tf_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
tf_model.summary()


visualkeras.layered_view(tf_model, legend=True).show()

early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)


history_tf = tf_model.fit(train_batches,
                       epochs=10,
                       validation_data=valid_batches,
                       verbose=1,
                       shuffle=True,
                       callbacks=[early_stopping])

get_history_plot(history_tf)
get_history_plot(history_tf, "loss", "val_loss")

tf_model.evaluate(test_batches)
tf_model.save('saved_model/tf_model')

new_tf_model = tf.keras.models.load_model('saved_model/tf_model')
new_tf_model.evaluate(test_batches)


predict_img('src/DATASET/TEST/O/O_13907.jpg', new_tf_model)
predict_img('src/DATASET/TEST/R/R_11001.jpg', new_tf_model)

