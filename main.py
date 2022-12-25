import streamlit as st
import cv2
from tensorflow.keras.models import load_model
import numpy as np


st.title(""" Organik-Geri Dönüştürülebilir Obje Sınıflandırıcısı  """)
st.image("picture.png")
model = load_model('saved_model/my_model')

class_names = ["organic", "recylable"]

test_image = st.file_uploader("Resmi yükleyiniz...", type="jpg")
submit = st.button("Tahmin Et")


if submit:
    if test_image is not None:

        file_bytes = np.asarray(bytearray(test_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        opencv_image = cv2.resize(opencv_image, (224, 224))
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        st.image(opencv_image)
        opencv_image = opencv_image / 255
        opencv_image = np.expand_dims(opencv_image, axis=0)
        y_pred = model.predict(opencv_image)
        y_pred_ = np.argmax(y_pred)
        pred = class_names[y_pred_]
        print(y_pred)
        st.title(f"Tahminimce bu bir {pred}  ürün")
