import os
import pickle
import base64
import streamlit as st
import numpy as np
import tensorflow as tf

def feature_extraction(paths, model):
    features_list = list()
    for i in paths:
        img = tf.keras.preprocessing.image.load_img(i, target_size=(299,299))
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = tf.expand_dims(img, axis=0)
        img_preprocessed = tf.keras.applications.inception_v3.preprocess_input(img)
        feature = model.predict(img_preprocessed)
        features_list.append(feature)

    features_arr = tf.squeeze(np.array(features_list)).numpy()
    return features_arr

image_dir='./images'
image_paths = [image_dir+'/'+path for path in os.listdir(image_dir)]
loaded_model = tf.keras.models.load_model('./cnn_model')

extracted_features = feature_extraction(image_paths, loaded_model)
pkl = pickle.dumps(extracted_features)
b64 = base64.b64encode(pkl).decode()
href = f'<a href="data:file/output_model;base64,{b64}" download="Image_features.pkl">Download Trained Model .pkl File</a>'
st.markdown(href, unsafe_allow_html=True)
