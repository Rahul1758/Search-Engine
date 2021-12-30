import os
import pickle
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

image_dir='./Search-Engine/images'
image_paths = [image_dir+'/'+path for path in os.listdir(image_dir)]
loaded_model = tf.keras.models.load_model('./Search-Engine/cnn_model')

extracted_features = feature_extraction(image_paths, loaded_model)

with open('Image_features.pkl', 'wb') as f:
    pickle.dump(extracted_features, f)
