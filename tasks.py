import wordninja
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

class Search_engine:
    def __init__(self):
        pass

    def image_feature_extraction(self, img_path, model):
        image=tf.keras.preprocessing.image.load_img(img_path, target_size=(299,299))
        image=tf.keras.preprocessing.image.img_to_array(image)
        image=tf.expand_dims(image, axis=0)
        final_img=tf.keras.applications.inception_v3.preprocess_input(image)
        feature=model.predict(final_img)
        
        return tf.squeeze(feature).numpy()

    def cosine_similarity(self, vec_1d, vec_2d):
        vec_2d_norm = np.linalg.norm(vec_2d, axis=1)
        vec_1d_norm = np.linalg.norm(vec_1d)
        
        return (vec_2d @ vec_1d)/(vec_2d_norm * vec_1d_norm)

    def get_word_embedding(self, word, vocabulary, vector_matrix):
        word_list = wordninja.split(word.lower())
        # Get embedding vector for input word
        if len(word_list)>1:
            vectors = np.array([vector_matrix[vocabulary.index(word)] for word in word_list])
            word_embed = np.mean(vectors, axis=0)
        else:
            word_embed = vector_matrix[vocabulary.index(word.lower())]
        return word_embed

    def find_similar_images(self, input_img_path, extracted_features, n_similar, img_paths, model):
        # Extracting input image features and calculating cosine similarities
        input_feature_vec = self.image_feature_extraction(input_img_path, model)
        similarity_array = self.cosine_similarity(input_feature_vec, extracted_features)
        
        # Finding indexes of most similar images
        top_similar_idx = np.argpartition(similarity_array, -n_similar)[-n_similar:]
        similar_imgs_paths = [img_paths[i] for i in top_similar_idx]

        st.write('### Input image: \n')
        input_img = Image.open(input_img_path)
        st.image(input_img)
        st.write('### Similar images: \n')
        for path in similar_imgs_paths:
            img = Image.open(path)
            st.image(img)
    
    def find_similar_words(self, word, vocabulary, vector_matrix, n_similar):
        
        input_embed = self.get_word_embedding(word, vocabulary, vector_matrix)
        similarity_array = self.cosine_similarity(input_embed, vector_matrix)
        
        # Finding indexes of most similar words
        top_similar_idx = np.argpartition(similarity_array, -n_similar)[-n_similar:]
        similar_words = [vocabulary[i] for i in top_similar_idx]
        similar_words = ', '.join(similar_words)
        
        st.write(f'### Input word: {word}\n')
        st.write(f'### Similar words: {similar_words}')

    def generate_tags(self, input_img_path, vocabulary, vector_matrix, n_similar, model):
        # Extracting input image features and calculating cosine similarities
        input_feature_vec = self.image_feature_extraction(input_img_path, model)
        similarity_array = self.cosine_similarity(input_feature_vec, vector_matrix)

        # Finding indexes of most similar words
        top_similar_idx = np.argpartition(similarity_array, -n_similar)[-n_similar:]
        similar_words = [vocabulary[i] for i in top_similar_idx]
        similar_words = ', '.join(similar_words)

        st.write('### Input image: \n')
        input_img = Image.open(input_img_path)
        st.image(input_img)
        st.write(f'### Tags for input image: {similar_words}')

    def tag_to_images(self, word, extracted_features, img_paths, vocabulary, vector_matrix, n_similar, model):
        
        input_embed = self.get_word_embedding(word, vocabulary, vector_matrix)
        similarity_array = self.cosine_similarity(input_embed, extracted_features)

        # Finding indexes of most similar words
        top_similar_idx = np.argpartition(similarity_array, -n_similar)[-n_similar:]
        similar_imgs_paths = [img_paths[i] for i in top_similar_idx]

        st.write(f'### Input word: {word} \n')
        st.write('### \n Similar images: \n')
        for path in similar_imgs_paths:
            img = Image.open(path)
            st.image(img)