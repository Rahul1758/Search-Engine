import os
import pickle
import streamlit as st
import tensorflow as tf
from tasks import Search_engine 
from tempfile import NamedTemporaryFile

# Load the CNN model
@st.cache()
def load_model(path):
    return tf.keras.models.load_model(path)
loaded_model = load_model('./cnn_model')

# Load the pickle files
image_features = pickle.load(open('Image_features.pkl','rb'))
vocab = pickle.load(open('Vocabulary.pkl','rb'))
embedding_vectors = pickle.load(open('Word_embeddings.pkl','rb'))

def main():

    html_temp = """<div style="background-color:tomato;padding:10px">
                    <h2 style="color:white;text-align:center;">Search Engine App</h2>
                    </div>"""
    st.markdown(html_temp,unsafe_allow_html=True)
    page=st.sidebar.selectbox('Tasks',['Find similar Objects','Find similar words', 'Generate tags', 'Tags to images'])
    se = Search_engine()
    image_dir = './images'
    image_paths = [image_dir+'/'+file for file in os.listdir(image_dir)]
    
    if page == 'Find similar Objects':
        st.title('Find similar Objects')
        st.write("The similar images section currently supports only the following Men's fashion apparels: \n\
                  **Backpacks, Belts, Deodorants, Jackets, Shirts, Shoes, Socks, Sunglasses, Wallet, Watch.** \n\
                  So please upload images accordingly.")
        file = st.file_uploader('Upload image', type=['png','jpeg','jpg'])
        temp_file = NamedTemporaryFile(delete=False)
        if file is not None:
            temp_file.write(file.getvalue())
            if st.button('Find'):
                se.find_similar_images(temp_file.name, image_features, 5, image_paths, loaded_model)
    
    if page == 'Find similar words':
        st.title('Find similar words')
        input_word = st.text_input('Enter a word')
        if st.button('Find'):
            se.find_similar_words(input_word, vocab, embedding_vectors, 5)

    if page == 'Generate tags':
        st.title('Generate tags for an image')
        file = st.file_uploader('Upload image', type=['png','jpeg','jpg'])
        temp_file = NamedTemporaryFile(delete=False)
        if file is not None:
            temp_file.write(file.getvalue())
            if st.button('Generate'):
                se.generate_tags(temp_file.name, vocab, embedding_vectors, 5, loaded_model)
    
    if page == 'Tags to images':
        st.title('Get related images to input word')
        st.write("The Tags to images section contains images of following Men's fashion apparels: \n\
                  **Backpacks, Belts, Deodorants, Jackets, Shirts, Shoes, Socks, Sunglasses, Wallet, Watch.**\n\
                  So please search for tags related to these apparels only.")
        input_word = st.text_input('Enter a word') 
        if st.button('Find'):
            se.tag_to_images(input_word, image_features, image_paths, vocab, embedding_vectors, 5, loaded_model)       

if __name__=='__main__':
    main()