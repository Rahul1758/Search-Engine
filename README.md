# Search-Engine
## Table of Content
  * [Demo](#demo)
  * [Objective](#objective)
  * [Inspiration](#inspiration)
  * [Approach](#approach)
  * [Data](#data)
  * [Libraries](#libraries)
  * [Installation](#installation)
  * [To Do](#to-do)
  * [Contact](#contact)

## Demo
Link: [https://share.streamlit.io/rahul1758/search-engine/main/app.py](https://share.streamlit.io/rahul1758/search-engine/main/app.py)
### Find Similar Images
![https://github.com/Rahul1758/Search-Engine/tree/main/gifs%20%26%20imgs/Similar_images.gif]()
### Find Similar Words
![https://github.com/Rahul1758/Search-Engine/tree/main/gifs%20%26%20imgs/Similar_words.gif]()
### Generate Tags for Image
![https://github.com/Rahul1758/Search-Engine/tree/main/gifs%20%26%20imgs/Tag_generation.gif]()
### Find Related Images
![https://github.com/Rahul1758/Search-Engine/tree/main/gifs%20%26%20imgs/Related_images.gif]()

## Objective
The Objective of this Project was to develop a Search Engine, which can be integrated with E-commerce sites for Small Vendors. These additional features could potentially **increase the Sale** on Vendor's website by helping the Customer find their desired product **easily & quickly**. The ultimate aim to save the Customer's valuable time and improve their Shopping experience.

## Inspiration
We all use Google on daily basis and Google's Search Engine is one of the most powerful out there. This project draws inspiration from Google's Search Engine features such as:
  * Reverse image search
  * Similar words suggestion when you search for meaning for a word
  * If you reverse search an image, Google also predicts tag for that particular image.
  * Related images for any search term parameters

## Approach
There many ways to find similarity between 2 vectors but I'll be using **Cosine similarity** to compare 2 vectors. And for that the 2 vectors need to be vectors of same shape.

📖 : https://www.geeksforgeeks.org/cosine-similarity/

My approach towards the above mentioned tasks are as follows:

1. Build a CNN model using pretrained model (**Inception_V3**) which extracts features from images. Then I'll compare the image feature vectors to find the similar ones to the input image using Cosine similarity. Here I'll also be passing image names as target labels so the model outputs feature vectors close to the word embedding of target labels (image names). **Glove embeddings**(50d vectors) will be used to represent the target labels.

📖 : https://nlp.stanford.edu/projects/glove/
2. Using the Glove embeddings I'll create an embedding matrix which contains embeddings for all the words in vocabulary (~400k words). Then I'll compare input word embedding with each word in embedding matrix and output the most similar words.
3. Using our previously trained CNN model, we'll obtain an output feature vector (of same shape as word embeddings) which then can be compared to each word embedding in embedding matrix and be used to output the most similar word embeddings to the image feature vector.
4. This is reverse task of 3. wherein we have to compare input word embedding with the image feature vector and find the similar images.

## Data 
I prepared custom dataset consisting of 20 images for each of the following Men's Fashion products: **Backpacks, Belts, Deodorants, Jackets, Shirts, Shoes, Socks, Sunglasses, Wallet, Watch**. The dataset was collected by Web-Scraping from **Flipkart, Amazon & Google Images**.

## Libraries
* Numpy
* Pillow
* WordNinja
* Streamlit
* TensorFlow

## Installation
The Code is written in Python 3.8. If you don't have Python installed you can find it [here](https://www.python.org/downloads/). If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip. To install the required packages and libraries, run this command in the project directory after [cloning](https://www.howtogeek.com/451360/how-to-clone-a-github-repository/) the repository:
```bash
pip install -r requirements.txt
```
Then run the following command which runs the Webapp locally:
```
streamlit run app.py
```
That's it!!

## To Do
* Enlarge the dataset by collecting more images related to specific domain it is being used for.
* Use higher dimension Embedding vectors (300D) which might improve the Word search & Tag generation, as higher dimesion vector capture better meaning of the words.

## Contact
If you have suggestions for improvement or any other query, you can reach me at following platforms:
  * [Linkedin](https://www.linkedin.com/in/rahul-menon-515702a7/)
