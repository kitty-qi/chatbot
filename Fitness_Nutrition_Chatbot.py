import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlTemplates import bot_template, user_template, css

#from transformers import pipeline
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from stop_words import get_stop_words
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

import numpy as np
import tensorflow as tf

import csv
# importing matplotlib modules
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import math

import os
import streamlit as st
ai_key= st.secrets["openai_api_key"]

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as img

import tensorflow
from tensorflow import keras

from keras.applications.inception_v3 import preprocess_input
from keras.models import load_model
from shutil import copy
from shutil import copytree, rmtree
import tensorflow.keras.backend
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras import regularizers
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow import keras
from tensorflow.keras import models

# initialize the session state in a Streamlit application
def initialize_session_state():
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def get_pdf_text(pdf_files):
    
    text = ""
    for pdf_file in pdf_files:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text()
    return text


# https://github.com/Alir3z4/python-stop-words
# https://github.com/Alir3z4/stop-words/tree/master
def preprocess_text(text):
    if isinstance(text, float):
        return ''

    stop_words = get_stop_words('en')
    lemmatizer = WordNetLemmatizer()
    
    # Lowercase the text
    text = text.lower()

    # Define a dictionary for unit unifications
    unit_unifications = {
        "cal": "calorie",
        "kcal": "1000 calorie",
        "g": "gram",
        "mg": "milligram",
        "mcg": "microgram",
        "µg": "microgram",
        "iu": "International Unit",
        "ml": "milliliter",
        "oz": "ounce",
        "lb": "pound",
        "mmol": "millimole",
        "µmol": "micromole",
        "cm": "centimeter",
        "kg": "kilogram",
        "L": "liter",
        "mL": "milliliter",
        "mm": "millimeter",
        "µg/mL": "microgram per milliliter",
        "mg/dL": "milligram per deciliter"
    }

    # Remove extra whitespaces (including newline characters and tabs) and keep a single space
    text = re.sub(r'\s+', ' ', text)

    # Remove punctuation and keep digit numbers
    text = re.sub(r'[^\w\s\d]', '', text)
    
    # Tokenize the text
    words = word_tokenize(text)

    # Lemmatize verbs and nouns
    words = [lemmatizer.lemmatize(word, 'v') for word in words]  # 'v' stands for verb.
    words = [lemmatizer.lemmatize(word, 'n') for word in words]  # 'n' stands for noun.
    
    # Words to exclude
    words_to_exclude = ["10", "39", "per"]

    # Create a modified stop words list by removing specific words
    modified_stop_words = [word for word in stop_words if word not in words_to_exclude]
    

    # Remove stop words
    words = [word for word in words if word not in modified_stop_words]
    
    # Join the words back into a text string
    text = ' '.join(words)
    
    return text



def get_chunk_text(text):
    
    text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap = 200,
    length_function = len
    )

    chunks = text_splitter.split_text(text)

    return chunks


def get_vector_store(text_chunks):
    
    # For OpenAI Embeddings   
    embeddings = OpenAIEmbeddings(openai_api_key= ai_key)

    vectorstore = FAISS.from_texts(texts = text_chunks, embedding = embeddings)
    
    return vectorstore


def get_conversation_chain(vector_store, max_tokens = None):
    if vector_store is None:
        st.warning("Vector store is not initialized.")
        return None

    # llm = ChatOpenAI(model = 'gpt-4o', openai_api_key= ai_key)
    # OpenAI Model
    if max_tokens is None:
        max_tokens = 150  # Set a default limit if none provided
    llm = ChatOpenAI(model='gpt-4o', openai_api_key=ai_key, max_tokens=max_tokens)

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )

    return conversation_chain

def load_nutrition_table():
    nutrition_table = {}
    with open('./data/nutrition101.csv', 'r') as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            if i == 0:
                continue
            name = row[1].strip()
            nutrition_table[name] = [
                {'name': 'protein', 'value': float(row[2])},
                {'name': 'calcium', 'value': float(row[3])},
                {'name': 'fat', 'value': float(row[4])},
                {'name': 'carbohydrates', 'value': float(row[5])},
                {'name': 'vitamins', 'value': float(row[6])}
            ]
    return nutrition_table

def predict_class(model, images, foods_sorted):
    predicted_labels = []
    for img_path in images:
        img = image.load_img(img_path, target_size=(200, 200))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255.

        pred = model.predict(img)
        index = np.argmax(pred)
        foods_sorted.sort()
        pred_value = foods_sorted[index]
        predicted_labels.append(pred_value)

    return predicted_labels


def get_nutritional_info(predicted_labels, nutrition_table):
    nutritional_info_list = []

    for label in predicted_labels:
        if label in nutrition_table:
            nutrition_info_dict = {nutrient['name']: nutrient['value'] for nutrient in nutrition_table[label]}
            nutritional_info_list.append({'label': label, 'nutrition_info': nutrition_info_dict})
        else:
            nutritional_info_list.append({'label': label, 'nutrition_info': None})

    return nutritional_info_list

def handle_user_input(question):
    if st.session_state.conversation:
        print("Before conversation call:", st.session_state.conversation)
        #response = st.session_state.conversation({'question': question})
        response = st.session_state.conversation({'question': question})
        print("After conversation call:", st.session_state.conversation)
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    else:
        st.warning("Conversation not initialized.")




def main():
    load_dotenv()

    st.set_page_config(page_title='Chat with Your own PDFs', page_icon=':books:')

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # Initialize conversation chain
    if not st.session_state.conversation:
        vector_store = None  # You can set a default value or handle it as per your requirements
        st.session_state.conversation = get_conversation_chain(vector_store, max_tokens=1000)

    # define label meaning
    foods_sorted = ['apple pie', 'baby back ribs', 'baklava', 'beef carpaccio', 'beef tartare', 'beet salad', 'beignets', 'bibimbap',
             'bread pudding', 'breakfast burrito', 'bruschetta', 'caesar salad', 'cannoli', 'caprese salad', 'carrot cake',
             'ceviche', 'cheese plate', 'cheesecake', 'chicken curry', 'chicken quesadilla', 'chicken wings', 'chocolate cake',
             'chocolate mousse', 'churros', 'clam chowder', 'club sandwich', 'crab cakes', 'creme brulee', 'croque madame',
             'cup cakes', 'deviled eggs', 'donuts', 'dumplings', 'edamame', 'eggs benedict', 'escargots', 'falafel',
             'filet mignon', 'fish and_chips', 'foie gras', 'french fries', 'french onion soup', 'french toast',
             'fried calamari', 'fried rice', 'frozen yogurt', 'garlic bread', 'gnocchi', 'greek salad',
             'grilled cheese sandwich', 'grilled salmon', 'guacamole', 'gyoza', 'hamburger', 'hot and sour soup', 'hot dog',
             'huevos rancheros', 'hummus', 'ice cream', 'lasagna', 'lobster bisque', 'lobster roll sandwich',
             'macaroni and cheese', 'macarons', 'miso soup', 'mussels', 'nachos', 'omelette', 'onion rings', 'oysters',
             'pad thai', 'paella', 'pancakes', 'panna cotta', 'peking duck', 'pho', 'pizza', 'pork chop', 'poutine',
             'prime rib', 'pulled pork sandwich', 'ramen', 'ravioli', 'red velvet cake', 'risotto', 'samosa', 'sashimi',
             'scallops', 'seaweed salad', 'shrimp and grits', 'spaghetti bolognese', 'spaghetti carbonara', 'spring rolls',
             'steak', 'strawberry shortcake', 'sushi', 'tacos', 'octopus balls', 'tiramisu', 'tuna tartare', 'waffles']

    nu_link = 'https://www.nutritionix.com/food/'

    nutrition_table = load_nutrition_table()

    # Loading the best saved model to make predictions.
    tensorflow.keras.backend.clear_session()
    # model from https://github.com/MaharshSuryawala/Food-Image-Recognition (Copyright (c) 2020 Maharsh Suryawala)
    model_best = load_model('best_model_101class.hdf5', compile=False)

    st.header('Chat with Your own PDFs :books:')
    question = st.text_input("Ask anything to your PDF: ")

    with st.sidebar:

        st.subheader("Upload your Documents Here: ")
        pdf_files = st.file_uploader("Choose your PDF Files and Press OK", type=['pdf'], accept_multiple_files=True)

        if st.button("OK"):
            with st.spinner("Processing your PDFs..."):

                # Get PDF Text
                raw_text = get_pdf_text(pdf_files)

                # Clean Text
                clean_text = preprocess_text(raw_text)

                # Get Text Chunks
                text_chunks = get_chunk_text(clean_text)

                # Create Vector Store
                vector_store = get_vector_store(text_chunks)
                st.write("DONE")

                # Assign vector_store to conversation
                st.session_state.conversation = get_conversation_chain(vector_store, max_tokens=1000)

    # Image Upload Sidebar
    with st.sidebar:
        st.subheader("Upload your Images Here: ")
        image_files = st.file_uploader("Choose your Image Files", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)

        if st.button("Image OK"):
            if image_files:
                with st.spinner("Processing your Images..."):

                    # Predict labels for the uploaded images
                    predicted_labels = predict_class(model_best, [img.name for img in image_files], foods_sorted)

                    # Fetch nutritional information for each predicted label
                    nutritional_info_list = get_nutritional_info(predicted_labels, nutrition_table)

                    # Display nutritional information
                    for item in nutritional_info_list:
                        label = item['label']
                        nutrition_info = item['nutrition_info']
                        if nutrition_info:
                            st.success(f'Nutritional information for {label}: {nutrition_info}')
                            question = f"Nutritional information for {label}: {nutrition_info}"
                        else:
                            st.warning(f'Nutritional information not found for {label}')
    
    if question:
        handle_user_input(question)

# Initialize session state
initialize_session_state()

if __name__ == '__main__':
    main()

