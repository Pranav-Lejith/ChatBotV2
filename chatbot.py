import tensorflow as tf
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import json
import random
import pickle
import streamlit as st

# Download NLTK data files
nltk.download('punkt')
nltk.download('wordnet')

# Load the trained model
model = tf.keras.models.load_model('chatbot_model.h5')

# Load the words and classes
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"Found in bag: {w}")
    return np.array(bag)

def predict_class(sentence):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# Streamlit app
st.title("Chatbot")

# Sidebar with information
st.sidebar.title("About:")
st.sidebar.info("This is a simple chatbot built using TensorFlow and Streamlit.")
st.sidebar.info("Created by Pranav Lejith(Amphibiar)")

# Initialize conversation history
if 'history' not in st.session_state:
    st.session_state.history = ""

# Input box for the user
user_input = st.text_input("You:", "")

# Process user input and generate response
if user_input:
    ints = predict_class(user_input)
    response = get_response(ints, intents)
    st.session_state.history += f"You: {user_input}\nBot: {response}\n\n"
    st.text_area("Bot:", response, height=100, max_chars=None)

# Display conversation history
st.subheader("Conversation History")
st.text_area("History", st.session_state.history, height=400, max_chars=None)
