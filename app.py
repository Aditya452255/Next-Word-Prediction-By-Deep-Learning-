import streamlit as st
import pickle
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ------------------------------
# Load saved files
# ------------------------------
@st.cache_resource
def load_resources():
    model_path = "lstm_model.h5"
    if not os.path.exists(model_path):
        alt_path = "lstm_model (1).h5"
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            raise FileNotFoundError(
                "Model file not found. Expected 'lstm_model.h5' (or 'lstm_model (1).h5') in the app directory."
            )
    model = load_model(model_path, compile=False)
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)
    return model, tokenizer, max_len

model, tokenizer, max_len = load_resources()

# ------------------------------
# Prediction function
# ------------------------------
def predict_next_word(text):
    sequence = tokenizer.texts_to_sequences([text])[0]
    sequence = pad_sequences([sequence], maxlen=max_len-1, padding='pre')

    preds = model.predict(sequence, verbose=0)
    predicted_index = np.argmax(preds)

    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word
    return ""

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Next Word Prediction", layout="centered")

st.title("🧠 Next Word Prediction (LSTM)")
st.write("Enter a sentence and the model will predict the **next word**.")

user_input = st.text_input("✍️ Enter text:", placeholder="Type a sentence here...")

if st.button("Predict Next Word"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        next_word = predict_next_word(user_input)
        st.success(f"**Predicted Next Word:** {next_word}")

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.caption("LSTM-based Next Word Prediction using Streamlit")