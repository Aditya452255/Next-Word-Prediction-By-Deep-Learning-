import streamlit as st
import pickle
import numpy as np
import os
import json

try:
    import tflite_runtime.interpreter as tflite
except Exception:  # pragma: no cover - fallback when runtime isn't available
    import tensorflow.lite as tflite

DEFAULT_FILTERS = "!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n"


class SimpleTokenizer:
    def __init__(self, word_index, oov_token=None, filters=DEFAULT_FILTERS, lower=True, split=" "):
        self.word_index = word_index
        self.oov_token = oov_token
        self.filters = filters
        self.lower = lower
        self.split = split

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            if self.lower:
                text = text.lower()
            translate_table = str.maketrans({ch: self.split for ch in self.filters})
            text = text.translate(translate_table)
            tokens = [t for t in text.split(self.split) if t]
            seq = []
            for token in tokens:
                idx = self.word_index.get(token)
                if idx is None and self.oov_token:
                    idx = self.word_index.get(self.oov_token)
                if idx is not None:
                    seq.append(idx)
            sequences.append(seq)
        return sequences


def pad_sequence(sequence, maxlen):
    if len(sequence) > maxlen:
        sequence = sequence[-maxlen:]
    return [0] * (maxlen - len(sequence)) + list(sequence)


# ------------------------------
# Load saved files
# ------------------------------
@st.cache_resource
def load_resources():
    model_path = "model.tflite"
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "TFLite model not found. Please generate 'model.tflite' and place it in the app directory."
        )

    tokenizer_json = "tokenizer.json"
    if os.path.exists(tokenizer_json):
        with open(tokenizer_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        tokenizer = SimpleTokenizer(
            word_index=data["word_index"],
            oov_token=data.get("oov_token"),
            filters=data.get("filters", DEFAULT_FILTERS),
            lower=data.get("lower", True),
            split=data.get("split", " "),
        )
    else:
        try:
            with open("tokenizer.pkl", "rb") as f:
                tokenizer = pickle.load(f)
        except Exception as exc:
            raise FileNotFoundError(
                "Tokenizer JSON not found and tokenizer.pkl could not be loaded. "
                "Please export tokenizer.json and add it to the repo."
            ) from exc

    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)

    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return interpreter, input_details, output_details, tokenizer, max_len


interpreter, input_details, output_details, tokenizer, max_len = load_resources()

# ------------------------------
# Prediction function
# ------------------------------
def predict_next_word(text):
    sequence = tokenizer.texts_to_sequences([text])[0]
    padded = pad_sequence(sequence, max_len - 1)

    input_index = input_details[0]["index"]
    input_dtype = input_details[0]["dtype"]
    input_data = np.array([padded], dtype=input_dtype)

    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()

    output_index = output_details[0]["index"]
    preds = interpreter.get_tensor(output_index)
    predicted_index = int(np.argmax(preds))

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