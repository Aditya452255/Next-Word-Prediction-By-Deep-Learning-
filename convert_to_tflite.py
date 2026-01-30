import json
import os
import pickle
import tensorflow as tf

MODEL_CANDIDATES = [
    "model.h5",
    "lstm_model.h5",
    "lstm_model (1).h5",
]


def find_model_path():
    for path in MODEL_CANDIDATES:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "No .h5 model found. Expected one of: {}".format(", ".join(MODEL_CANDIDATES))
    )


def convert_model_to_tflite(model_path: str, output_path: str = "model.tflite"):
    model = tf.keras.models.load_model(model_path, compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    print(f"Saved TFLite model to {output_path}")


def export_tokenizer_json(pkl_path: str = "tokenizer.pkl", json_path: str = "tokenizer.json"):
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Tokenizer pickle not found: {pkl_path}")

    with open(pkl_path, "rb") as f:
        tokenizer = pickle.load(f)

    data = {
        "word_index": tokenizer.word_index,
        "oov_token": getattr(tokenizer, "oov_token", None),
        "filters": getattr(tokenizer, "filters", None),
        "lower": getattr(tokenizer, "lower", True),
        "split": getattr(tokenizer, "split", " "),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    print(f"Saved tokenizer JSON to {json_path}")


if __name__ == "__main__":
    model_path = find_model_path()
    convert_model_to_tflite(model_path)
    export_tokenizer_json()
