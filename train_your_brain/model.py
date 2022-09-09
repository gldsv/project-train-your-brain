from transformers import TFAutoModel
import tensorflow as tf
import pandas as pd
import numpy as np
from google.cloud import storage
import os

weights_file_name = "model_weights.h5"

def initialize_model():

    backbone = TFAutoModel.from_pretrained("camembert-base")
    input_ids = tf.keras.layers.Input(shape=(512),dtype=tf.int32)
    attention_mask = tf.keras.layers.Input(shape=(512),dtype=tf.int32)
    x = backbone(**{'input_ids':input_ids,'attention_mask':attention_mask})[0]
    backbone.trainable=False
    outputs = tf.keras.layers.Dense(9,activation='softmax')(x)

    model = tf.keras.Model(inputs={'input_ids':input_ids,'attention_mask':attention_mask},outputs=outputs)

    print("✅ model initialized")

    return model


def compile_model(model):
    adam = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=adam)

    print("✅ model compiled")

    return model


def download_weights_from_cloud():
    bucket = os.environ.get("BUCKET")
    file_path = f"model_weights/{weights_file_name}"
    client = storage.Client()
    bucket = client.bucket(bucket)
    blob = bucket.blob(file_path)

    print("⚙️ Downloading weights from Cloud Storage bucket")

    blob.download_to_filename(f"./model/{weights_file_name}")

    print("✅ Weights downloaded from Cloud Storage bucket")


def load_weights(model):
    model.load_weights(f"./model/{weights_file_name}")

    print("✅ weights loaded")

    return model


def pred(X_pred, date_pred):
    model = initialize_model()
    model = compile_model(model)
    model = load_weights(model)

    input_ids = np.array([np.array(i) for i in X_pred["ids"].values])
    attention_mask = np.array([np.array(i) for i in X_pred["mask"].values])

    y_pred = model.predict({"input_ids":input_ids,"attention_mask":attention_mask})

    print("⭐️ pred done")

    with open(f'./model/pred_{date_pred}.npy', 'wb') as f:
        np.save(f, pred)

    return y_pred

if __name__ == "__main__":
    download_weights_from_cloud()
