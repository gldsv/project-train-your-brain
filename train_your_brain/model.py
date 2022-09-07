from transformers import TFAutoModel
import tensorflow as tf
import pandas as pd
import numpy as np
# from keras.models import load_weights

def initialize_model():

    backbone = TFAutoModel.from_pretrained("camembert-base")
    input_ids = tf.keras.layers.Input(shape=(512),dtype=tf.int32)
    attention_mask = tf.keras.layers.Input(shape=(512),dtype=tf.int32)
    x = backbone(**{'input_ids':input_ids,'attention_mask':attention_mask})[0]
    backbone.trainable=False
    outputs = tf.keras.layers.Dense(9,activation='softmax')(x)

    model = tf.keras.Model(inputs={'input_ids':input_ids,'attention_mask':attention_mask},outputs=outputs)

    print("\n ✅ model initialized")

    return model


def compile_model(model):
    adam = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=adam)

    print("\n ✅ model compiled")

    return model


def load_weights(model):
    model.load_weights("./model/model_weights.h5")

    print("\n ✅ weights loaded")

    return model

def pred(X_pred, date_pred):
    model = initialize_model()
    model = compile_model(model)
    model = load_weights(model)

    input_ids = np.array([np.array(i) for i in X_pred["ids"].values])
    attention_mask = np.array([np.array(i) for i in X_pred["mask"].values])

    y_pred = model.predict({"input_ids":input_ids,"attention_mask":attention_mask})

    print("\n ⭐️ pred done")

    with open(f'./model/pred_{date_pred}.npy', 'wb') as f:
        np.save(f, pred)

    return y_pred
