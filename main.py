import pandas as pd
import numpy as np

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
import keras
from tensorflow.keras import layers

dis_and_smpts = pd.read_csv('main.csv') # Download data from csv into DataFrame
dis_and_smpts.head()

dis_and_smpts_features = dis_and_smpts.copy()
dis_and_smpts_labels = dis_and_smpts_features.pop('label')

dis_and_smpts_features.pop('frequency')

inputs = {}
for name, column in dis_and_smpts_features.items():
  inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=tf.float32)

# Run through a normalisation layer
x = layers.Concatenate()(list(inputs.values()))
norm = layers.Normalization()
norm.adapt(np.array(dis_and_smpts[inputs.keys()]))
all_inputs = norm(x)

preprocessed_inputs = [all_inputs]

preprocessed_inputs_cat = layers.Concatenate()(preprocessed_inputs)

dis_and_smpts_preprocessing = tf.keras.Model(inputs, preprocessed_inputs_cat)

tf.keras.utils.plot_model(model = dis_and_smpts_preprocessing , rankdir="LR", dpi=72, show_shapes=True)

dis_and_smpts_features_dict = {name: np.array(value)
                         for name, value in dis_and_smpts_features.items()}

features_dict = {name:values[:1] for name, values in dis_and_smpts_features_dict.items()}
dis_and_smpts_preprocessing(features_dict)

def dis_and_smpts_model(preprocessing_head, inputs):
  body = tf.keras.Sequential([
    layers.Dense(64),
    layers.Dense(1)
  ])

  preprocessed_inputs = preprocessing_head(inputs)
  result = body(preprocessed_inputs)
  model = tf.keras.Model(inputs, result)

  model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam())
  return model

dis_and_smpts_model = dis_and_smpts_model(dis_and_smpts_preprocessing, inputs)