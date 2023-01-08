import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import os
import random
import tensorflow as tf
from tensorflow.keras.layers import Dense, Bidirectional, GRU, Embedding, Input, GlobalMaxPooling1D
from tensorflow.keras.metrics import Recall, Precision
import numpy as np
import config
from algorithm.preprocess.preprocess import prep_TEXT

MODEL_NAME = config.MODEL_NAME
MODEL_SAVE_PATH = config.MODEL_SAVE_PATH

seed_value = config.RAND_SEED
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)


class RNN_pretrained_embed():
    def __init__(self):
        pass

    def __build_model_compile(self, num_y_classes, num_layers, neurons_num, embed_lay_output,learning_rate):
        """This fucntion builds the model and compile it"""
        # tensorflow hub universal sentence encoder
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        metrics = [Precision(),
                   Recall()
                   ]
        
        text_vectorizer = prep_TEXT.load_text_vectorizer()        
        max_vocab_length = len(text_vectorizer.get_vocabulary())
        max_length = len(tf.squeeze(text_vectorizer(["dsads"]))) #We defined output lenght during preprocessing, now getting it for embedding layer

        embed_layer = Embedding(input_dim=max_vocab_length, # set input shape
                             output_dim=embed_lay_output, # set size of embedding vector
                             input_length=max_length, # how long is each input
                             name="Embedding_Layer") 

        model = tf.keras.Sequential()
        model.add(Input(shape=[1,],dtype=tf.string))
        model.add(text_vectorizer)
        model.add(embed_layer)

        for i in range(num_layers):
            model.add(Bidirectional(GRU(neurons_num,return_sequences=True),
                        name=f"Bidirectional_layer_{i}"))

        model.add(GlobalMaxPooling1D())

        if num_y_classes > 2:
            model.add(Dense(num_y_classes, activation='softmax'))
            loss = tf.keras.losses.CategoricalCrossentropy()
            model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        else:                             
            model.add(Dense(1, activation='sigmoid'))
            loss = tf.keras.losses.BinaryCrossentropy()
            model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        return model

    def fit(self, x_train, y_train, x_val=None, y_val=None, 
            epochs=10,
            num_layers=1,
            neurons_num=50,
            embed_lay_output=120,
            learning_rate=1e-2,
            ):

        num_classes = len(np.unique(y_train))        

        self.model = self.__build_model_compile(
            num_y_classes=num_classes, num_layers=num_layers,
            neurons_num=neurons_num, embed_lay_output=embed_lay_output,
            learning_rate=learning_rate)
            
        if num_classes>2:
            y_train = tf.squeeze(tf.one_hot(y_train,num_classes))
            if not y_val is None:
                y_val = tf.squeeze(tf.one_hot(y_val,num_classes))
        
        callbacks = [] # we add desired call backs in a list # If no callback is desired, leave list empty
        # Kindly change patience as you see fit in your project
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="loss" if x_val is None else "val_loss",
                patience=2,
                verbose=1,
                restore_best_weights=True
                )
            ) 

        self.model.fit(
            x_train,
            y_train,
            epochs=epochs,
            validation_data=None if x_val is None else (x_val, y_val),
            validation_steps=None if x_val is None else len(x_val),
            callbacks=callbacks,
            verbose=1
        )

        return self.model

    def save_model(self, save_path=MODEL_SAVE_PATH):
        path = os.path.join(save_path, MODEL_NAME)
        self.model.save(path)


def load_model(save_path=MODEL_SAVE_PATH):
    path = os.path.join(save_path, MODEL_NAME)
    model = tf.keras.models.load_model(path)
    print(f"Loaded model from: {path} successfully")
    return model
