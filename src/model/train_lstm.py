from models import lstm_vae_model
from utils.metrics import _reconstruction_loss_lstm, _kl_loss, harmonic_progressions_to_midi
from tensorflow.keras.utils import register_keras_serializable

import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping 
from tensorflow.keras import mixed_precision

import os
import pickle
import json

import numpy as np
from sklearn.model_selection import train_test_split


import gc
gc.collect()
tf.keras.backend.clear_session()

mixed_precision.set_global_policy('float32')

# PARAMETERS ========================================================            
LATENT_SPACE_DIM = 32

LEARNING_RATE_LSTM = 0.001

RECONSTRUCTION_LOSS_WEIGHT_LSTM = 3
KL_LOSS_WEIGHT_LSTM = 1

BATCH_SIZE_LSTM = 128
EPOCHS_LSTM = 500

# SAVE FOLDER ========================================================
SAVE_FOLDER_LSTM = 'deepGame4Music\\models\\LSTM_VAE_model\\test_07'

# DATASET PATHS ========================================================
DATASET_LSTM01 = 'data\\MIDI\\dataset\\By_genre\\progresions_by_genre.npy'
DATASET_LSTM02 = 'data\\MIDI\\dataset\\By_mode\\progresions_by_mode.npy'

# TRAIN CLASS ========================================================
class Train():
    def __init__(self, model, train, test, epochs=10, batch_size=32, learning_rate=0.001, save_path="models"):
        self.vae = model
        self.train_data = train
        self.test_data = test
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.save_path = save_path
        self.history = None
        self.call()
    
    def call(self):
        self.summary()
        self._compile()
        self.train()
        self._compile()
        self.save()
            
    def summary(self):        
        self.vae.encoder.summary()
        self.vae.decoder.summary()
        self.vae.model.summary()  
              
    def _compile(self):
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.learning_rate) 

        def _kl_loss_metric(y_true, y_pred):
            _, mu, log_var = self.vae.encoder(y_true)
            return _kl_loss(mu, log_var)
        
        
        _kl_loss_metric.__name__ = '_kl_loss'
                
        self.vae.model.compile(
            optimizer=optimizer,
            loss=None,
            metrics = [_reconstruction_loss_lstm, _kl_loss_metric]
        )
 
    def train(self):
        callbacks = [
            EarlyStopping(monitor = 'val_loss', patience=30, restore_best_weights=True), 
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=15, verbose=1) 
        ]
        self.history = self.vae.model.fit(
            self.train_data,
            self.train_data,
            batch_size=self.batch_size,
            epochs=self.epochs,
            shuffle=True,
            validation_data=(self.test_data, self.test_data),
            callbacks=callbacks,
        )
        print("Training finished")
        
    # Save model ========================================================
    def save(self):        
        self._save_parameters() 
        self._save_model()
        self._save_history()
        print("Model saved")
        
    def _save_parameters(self):
        parameters = [self.vae.input_shape, self.vae.latent_space_dim, self.vae.reconstruction_loss_weights, self.vae.kl_loss_weights, self.learning_rate]
        parameters_path = os.path.join(self.save_path, "parameters_vae.pkl")
        with open(parameters_path, "wb") as f:
            pickle.dump(parameters, f) 
    
    def _save_model(self):
        model_path = os.path.join(self.save_path, "vae_model")
        encoder_path = os.path.join(self.save_path, "encoder_model")
        decoder_path = os.path.join(self.save_path, "decoder_model")
        
        self.vae.model.save(model_path, save_format='tf', save_traces=True)
        self.vae.encoder.save(encoder_path, save_format='tf', save_traces=True)
        self.vae.decoder.save(decoder_path, save_format='tf', save_traces=True)
    
    def _save_history(self):
        history_path = os.path.join(self.save_path, 'training_history.json')
        history_dict = {key: [float(val) for val in values] for key, values in self.history.history.items()}
        with open(history_path, 'w') as f:
            json.dump(history_dict, f)
    
if __name__ == '__main__': 
    
    # preprocess data -----------------------------------------------------------------------------------------------
    progressions01 = np.load(DATASET_LSTM01)
    progressions02 = np.load(DATASET_LSTM02)    
    progressions = np.concatenate((progressions01, progressions02), axis=0)
    
    # normalize time duration with min-max scaler -------------------------------------------------------------------     
    max_duration = np.max(progressions[:, :, -1])
    min_duration = np.min(progressions[:, :, -1])    
    progressions[:, :, -1] = (progressions[:, :, -1] - min_duration) / (max_duration - min_duration)
    
    # select only the first 8 chords and keep the time duration ---------------------------------------------------
    # progressions = progressions[:, :8, :]
    
    # select only the first 8 chords and remove the time duration ---------------------------------------------------
    # progressions = progressions[:, :8, :-1]
    
    # select only the first 4 chords and keep the time duration ---------------------------------------------------
    progressions = progressions[:, :4, :]
    
    # select only the first 4 chords and remove the time duration ---------------------------------------------------    
    # progressions = progressions[:, :4, :-1]
    
    # capture the shape of the dataset --------------------------------------------------------------------------------
    progressions_shape = progressions.shape 
    
    # split data -----------------------------------------------------------------------------------------------------     
    train_data, val_data = train_test_split(progressions, test_size=0.2, random_state=42)

    # save train data --------------------------------------------------------------------------------------------
    train_data_path = os.path.join(SAVE_FOLDER_LSTM, 'train_data_lstm_01.npy')
    val_data_path = os.path.join(SAVE_FOLDER_LSTM, 'val_data_lstm_01.npy')
    np.save(train_data_path, train_data)
    np.save(val_data_path, val_data)

    #Imprimir informacion del entrenamiento
    print("="*50)
    print('Dimension del dataset: ', progressions_shape)
    print('Dimension del train_set: ', train_data.shape)
    print('Dimension del val_set: ', val_data.shape)
    print('Dimension del espacio latente: ', LATENT_SPACE_DIM)
    print('Numero de epocas: ', EPOCHS_LSTM)
    print('Learning rate: ', LEARNING_RATE_LSTM)
    print('Peso de la perdida de reconstruccion: ', RECONSTRUCTION_LOSS_WEIGHT_LSTM)
    print('Peso de la perdida KL: ', KL_LOSS_WEIGHT_LSTM)
    print('Batch size: ', BATCH_SIZE_LSTM)
    print("="*50)

    # #Train model
    LSTM_VAE = lstm_vae_model.LSTM_VAE(input_shape=(progressions_shape[1],progressions_shape[2]), latent_space_dim=LATENT_SPACE_DIM, reconstruction_loss_weights=RECONSTRUCTION_LOSS_WEIGHT_LSTM, kl_loss_weights=KL_LOSS_WEIGHT_LSTM)

    Train_LSTM_VAE = Train(LSTM_VAE, 
                        train=train_data, 
                        test=val_data, 
                        epochs=EPOCHS_LSTM, 
                        batch_size=BATCH_SIZE_LSTM, 
                        learning_rate=LEARNING_RATE_LSTM, 
                        save_path=SAVE_FOLDER_LSTM)        