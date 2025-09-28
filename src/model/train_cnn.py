from models import cnn_vae_model
import tensorflow as tf
from tensorflow.keras import mixed_precision
import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split 
from utils.metrics import _reconstruction_loss_cnn, _kl_loss
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard
import json

import gc
gc.collect()
tf.keras.backend.clear_session()

mixed_precision.set_global_policy('float32')

LATENT_SPACE_DIM = 32

LEARNING_RATE_CNN = 0.01
  
RECONSTRUCTION_LOSS_WEIGHT_CNN = 100
KL_LOSS_WEIGHT_CNN = 1
BATCH_SIZE_CNN = 1
EPOCHS_CNN = 500
SAVE_FOLDER_CNN = 'deepGame4Music\\models\\CNN_VAE_model\\test_13'

DATASET_CNN = 'data\\IMG\\dataset\\procceced_frames.npy'

# clear memory ========================================================
class ClearMemory(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        tf.keras.backend.clear_session()
        gc.collect()
        # to free memory of TF
        tf.compat.v1.reset_default_graph()
        
# KL Annealing ========================================================        
class KLAnnealingCallback(tf.keras.callbacks.Callback):
    def __init__(self, final_weight=1.0, total_epochs=50, start_weight=0.0):
         super(KLAnnealingCallback, self).__init__()
         self.final_weight = final_weight
         self.total_epochs = total_epochs
         self.start_weight = start_weight

    def on_epoch_begin(self, epoch, logs=None):
         # Calcular el nuevo peso: aumenta linealmente de start_weight hasta final_weight
         new_weight = self.start_weight + (self.final_weight - self.start_weight) * min(1.0, epoch / self.total_epochs)
         # Acceder a la capa de pérdida por su nombre ("vae_loss_layer")
         vae_loss_layer = self.model.get_layer("vae_loss_layer")
         vae_loss_layer.kl_loss_weight_var.assign(new_weight)
         print(f"Epoch {epoch+1}: KL loss weight annealed to {new_weight:.4f}")

# Train model ========================================================
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
        self.save()

    def summary(self):        
        self.vae.encoder.summary()
        self.vae.decoder.summary()
        self.vae.model.summary()  
              
    def _compile(self):
        #optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.learning_rate) 
        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_CNN, clipnorm=1.0)
        
        # # Métrica de reconstrucción (segunda salida del modelo)
        # def _reconstruction_metric(y_true, y_pred):
        #     return y_pred[1]  # Índice 1: reconstruction_loss

        # # Métrica de KL loss (tercera salida del modelo)
        # def _kl_loss_metric(y_true, y_pred):
        #     return y_pred[2]  # Índice 2: kl_loss
        
        # def _reconstruction_metric(y_true, y_pred):
        #     _reconstruction_metric = _reconstruction_loss_cnn(y_true, y_pred)
        #     return _reconstruction_metric
        
        def _kl_loss_metric(y_true, y_pred):
            _, mu, log_var = self.vae.encoder(y_true)
            return _kl_loss(mu, log_var)
        
        # Asignar nombres a las métricas
        #_reconstruction_metric.__name__ = "_reconstruction_loss"
        _kl_loss_metric.__name__ = "_kl_loss"    
                        
        self.vae.model.compile(
            optimizer=optimizer,
            loss=None,
            metrics = [_reconstruction_loss_cnn, _kl_loss_metric]
        ) 
        
    def train(self):
        callbacks = [
            EarlyStopping(monitor = 'val_loss', patience=7, restore_best_weights=True), 
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
            KLAnnealingCallback(final_weight=KL_LOSS_WEIGHT_CNN, total_epochs=70, start_weight=0.0),
            ClearMemory()
        ]
        
        self.history = self.vae.model.fit(
            self.train_data,
            self.train_data,
            batch_size=self.batch_size,
            epochs=self.epochs,
            shuffle=True,
            validation_data=(self.test_data, self.test_data),
            callbacks=callbacks
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
      
    # Cargar dataset ----------------------------------------------------      
    video_data = np.load(DATASET_CNN)
    video_data = tf.image.resize(video_data, (160, 120)).numpy()
    
    # add gaussian noise -----------------------------------------------
    video_data = np.clip(video_data + np.random.normal(0, 0.1, video_data.shape), 0, 1)
    video_data_shape = video_data.shape
    
    #Separar en train y test -------------------------------------------
    train_data, val_data = train_test_split(video_data, test_size=0.2, random_state=42)
    
    # Imprimir informacion del entrenamiento ----------------------------
    print("="*50)
    print('Dimension del dataset: ', video_data.shape)
    print('Dimension del train: ', train_data.shape)
    print('Dimension del test: ', val_data.shape)
    print('Dimension del espacio latente: ', LATENT_SPACE_DIM)
    print('Numero de epocas: ', EPOCHS_CNN)
    print('Learning rate: ', LEARNING_RATE_CNN)
    print('Peso de la perdida de reconstruccion: ', RECONSTRUCTION_LOSS_WEIGHT_CNN)
    print('Peso de la perdida KL: ', KL_LOSS_WEIGHT_CNN)
    print('Batch size: ', BATCH_SIZE_CNN)
    print("="*50)
    
    # ====================================================================
    # Limitar el uso de memoria de la GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    CNN_VAE = cnn_vae_model.CNN_VAE(input_shape=(video_data_shape[1],video_data_shape[2], video_data_shape[3]), 
                                    latent_space_dim=LATENT_SPACE_DIM, 
                                    reconstruction_loss_weights=RECONSTRUCTION_LOSS_WEIGHT_CNN, 
                                    kl_loss_weights=KL_LOSS_WEIGHT_CNN)

    
    Train_CNN_VAE = Train(CNN_VAE, 
                        train=train_data, 
                        test=val_data, 
                        epochs=EPOCHS_CNN, 
                        batch_size=BATCH_SIZE_CNN, 
                        learning_rate=LEARNING_RATE_CNN, 
                        save_path=SAVE_FOLDER_CNN)