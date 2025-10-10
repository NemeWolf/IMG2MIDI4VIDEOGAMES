import os
import json

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.Music_VAE import MusicVAE
from utils import _reconstruct_data_from_parquet

class VAETrainer:
    """
    Classe para gestionar el ciclo de vida del entrenamiento de MusicVAE.
    """
    def __init__(self, config, base_results_dir="results"):
        """
        Inicializa el entrenador con una configuración de hiperparámetros.

        Args:
            config (dict): Diccionario con los hiperparámetros para el run.
            base_results_dir (str): Directorio base para guardar los resultados.
        """
        self.config = config

        # Crear un nombre único para la carpeta de este experimento
        run_name = (
            f"run_ldim{config['model_params']['latent_dim']}"
            f"_kl{config['train_params']['kl_weight']}"
            f"_batch{config['train_params']['batch_size']}"
        )
        self.run_dir = os.path.join(base_results_dir, run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        print(f"Resultados para este run se guardarán en: {self.run_dir}")

        # Guardar la configuración del experimento
        with open(os.path.join(self.run_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=4)

    def load_data(self, train, dev):
        """Carga y prepara los datos de entrenamiento y validación."""
        print("Cargando y preparando datos...")

        # Extraer la columna 'piano_rolls' y convertirla a un array numpy de float32
        x_train = np.array(train['piano_rolls'].tolist(), dtype=np.float32)
        x_dev = np.array(dev['piano_rolls'].tolist(), dtype=np.float32)


        print(f"Datos de entrenamiento: {x_train.shape}")
        print(f"Datos de validación: {x_dev.shape}")
        return x_train, x_dev

    def train(self, x_train, x_dev):
        """
        Construye, compila y entrena el modelo MusicVAE.
        """
        print("Construyendo el modelo MusicVAE...")
        model_params = self.config['model_params'].copy()
        model_params['kl_weight'] = self.config['train_params']['kl_weight']
        model_builder = MusicVAE(**model_params)
        
        vae = model_builder.vae

        # --- Preparar datos para el modelo ---
        # El modelo VAE espera una lista de entradas, incluyendo los estados iniciales
        # Durante el entrenamiento, estos estados son siempre ceros.
        decoder_lstm_units = self.config['model_params']['decoder_lstm_units']

        # Datos de entrenamiento
        zeros_h_train = np.zeros((x_train.shape[0], decoder_lstm_units), dtype=np.float32)
        zeros_c_train = np.zeros((x_train.shape[0], decoder_lstm_units), dtype=np.float32)

        # Datos de validación
        zeros_h_dev = np.zeros((x_dev.shape[0], decoder_lstm_units), dtype=np.float32)
        zeros_c_dev = np.zeros((x_dev.shape[0], decoder_lstm_units), dtype=np.float32)

        # --- Callbacks ---
        # Guardar el mejor modelo basado en la pérdida de validación
        checkpoint_path = os.path.join(self.run_dir, 'best_model.weights.h5')
        model_checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        )

        # Parada temprana si el modelo deja de mejorar
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10, # Número de épocas sin mejora antes de parar
            verbose=1,
            restore_best_weights=True
        )

        print("\n--- Iniciando Entrenamiento ---")
        history = vae.fit(
            [x_train, zeros_h_train, zeros_c_train],
            x_train, # El objetivo es reconstruir la entrada
            validation_data=([x_dev, zeros_h_dev, zeros_c_dev], x_dev),
            epochs=self.config['train_params']['epochs'],
            batch_size=self.config['train_params']['batch_size'],
            callbacks=[model_checkpoint, early_stopping]
        )


        # --- Guardar historial de entrenamiento ---
        history_df = pd.DataFrame(history.history)
        history_path = os.path.join(self.run_dir, 'training_history.csv')
        history_df.to_csv(history_path, index=False)

        print(f"\nEntrenamiento completado. Mejor modelo guardado en '{checkpoint_path}'.")
        print(f"Historial de entrenamiento guardado en '{history_path}'.")

if __name__ == '__main__':
    # --- Directorios de Datos ---
    MODEL_DIR = '/home/neme/workspace/Model/src/model/models/Music_VAE_models/'
    
    # --- Procesar Datos ---
    
    # Cargar datos
    dataset_path = '/home/neme/workspace/Data/MIDI/final_datasets/estandar_reducido/'
            
    train_set_path = os.path.join(dataset_path, 'train_set_reduced.parquet')
    test_set_path = os.path.join(dataset_path, 'test_set_reduced.parquet')
    dev_set_path =  os.path.join(dataset_path, 'dev_set_reduced.parquet')

    train_set = pd.read_parquet(train_set_path)
    test_set = pd.read_parquet(test_set_path)
    dev_set = pd.read_parquet(dev_set_path)
    
    # Reconstruir datos
    train_set_reconstructed = _reconstruct_data_from_parquet(train_set)
    test_set_reconstructed = _reconstruct_data_from_parquet(test_set)
    dev_set_reconstructed = _reconstruct_data_from_parquet(dev_set)
    
    train_data = np.array(train_set_reconstructed['piano_rolls'].tolist())
    test_data = np.array(test_set_reconstructed['piano_rolls'].tolist())
    dev_data = np.array(dev_set_reconstructed['piano_rolls'].tolist())
    
    # -- Entrenamiento de Modelo --
    
    # Configuración del experimento
    
    config = {
    "model_params": {
                "features_dim": 84,
                "sequence_length": 16,
                "latent_dim": 256,
                "encoder_lstm_units": [256, 128, 512],
                "conductor_lstm_units": 256,
                "decoder_lstm_units": 512,
            },
            "train_params": {
                "kl_weight": 0.5,
                "epochs": 100,
                "batch_size": 128
            }
    }

    # Crear y ejecutar el entrenador
    trainer = VAETrainer(config=config, base_results_dir=MODEL_DIR)
    trainer.train(train_data, dev_data)

    # Cargar el historial de entrenamiento
    history_path = os.path.join(trainer.run_dir, 'training_history.csv')
    history_df = pd.read_csv(history_path)

    # Graficar la pérdida de entrenamiento y validación
    plt.figure(figsize=(12, 6))
    plt.plot(history_df['loss'], label='Training Loss')
    plt.plot(history_df['val_loss'], label='Validation Loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # --- Experimentos ---
    
    # experiments = [
    #     {
    #         "model_params": {
    #             "features_dim": 84,
    #             "sequence_length": 16,
    #             "latent_dim": 256,
    #             "encoder_lstm_units": [256, 128, 512],
    #             "conductor_lstm_units": 256,
    #             "decoder_lstm_units": 512,
    #         },
    #         "train_params": {
    #             "kl_weight": 0.5,
    #             "epochs": 100,
    #             "batch_size": 128
    #         }
    #     },
    # # --- Experimento 2: Espacio latente más pequeño ---
    #     {
    #         "model_params": {
    #             "features_dim": 84,
    #             "sequence_length": 16,
    #             "latent_dim": 128,  # <--- Cambio
    #             "encoder_lstm_units": [256, 128, 512],
    #             "conductor_lstm_units": 256,
    #             "decoder_lstm_units": 512,
    #         },
    #         "train_params": {
    #             "kl_weight": 0.5,
    #             "epochs": 100,
    #             "batch_size": 128
    #         }
    #     },
    # ]

    # # --- Bucle de Ejecución de Experimentos ---
    # for i, config in enumerate(experiments):
    #     print(f"\n\n--- INICIANDO EXPERIMENTO {i+1}/{len(experiments)} ---")
    #     print(json.dumps(config, indent=2))
        
    #     trainer = VAETrainer(config=config)
        
    #     try:
    #         x_train, x_dev = trainer.load_data(TRAIN_SET_PATH, DEV_SET_PATH)
    #         trainer.train(x_train, x_dev)
    #     except FileNotFoundError:
    #         print("\nERROR: No se encontraron los archivos del dataset.")
    #         print("Asegúrate de haber ejecutado 'processing_data.py' primero y de que las rutas son correctas.")
    #         break