import os
import json

# Suprimir logs de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=INFO, 1=WARNING, 2=ERROR, 3=FATAL
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suprimir optimizaciones OneDNN
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Solo usar GPU 0 si tienes múltiples

import tensorflow as tf

# Configurar TensorFlow para ser menos verboso
tf.get_logger().setLevel('ERROR')

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
  
# Importar paquetes personalizados
from models.Music_VAE import MusicVAE
from utils import reconstruct_data_from_parquet, augment_data_real
from utils.tensorboard_utils import create_tensorboard_manager
from utils.visualization_utils import create_training_visualizer

class VAETrainer:
    """
    Clase para gestionar el ciclo de vida del entrenamiento de MusicVAE.
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
            
        # Inicializar managers
        self.tensorboard_manager = create_tensorboard_manager(self.run_dir, self.config)
        self.visualizer = create_training_visualizer(self.run_dir)
        
    def load_data(self, train_path, dev_path):
        """Carga y prepara los datos de entrenamiento y validación."""
        print("Cargando y preparando datos...")

        # Cargar datos
        train = pd.read_parquet(train_path)
        dev = pd.read_parquet(dev_path)

        # Reconstruir datos
        train = reconstruct_data_from_parquet(train)
        dev = reconstruct_data_from_parquet(dev)

        # Extraer la columna 'piano_rolls' y convertirla a un array numpy de float32
        x_train = np.array(train['piano_rolls'].tolist(), dtype=np.float32)
        x_dev = np.array(dev['piano_rolls'].tolist(), dtype=np.float32)

        print(f"Datos de entrenamiento: {x_train.shape}")
        print(f"Datos de validación: {x_dev.shape}")

        return x_train, x_dev

    def _create_model_visualization(self):
        """Crea visualizaciones del modelo usando TensorBoardManager."""
        model_params = self.config['model_params'].copy()
        model_params['kl_weight'] = self.config['train_params']['kl_weight']
        model_builder = MusicVAE(**model_params)
        
        self.tensorboard_manager.create_model_visualization(model_builder)
    
    def train(self, x_train, x_dev):
        """
        Construye, compila y entrena el modelo MusicVAE.
        """
        print("Construyendo el modelo MusicVAE...")
        model_params = self.config['model_params'].copy()
        model_params['kl_weight'] = self.config['train_params']['kl_weight']
        model_builder = MusicVAE(**model_params)
        
        vae = model_builder.vae

        # ¿AUMENTAR DATOS REALMENTE?
        use_real_augmentation = self.config.get('use_real_augmentation', False)
        augmentation_factor = self.config.get('augmentation_factor', 2)
        
        if use_real_augmentation:
            x_train = augment_data_real(x_train, augmentation_factor)
        else:
            x_train = x_train
        
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
                
        tensorboard_callback = self.tensorboard_manager.create_callback(
            histogram_freq=1,           # Actualizar histogramas cada época
            write_graph=True,           # Escribir grafo
            write_images=False,         # No imágenes (evita problemas)
            profile_batch=0,            # Sin profiling (más rápido)
            embeddings_freq=0,          # Sin embeddings (más rápido)
            update_freq='batch'         # ACTUALIZAR CADA BATCH (tiempo real)
        )
        
        print("\n--- Iniciando Entrenamiento ---")
        print(f"TensorBoard logs: {os.path.join(self.run_dir, 'logs')}")
        print("Para visualizar: tensorboard --logdir=logs --port=6006")
        

        history = vae.fit(
            [x_train, zeros_h_train, zeros_c_train],
            x_train,
            validation_data=([x_dev, zeros_h_dev, zeros_c_dev], x_dev),
            epochs=self.config['train_params']['epochs'],
            batch_size=self.config['train_params']['batch_size'],
            callbacks=[model_checkpoint, early_stopping, tensorboard_callback]
        )

        # --- Guardar historial de entrenamiento ---
        history_df = pd.DataFrame(history.history)
        history_path = os.path.join(self.run_dir, 'training_history.csv')
        history_df.to_csv(history_path, index=False)

        print(f"\nEntrenamiento completado. Mejor modelo guardado en '{checkpoint_path}'.")
        print(f"Historial de entrenamiento guardado en '{history_path}'.")
        
    # --- Visualización del historial de entrenamiento ---
    
    def plot_training_history(self, save_plots=True, show_plots=True):
        """Visualiza el historial de entrenamiento usando TrainingVisualizer."""
        self.visualizer.plot_training_history(save_plots, show_plots)

    def launch_tensorboard(self, port=6006, auto_open=True):
        """Lanza TensorBoard usando TensorBoardManager."""
        return self.tensorboard_manager.launch_tensorboard(port, auto_open)

if __name__ == '__main__':
    # --- Directorios de Datos ---
    MODEL_DIR = '/home/neme/workspace/Model/src/model/models/Music_VAE_models/'
    
    # --- Procesar Datos ---
    
    # Cargar datos
    dataset_path = '/home/neme/workspace/Data/MIDI/final_datasets/estandar_reducido/'
            
    train_set_path = os.path.join(dataset_path, 'train_set_reduced.parquet')
    test_set_path = os.path.join(dataset_path, 'test_set_reduced.parquet')
    dev_set_path =  os.path.join(dataset_path, 'dev_set_reduced.parquet')
    
    # -- Entrenamiento de Modelo --
    
    # Configuración del experimento    
    config = {
    "model_params": {
        "features_dim": 84,
        "sequence_length": 16,
        "latent_dim": 96,  
        "encoder_lstm_units": [56, 28, 224],  
        "conductor_lstm_units": 112,  
        "decoder_lstm_units": 224,  
        "recurrent_dropout": 0.2,
        "dropout_rate": 0.35,  
        "l2_regularization": 0.00015  
    },
    "train_params": {
        "kl_weight": 0.6, 
        "epochs": 60,
        "batch_size": 512
    },
    "use_real_augmentation": False,
    "augmentation_factor": 0,
}

    # Crear y ejecutar el entrenador
    trainer = VAETrainer(config=config, base_results_dir=MODEL_DIR)
    
    # Lanzar TensorBoard ANTES del entrenamiento
    print("🚀 Lanzando TensorBoard...")
    tensorboard_process = trainer.launch_tensorboard(auto_open=False) 
        
    # Crear visualizaciones del modelo
    trainer._create_model_visualization()
    
    # Cargar datos
    train_data, dev_data = trainer.load_data(train_set_path, dev_set_path)
    
    # Iniciar el entrenamiento (TensorBoard ya está corriendo)
    print("\n🎯 Iniciando entrenamiento...")
    print("📈 Ve a http://localhost:6006 para monitorear en tiempo real")    
    trainer.train(train_data, dev_data)
    
    # Visualizar el historial de entrenamiento
    trainer.plot_training_history(save_plots=True, show_plots=True)
    
    # # Mantener TensorBoard corriendo
    # if tensorboard_process:
    #     print("\n✨ Entrenamiento completado!")
    #     print("📊 TensorBoard sigue corriendo para analizar resultados")
    #     print("❌ Presiona Ctrl+C para detener TensorBoard")
    #     try:
    #         tensorboard_process.wait()
    #     except KeyboardInterrupt:
    #         trainer.tensorboard_manager.stop_tensorboard(tensorboard_process)
    #         print("🛑 TensorBoard detenido")