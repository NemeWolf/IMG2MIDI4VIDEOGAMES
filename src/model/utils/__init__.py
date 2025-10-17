# Reconstruir datos
import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from .tensorboard_utils import TensorBoardManager, create_tensorboard_manager
from .visualization_utils import TrainingVisualizer, create_training_visualizer

def reconstruct_data_from_parquet(df_raw: pd.DataFrame) -> pd.DataFrame:
    piano_rolls_reconstructed = []
    sequence_length = 16
    piano_roll_size = 84

    for idx, row in tqdm(df_raw.iterrows(), total=len(df_raw)):
        # Reconstruir piano roll
        piano_roll_flat = row['piano_rolls']
        piano_roll = np.array(piano_roll_flat).reshape(
            sequence_length, piano_roll_size
        )
        piano_rolls_reconstructed.append(piano_roll)

    # Crear DataFrame reconstruido
    df_reconstructed = df_raw.copy()
    df_reconstructed['piano_rolls'] = piano_rolls_reconstructed

    # Limpiar columnas auxiliares de parquet
    cols_to_drop = ['sequence_length', 'piano_roll_size']
    for col in cols_to_drop:
        if col in df_reconstructed.columns:
            df_reconstructed.drop(columns=[col], inplace=True)

    return df_reconstructed

def augment_data_real(x_train, augmentation_factor=3):
        """
        Aumenta realmente el dataset creando múltiples versiones transpuestas.
        
        Args:
            x_train: Datos originales
            augmentation_factor: Cuántas versiones adicionales crear por muestra
            
        Returns:
            Datos aumentados (originales + transpuestos)
        """
        print(f"\nAumentando datos con factor {augmentation_factor}...")
        print(f"Datos originales: {x_train.shape[0]} muestras")
        
        augmented_data = [x_train]  # Comenzar con datos originales
        
        for aug_idx in range(augmentation_factor):
            print(f"Creando aumento {aug_idx + 1}/{augmentation_factor}...")
            
            # Crear versión transpuesta de todo el dataset
            transposed_data = []
            
            for sample in tqdm(x_train, desc=f"Transposición {aug_idx + 1}"):
                # Transposición aleatoria diferente para cada muestra
                semitones = np.random.randint(-5, 7)
                transposed_sample = np.roll(sample, shift=semitones, axis=-1)
                transposed_data.append(transposed_sample)
            
            augmented_data.append(np.array(transposed_data))
        
        # Concatenar todos los datos
        final_data = np.concatenate(augmented_data, axis=0)
        
        print(f"Datos aumentados: {final_data.shape[0]} muestras")
        print(f"Factor de aumento real: {final_data.shape[0] / x_train.shape[0]:.1f}x")
        
        return final_data