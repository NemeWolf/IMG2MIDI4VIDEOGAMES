import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# --- FUNCIÓN DE AUMENTO DE DATOS ---
# Esta función se usará "al vuelo" durante el entrenamiento, no aquí.
# La incluimos para tenerla lista para la siguiente etapa.

def transpose_piano_roll(piano_roll, semitones, piano_range=(24, 108)):
    """
    Transpone una secuencia de piano roll un número de semitonos.
    
    Args:
        piano_roll (np.ndarray): Matriz de (N, piano_size).
        semitones (int): Número de semitonos a transponer (positivo o negativo).
        piano_range (tuple): El rango de notas MIDI del piano roll.
        
    Returns:
        np.ndarray: La matriz de piano roll transpuesta.
    """
    # Creamos una copia para no modificar el original
    transposed_roll = np.zeros_like(piano_roll)
    piano_size = piano_range[1] - piano_range[0]
    
    # Iteramos por cada paso de tiempo y cada nota
    for step in range(piano_roll.shape[0]):
        for note_idx in range(piano_size):
            if piano_roll[step, note_idx] == 1:
                # Calculamos el nuevo índice de la nota
                new_note_idx = note_idx + semitones
                # Nos aseguramos de que la nota transpuesta no se salga del rango del piano
                if 0 <= new_note_idx < piano_size:
                    transposed_roll[step, new_note_idx] = 1
                    
    return transposed_roll

# --- CLASE PRINCIPAL PARA EL PROCESAMIENTO FINAL ---

class FinalDatasetProcessor:
    """
    Carga los datasets pre-procesados, los limpia, los combina
    estratégicamente y los guarda en los conjuntos de datos finales.
    """
    
    def __init__(self, chordonomicon_path, popular_hook_path, output_dir):
        self.chordonomicon_path = chordonomicon_path
        self.popular_hook_path = popular_hook_path
        self.output_dir = output_dir

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Directorio de salida creado en: {self.output_dir}")

    def load_data(self):
        """Carga los dos datasets procesados desde archivos .pkl."""
        print("Cargando datasets pre-procesados...")
        try:
            # Usamos pd.read_pickle que es más robusto para DataFrames
            chordnomicon = pd.read_pickle(self.chordonomicon_path)
            popular_hook = pd.read_pickle(self.popular_hook_path)
            
            chordnomicon_piano_rolls = chordnomicon['piano_rolls']
            popular_hook_piano_rolls = popular_hook['piano_rolls']
            
            chordnomicon_chords_symbols = chordnomicon['chord_symbols']
            popular_hook_chords_symbols = popular_hook['chord_symbols']
            
            chordnomicon_metadata = chordnomicon['metadata']
            popular_hook_metadata = popular_hook['metadata']
                        
            self.df_chordonomicon = pd.DataFrame({
                'piano_roll': list(chordnomicon_piano_rolls),
                'chord_symbols': chordnomicon_chords_symbols,
                **chordnomicon_metadata
            })
            self.df_popular_hook = pd.DataFrame({
                'piano_roll': list(popular_hook_piano_rolls),
                'chord_symbols': popular_hook_chords_symbols,
                **popular_hook_metadata
            })
            
            print(f"Datos cargados: {len(self.df_chordonomicon)} secuencias de Chordonomicon, "
                  f"{len(self.df_popular_hook)} secuencias de Popular Hook.")
        except FileNotFoundError as e:
            print(f"Error: No se pudo encontrar un archivo .pkl. Asegúrate de que las rutas son correctas.")
            raise e

    def clean_data(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        """Realiza la limpieza de un DataFrame: elimina nulos y duplicados."""
        print(f"Limpiando el dataset '{name}'...")
        # Eliminar filas donde el piano_roll pueda ser nulo o vacío
        df.dropna(subset=['piano_roll'], inplace=True)
        
        # Eliminar duplicados exactos en las progresiones de piano roll
        # Para hacer esto, convertimos temporalmente el array a bytes para que sea "hashable"
        initial_rows = len(df)
        df['piano_roll_hash'] = df['piano_roll'].apply(lambda x: x.tobytes())
        df.drop_duplicates(subset=['piano_roll_hash'], keep='first', inplace=True)
        df.drop(columns=['piano_roll_hash'], inplace=True)
        final_rows = len(df)
        
        print(f"Se eliminaron {initial_rows - final_rows} secuencias duplicadas o vacías. "
              f"Tamaño final: {final_rows} secuencias.")
        return df

    def create_final_datasets(self):
        """
        Ejecuta el pipeline completo de carga, limpieza, división y guardado.
        """
        self.load_data()

        # 1. Limpiar ambos datasets
        df_chordonomicon_clean = self.clean_data(self.df_chordonomicon, "Chordonomicon")
        df_popular_hook_clean = self.clean_data(self.df_popular_hook, "Popular Hook")

        # 2. Dividir Popular Hook
        print("Dividiendo Popular Hook en 'Video Game' y 'Otros géneros'...")
        is_videogame = df_popular_hook_clean['genres'].str.contains('Video Game', na=False, case=False)
        df_ph_videogame = df_popular_hook_clean[is_videogame]
        df_ph_other = df_popular_hook_clean[~is_videogame]
        
        # --- 3. Ensamblar los Datasets Finales ---
        # Dataset para el entrenamiento principal del VAE
        dataset_vae_main = pd.concat([df_chordonomicon_clean, df_ph_other], ignore_index=True)
        
        # Dataset para el fine-tuning final (solo la música por ahora)
        dataset_finetune_music = df_ph_videogame
        
        # --- 4. Guardar los Datasets ---
        path_vae_main = os.path.join(self.output_dir, 'dataset_vae_main.pkl')
        path_finetune = os.path.join(self.output_dir, 'dataset_finetune_music.pkl')
        
        print("\n--- Resumen de Datasets Finales ---")
        print(f"Dataset VAE Main (Chordonomicon + Otros): {len(dataset_vae_main)} secuencias")
        print(f"Dataset Fine-Tuning (Video Game): {len(dataset_finetune_music)} secuencias")
        
        print(f"\nGuardando 'dataset_vae_main.pkl'...")
        dataset_vae_main.to_pickle(path_vae_main)
        
        print(f"Guardando 'dataset_finetune_music.pkl'...")
        dataset_finetune_music.to_pickle(path_finetune)
        
        print("\n¡Proceso completado! Tus datasets están limpios, organizados y listos.")

if __name__ == '__main__':
    # --- Configuración ---
    # Directorio donde se encuentran los archivos .pkl de entrada
    CHORDONOMICON_PATH = '/mnt/c/Users/nehem/OneDrive - Universidad de Chile/Universidad/6to año/Data/MIDI/preprocced/Chordomicon/dataset_01.pkl'
    POPULARHOOK_PATH = '/mnt/c/Users/nehem/OneDrive - Universidad de Chile/Universidad/6to año/Data/MIDI/preprocced/Popular-hook/dataset_01.pkl'
    # Directorio donde se guardarán los datasets finales
    PROCESSED_DATA_DIR = '/mnt/c/Users/nehem/OneDrive - Universidad de Chile/Universidad/6to año/Data/MIDI/preprocced/'
        
    # --- Ejecución ---
    processor = FinalDatasetProcessor(
        chordonomicon_path=CHORDONOMICON_PATH,
        popular_hook_path=POPULARHOOK_PATH,
        output_dir=PROCESSED_DATA_DIR
    )
    
    processor.create_final_datasets()