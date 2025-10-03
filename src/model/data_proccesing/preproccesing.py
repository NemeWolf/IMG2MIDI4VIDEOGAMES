import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split

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

    def create_final_datasets(self, key_correlation_threshold=0.7, test_size=0.1, dev_size=0.1):
        """
        Ejecuta el pipeline completo de carga, limpieza, división y guardado.
        """
        self.load_data()

        # 1. Limpiar ambos datasets
        df_chordonomicon_clean = self.clean_data(self.df_chordonomicon, "Chordonomicon")
        df_popular_hook_clean = self.clean_data(self.df_popular_hook, "Popular Hook")

        # 2. Filtrar Chordonomicon por confianza de tonalidad
        print(f"Filtrando Chordonomicon por confianza de tonalidad > {key_correlation_threshold}...")
        initial_rows = len(df_chordonomicon_clean)
        df_chordonomicon_filtered = df_chordonomicon_clean[df_chordonomicon_clean['key_correlation'] > key_correlation_threshold]
        print(f"Se eliminaron {initial_rows - len(df_chordonomicon_filtered)} secuencias con tonalidad ambigua.")
        
        # 3. Unificar Esquema de Metadatos
        print("Unificando esquemas de metadatos...")
        # Renombrar columnas de Chordonomicon para que coincidan
        df_chordonomicon_filtered['tonality'] = df_chordonomicon_filtered['key_tonic'] + ' ' + df_chordonomicon_filtered['key_mode']
        df_chordonomicon_filtered.rename(columns={'artist_id': 'artist', 'song_id': 'song'}, inplace=True)

        # Definir columnas comunes
        COMMON_COLS = ['artist', 'song', 'tonality', 'genres', 'piano_roll', 'chord_symbols']
        
        # Seleccionar columnas
        df_chordonomicon_final = df_chordonomicon_filtered[COMMON_COLS]
        
        # 4. Dividir Popular Hook
        print("Dividiendo Popular Hook...")
        is_videogame = df_popular_hook_clean['genres'].str.contains('Video Game', na=False, case=False)
        df_ph_videogame = df_popular_hook_clean[is_videogame]
        df_ph_other = df_popular_hook_clean[~is_videogame]
        
        # 5. Ensamblar los Datasets
        # Dataset principal: Chordonomicon (filtrado) + Popular Hook (no videojuegos)
        dataset_vae_main = pd.concat([df_chordonomicon_final, df_ph_other[COMMON_COLS]], ignore_index=True)
        
        # Dataset de fine-tuning: Solo videojuegos, conservando la emoción
        FINETUNE_COLS = COMMON_COLS + ['midi_emotion_predected']
        dataset_finetune_music = df_ph_videogame[FINETUNE_COLS]
        
        # 6. Dividir en Train/Dev/Test
        print("Dividiendo el dataset principal en train/dev/test...")
        train_val_df, test_df = train_test_split(dataset_vae_main, test_size=test_size, random_state=42)
        
        # Ajustar el dev_size para que sea sobre el total original
        dev_size_adjusted = dev_size / (1 - test_size)
        train_df, dev_df = train_test_split(train_val_df, test_size=dev_size_adjusted, random_state=42)

        # --- 7. Guardar los Datasets ---
        print("\n--- Resumen de Datasets Finales ---")
        print(f"Set de Entrenamiento (Train): {len(train_df)} secuencias")
        print(f"Set de Desarrollo (Dev/Validation): {len(dev_df)} secuencias")
        print(f"Set de Prueba (Test): {len(test_df)} secuencias")
        print(f"Dataset de Fine-Tuning (Video Game Music): {len(dataset_finetune_music)} secuencias")
        
        train_df.to_pickle(os.path.join(self.output_dir, 'train_set.pkl'))
        dev_df.to_pickle(os.path.join(self.output_dir, 'dev_set.pkl'))
        test_df.to_pickle(os.path.join(self.output_dir, 'test_set.pkl'))
        dataset_finetune_music.to_pickle(os.path.join(self.output_dir, 'dataset_finetune_music.pkl'))
        
        print(f"\n¡Proceso completado! Archivos guardados en '{self.output_dir}'.")

if __name__ == '__main__':
    # --- Configuración ---
    # Directorio donde se encuentran los archivos .pkl de entrada
    CHORDONOMICON_PATH = '/home/neme/workspace/Data/MIDI/preprocced/Chordomicon/batch/dataset_01_1.pkl'
    POPULARHOOK_PATH = '/home/neme/workspace/Data/MIDI/preprocced/Popular-hook/dataset_01_1.pkl'
    # Directorio donde se guardarán los datasets finales
    PROCESSED_DATA_DIR = '/home/neme/workspace/Data/MIDI/processed_final/'
        
    # --- Ejecución ---
    processor = FinalDatasetProcessor(
        chordonomicon_path=CHORDONOMICON_PATH,
        popular_hook_path=POPULARHOOK_PATH,
        output_dir=PROCESSED_DATA_DIR
    )
    
    processor.create_final_datasets()