import pandas as pd
import music21
import numpy as np
import os
from tqdm import tqdm

class PopularHookPreprocessor:
    """
    Clase para procesar el dataset Popular Hook, adaptada a la estructura de archivos
    descrita en info_tables, incluyendo la corrección de rutas y la extracción de 
    metadatos de emoción desde archivos CSV separados.
    """

    def __init__(self, dataset_path, info_tables_file, sequence_length=16, stride=4, piano_range=(24, 108)):
        self.dataset_path = dataset_path
        self.info_tables_path = info_tables_file
        self.sequence_length = sequence_length
        self.stride = stride
        self.piano_range = piano_range
        self.piano_size = piano_range[1] - piano_range[0]

        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"No se encontró el directorio del dataset en: {self.dataset_path}")
        if not os.path.exists(self.info_tables_path):
            raise FileNotFoundError(f"No se encontró el archivo info_tables en: {self.info_tables_path}")

        print("Cargando archivo info_tables...")
        self.metadata_df = pd.read_excel(INFO_TABLES_FILE_PATH, engine='openpyxl')
        print(f"Metadatos cargados. Se encontraron {len(self.metadata_df)} entradas.")

    def _extract_emotion_from_csv(self, emotion_csv_path: str) -> dict:
        """Lee el archivo CSV de emoción y extrae los valores relevantes."""
        try:
            emotion_df = pd.read_csv(emotion_csv_path)
            # El dataset puede tener columnas como 'Q1', 'Q2', 'Q3', 'Q4' de Russell.
            return {'midi_emotion_predected': emotion_df['midi_emotion_predected'].iloc[0]}
        
        except FileNotFoundError:
            print(f"Archivo de emoción no encontrado: {emotion_csv_path}")
            return {}
        except Exception as e:
            print(f"Error leyendo emoción {emotion_csv_path}: {e}")
            return {}

    def _extract_chords_from_midi(self,midi_path: str) -> list:
        """
        Extrae una secuencia de acordes de un archivo MIDI, buscando y utilizando
        únicamente la pista llamada 'Chord'.
        """
        try:
            # 1. Cargar el archivo MIDI completo
            score = music21.converter.parse(midi_path)
            
            # 2. Buscar la pista (Part) que contenga 'Chord' en su nombre
            chord_part = None
            
            for part in score.parts:
                # El nombre de la pista suele estar en el atributo .id o .partName
                # Usamos .title() y 'in' para ser flexibles (ej. 'chord', 'Chord', 'CHORD')
                if 'Chord' in str(part.partName).title():
                    chord_part = part
                    break # Encontramos la pista, salimos del bucle
            
            # 3. Si no se encuentra una pista de acordes, no podemos continuar
            if chord_part is None:
                # Opcional: podrías intentar hacer chordify() a toda la partitura como fallback
                # score.chordify() si quieres, pero es más seguro descartar el archivo.
                return None

            # 4. ¡La clave! Aplicar Chordify solo sobre la pista de acordes
            chordified_part = chord_part.chordify()
            
            # 5. Extraer los acordes de la parte ya procesada
            chords = [element for element in chordified_part.recurse().getElementsByClass('Chord')]
                
            return chords if chords else None
        
        except Exception as e:
            print(f"No se pudo procesar el archivo MIDI {os.path.basename(midi_path)}: {e}")
            return None

    def _chords_to_piano_roll(self, chord_sequence: list) -> np.ndarray:
        """Convierte una secuencia de N acordes de music21 a una matriz de piano roll."""
        piano_roll = np.zeros((self.sequence_length, self.piano_size), dtype=np.int8)
        for i, chord in enumerate(chord_sequence):
            for pitch in chord.pitches:
                midi_note = pitch.midi
                if self.piano_range[0] <= midi_note < self.piano_range[1]:
                    note_index = midi_note - self.piano_range[0]
                    piano_roll[i, note_index] = 1
        return piano_roll

    def process_dataset(self, save_path: str, genre_filter=None):
        """
        Procesa el dataset completo, filtra por género (opcional) y guarda los resultados.
        """
        target_df = self.metadata_df
        if genre_filter:
            print(f"Filtrando el dataset por el género: '{genre_filter}'...")
            # Filtrar si la columna 'genres' contiene el string del filtro
            target_df = self.metadata_df[self.metadata_df['genres'].str.contains(genre_filter, na=False)].copy()
            print(f"Se encontraron {len(target_df)} entradas para el género '{genre_filter}'.")

        processed_data = []

        print("Procesando archivos MIDI y de emoción...")
        
        all_sequences_chords = []
        all_sequences_pianoroll = []
        all_metadata = []    
        
        for index, row in tqdm(target_df.iterrows(), total=target_df.shape[0]):
            path_from_info = row.get('path', '')
            path_from_info = path_from_info[2:]
            
            if not path_from_info:
                continue

            # 1. Lógica de corrección de rutas
            section_folder_path = path_from_info.replace('.mid', '')
            full_section_path = os.path.join(self.dataset_path, section_folder_path)
            
            section_name = os.path.basename(full_section_path)

            # 2. Construir rutas a los archivos MIDI y de emoción
            midi_file_path = os.path.join(full_section_path, f"{section_name}.mid")
            emotion_csv_path = os.path.join(full_section_path, f"{section_name}_midi_emotion_result.csv")

            if not os.path.exists(midi_file_path) or not os.path.exists(emotion_csv_path):
                continue
            
            # 3. Extraer datos de ambas fuentes
            m21_chords = self._extract_chords_from_midi(midi_file_path)
            emotion_data = self._extract_emotion_from_csv(emotion_csv_path)

            if not m21_chords or len(m21_chords) < self.sequence_length:
                continue

            # 4. Aplicar ventana deslizante y empaquetar datos
            
            windows_amount = len(m21_chords) - self.sequence_length + 1
            
            for i in range(0, windows_amount, self.stride):
                sequence = m21_chords[i:i + self.sequence_length]
                
                piano_roll_sequence = self._chords_to_piano_roll(sequence)
                chord_symbol_sequence = [c.pitchedCommonName for c in sequence]
                
                all_sequences_chords.append(chord_symbol_sequence)
                all_sequences_pianoroll.append(np.array(piano_roll_sequence))

                all_metadata.append({
                    'idx': row.get('idx'),
                    'artist': row.get('singer', 'Unknown'),
                    'song': row.get('song', 'Unknown'),
                    'section': row.get('section', 'Unknown'),
                    'tonality': row.get('tonality', 'Unknown'),
                    'genres': row.get('genres', 'Unknown'),
                    'path': full_section_path,
                    **emotion_data # Añadir datos de emoción
                    }
                )       
                                    
        processed_data = {
            'piano_rolls': np.array(all_sequences_pianoroll),
            'chord_symbols': all_sequences_chords,
            'metadata': pd.DataFrame(all_metadata)
        }
        print(f"Procesamiento completo. Se generaron {len(processed_data['piano_rolls'])} secuencias.")
        
        print(f"Guardando datos en {save_path}...")
        pd.to_pickle(processed_data, save_path)
        print("Datos guardados exitosamente.")
        return processed_data

if __name__ == '__main__':
    # --- Configuración ---
    
    # Directorio raíz donde descomprimiste el dataset Popular Hook
    DATASET_ROOT_PATH = '/mnt/c/Users/nehem/Desktop/Tesis/Data/MIDI/popular-hook'
    
    # Nombre del archivo de metadatos principal
    INFO_TABLES_FILENAME = 'info_tables.xlsx'
    
    # Ruta completa al archivo de metadatos
    INFO_TABLES_FILE_PATH = os.path.join(DATASET_ROOT_PATH, INFO_TABLES_FILENAME)
    
    # Ruta de salida para el archivo procesado de videojuegos
    OUTPUT_PATH = '/mnt/c/Users/nehem/OneDrive - Universidad de Chile/Universidad/6to año/Data/MIDI/preprocced/Popular-hook/dataset_01.pkl' # Ruta para guardar el dataset procesado

    # --- Ejecución ---
    try:
        preprocessor = PopularHookPreprocessor(
            dataset_path=DATASET_ROOT_PATH,
            info_tables_file=INFO_TABLES_FILE_PATH,
            sequence_length=16,
            stride=4
        )
        
        processed_dataset = preprocessor.process_dataset(
            save_path=OUTPUT_PATH,
            genre_filter=None  # Cambia a 'Video Game' si quieres filtrar por ese género
        )
        
        # --- Verificación ---
        if processed_dataset:
            print("\n--- Ejemplo de una secuencia procesada de Popular Hook ---")
            
            sample = {
                'piano_roll': processed_dataset['piano_rolls'][-1],
                'chord_symbols': processed_dataset['chord_symbols'][-1],
                **processed_dataset['metadata'].iloc[-1].to_dict()
            }
            
            print(f"Índice (idx): {sample.get('idx')}")
            print(f"Artista/Juego: {sample.get('artist')}")
            print(f"Canción: {sample.get('song')}")
            print(f"Sección: {sample.get('section')}")
            print(f"Símbolos de Acordes: {sample['chord_symbols']}")
            print(f"Tonalidad: {sample.get('tonality')}")
            print(f"Géneros: {sample.get('genres')}")
            print(f"Emoción (Cuadrante de Russell): {sample.get('midi_emotion_predected', 'No disponible')}")
            print(f"Forma del Piano Roll: {sample['piano_roll'].shape}")

    except FileNotFoundError as e:
        print(f"\nERROR: No se pudo encontrar un archivo o directorio. Por favor, revisa las rutas en la sección de 'Configuración' del script.")
        print(f"Detalle: {e}")
    except Exception as e:
        print(f"\nOcurrió un error inesperado: {e}")