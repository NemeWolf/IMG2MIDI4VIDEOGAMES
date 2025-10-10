import pandas as pd
import music21
import numpy as np
import os
import re
import glob
import math
from tqdm import tqdm
import json

class PopularHookPreprocessorResumable:
    """
    Procesador para Popular Hook con batching, reanudación y enventanado dinámico.
    """
    def __init__(self, dataset_path, # Ruta al directorio del dataset
                 info_tables_file, # Ruta al archivo info_tables.xlsx
                 num_progresions, # Número de progresiones a procesar
                 sequence_length=16, # Longitud de la secuencia de acordes
                 batch_size=5000, # Tamaño del batch para guardar
                 max_windows_per_progression=10, # Máximo de ventanas por progresión
                 resume_from_idx=None, # Índice 'idx' desde donde reanudar (None para no reanudar)
                 resume_windows_done=0 # Número de ventanas ya procesadas en la fila de reanudación
                 ):
        
        self.dataset_path = dataset_path
        self.info_tables_path = info_tables_file
        self.num_progresions = num_progresions
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.max_windows_per_progression = max_windows_per_progression
        self.resume_from_idx = resume_from_idx
        self.resume_windows_done = int(resume_windows_done or 0)
        
        # Rango de notas del piano (A0 a C8)
        self.piano_range = (24, 108)
        # Tamaño del piano
        self.piano_size = self.piano_range[1] - self.piano_range[0]

        # Prueba de existencia de rutas
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"No se encontró el directorio del dataset en: {self.dataset_path}")
        if not os.path.exists(self.info_tables_path):
            raise FileNotFoundError(f"No se encontró el archivo info_tables en: {self.info_tables_path}")

        # Cargar metadatos
        print("Cargando archivo info_tables...")
        
        if self.num_progresions is None:
            self.metadata_df = pd.read_excel(self.info_tables_path, engine='openpyxl')
        else:
            self.metadata_df = pd.read_excel(self.info_tables_path, engine='openpyxl', nrows=self.num_progresions)

        print(f"Metadatos cargados. Se encontraron {len(self.metadata_df)} entradas.")

    def _extract_emotion_from_csv(self, emotion_csv_path: str) -> dict:
        """
        Extrae la emoción predicha desde el archivo CSV de cada progresión.
        args:
            emotion_csv_path (str): Ruta al archivo CSV de emoción.
        returns:
            dict: Diccionario con la emoción predicha.
        """
        try:
            emotion_df = pd.read_csv(emotion_csv_path)
            return {'midi_emotion_predected': emotion_df['midi_emotion_predected'].iloc[0]}
        except FileNotFoundError:
            print(f"Archivo de emoción no encontrado: {emotion_csv_path}")
            return {}
        except Exception as e:
            print(f"Error leyendo emoción {emotion_csv_path}: {e}")
            return {}
    def _parse_tonality_to_key(self, tonality_str: str):
        """
        Convierte string de tonalidad del dataset Popular Hook a objeto Key de music21.
        args:
            tonality_str (str): String de tonalidad (ej: 'A minor', 'C Major', 'D dorian')
        returns:
            music21.key.Key: Objeto Key o None si no se puede parsear
        """
        if not tonality_str or tonality_str.lower() in ['unknown', 'n/a', '']:
            return None
        
        try:
            # Normalizar el string y convertir a minúsculas para el modo
            tonality_clean = tonality_str.strip()
            
            # Separar tono y modo
            parts = tonality_clean.split()
            if len(parts) != 2:
                print(f"Formato de tonalidad no reconocido: '{tonality_str}'")
                return None
                
            tone, mode = parts[0], parts[1].lower()  # Convertir modo a minúsculas
            
            # Mapeo de modos del dataset a music21
            mode_mapping = {
                'major': 'major',
                'minor': 'minor',
                'dorian': 'dorian',
                'phrygian': 'phrygian',
                'lydian': 'lydian',
                'mixolydian': 'mixolydian',
                'locrian': 'locrian',
                'harmonicminor': 'minor',
                'phrygiandominant': 'phrygian'
            }
            
            # Verificar si el modo existe en el mapeo
            if mode not in mode_mapping:
                print(f"Modo no reconocido: '{mode}' en tonalidad '{tonality_str}'")
                return None
            
            # Mapeo de tonos del dataset
            tone_mapping = {
                'D#': 'D#', 'F': 'F', 'F#': 'F#', 'Db': 'D-',
                'A#': 'A#', 'G': 'G', 'E': 'E', 'D': 'D',
                'Eb': 'E-', 'A': 'A', 'Bb': 'B-', 'C': 'C',
                'B': 'B', 'G#': 'G#', 'C#': 'C#', 'Gb': 'G-',
                'E#': 'E#', 'Ab': 'A-'
            }
            
            # Convertir tono y modo
            music21_tone = tone_mapping.get(tone, tone)
            music21_mode = mode_mapping[mode]  # Ya verificamos que existe
                        
            # Crear el objeto Key
            return music21.key.Key(music21_tone, music21_mode)
                    
        except Exception as e:
            print(f"Error parseando tonalidad '{tonality_str}' --> : {e}")
            return None    
    
    
    def _extract_chords_from_midi(self, midi_path: str, tonality: str) -> list:
        """
        Extrae la secuencia de acordes de un archivo MIDI usando music21.
        args:
            midi_path (str): Ruta al archivo MIDI.
        returns:
            list: Lista de objetos Chord de music21 o None si no se encuentran acordes
        """
        try:
            score = music21.converter.parse(midi_path)
            
            # Configurar tonalidad
            key_obj = None
            if tonality:
                try:
                    key_obj = self._parse_tonality_to_key(tonality)
                    if key_obj:
                        score.insert(0, key_obj)         
                                       
                except Exception as e:
                    print(f"No se pudo configurar la tonalidad '{tonality}': {e}")
            
            # Los archivos .mid contienen una parte llamada "Chord" que tiene los acordes
            chord_part = None # Parte que contiene los acordes                                
            for part in score.parts:
                if 'Chord' in str(part.partName).title():
                    chord_part = part
                    break
            if chord_part is None:
                return [None]
            
            # Aplicar tonalidad a la parte también
            if key_obj:
                chord_part.insert(0, key_obj)   
            
            # Extraemos los acordes
            chordified_part = chord_part.chordify()
            
            # Obtenemos todos los objetos Chord
            chords = [element for element in chordified_part.recurse().getElementsByClass('Chord')]
            return chords if chords else [None]
        
        except Exception as e:
            print(f"No se pudo procesar el archivo MIDI {os.path.basename(midi_path)}: {e}")
            return [None]

    def _chords_to_piano_roll(self, chord_sequence: list) -> np.ndarray:
        """
        Convierte una secuencia de acordes en un piano roll.
        args:
            chord_sequence (list): Lista de objetos Chord de music21.
        returns:
            np.ndarray: Piano roll binario de forma (sequence_length, piano_size).
        """
        
        piano_roll = np.zeros((self.sequence_length, self.piano_size), dtype=np.int8)
        for i, chord in enumerate(chord_sequence):
            for pitch in chord.pitches:
                midi_note = pitch.midi
                if self.piano_range[0] <= midi_note < self.piano_range[1]:
                    note_index = midi_note - self.piano_range[0]
                    piano_roll[i, note_index] = 1
        return piano_roll

    def _next_existing_batch_index(self, batch_dir):
        """
        Encuentra el siguiente índice de batch basado en los archivos existentes.
        args:
            batch_dir (str): Directorio donde se guardan los batches.
        returns:
            int: Siguiente índice de batch.
        """
        os.makedirs(batch_dir, exist_ok=True)
        files = glob.glob(os.path.join(batch_dir, 'dataset_01_*.pkl'))
        if not files:
            return 0
        nums = []
        for p in files:
            m = re.search(r'_(\d+)\.pkl$', os.path.basename(p))
            if m:
                nums.append(int(m.group(1)))
        return max(nums) if nums else 0

    def process_dataset(self, save_path: str, genre_filter=None):
        """
        Procesa el conjunto de datos, guardando en batches y permitiendo reanudación.
        args:
            save_path (str): Ruta donde se guardará el dataset procesado.
            genre_filter (str, optional): Género musical para filtrar. Si es None, se procesan todos los géneros.
        returns:
            dict: Diccionario con los datos procesados del último batch (parcial).
        """        
        # CARGAR DATOS
        
        # Captura del DataFrame objetivo
        target_df = self.metadata_df
        
        # Filtrado por género si se especifica
        if genre_filter:
            print(f"Filtrando el dataset por el género: '{genre_filter}'...")
            target_df = self.metadata_df[self.metadata_df['genres'].str.contains(genre_filter, na=False)].copy()
            print(f"Se encontraron {len(target_df)} entradas para el género '{genre_filter}'.")

        # Inicialización de listas para el batch
        batch = []
        batch_count = 0
        
        # Configuración para reanudación
        batch_dir = os.path.join(os.path.dirname(save_path), 'batch')
        current_batch_idx = self._next_existing_batch_index(batch_dir)

        # Variables para reanudación
        resuming = self.resume_from_idx is not None
        reached_resume_row = False
        resume_skip_windows = int(self.resume_windows_done) if resuming else 0

        # PROCESAMIENTO PRINCIPAL ============================
        
        print("Procesando archivos MIDI y de emoción por lotes...")
        
        # Iteramos por cada fila del DataFrame
        for idx, row in tqdm(target_df.iterrows(), total=target_df.shape[0]):
            # Reanudar: saltar hasta la fila objetivo
            if resuming and not reached_resume_row:
                if row.get('idx') != self.resume_from_idx:
                    continue
                reached_resume_row = True

            # Extraemos rutas y verificamos existencia
            path_from_info = row.get('path', '')
            path_from_info = path_from_info[2:]
            if not path_from_info:
                continue
            
            # capturamos tonalidad dada por metadatos
            tonality_str = str(row.get('tonality'))  # Obtener tonalidad de metadatos
            
            # Construimos rutas completas a archivos
            section_folder_path = path_from_info.replace('.mid', '')
            full_section_path = os.path.join(self.dataset_path, section_folder_path)
            section_name = os.path.basename(full_section_path)
            
            midi_file_path = os.path.join(full_section_path, f"{section_name}.mid") # Archivo MIDI
            emotion_csv_path = os.path.join(full_section_path, f"{section_name}_midi_emotion_result.csv") # Archivo de emoción
            
            # Si no existen los archivos, saltar            
            if not os.path.exists(midi_file_path) or not os.path.exists(emotion_csv_path):
                continue
            
            # Extraemos acordes y emoción
            m21_chords = self._extract_chords_from_midi(midi_file_path, tonality_str)
            emotion_data = self._extract_emotion_from_csv(emotion_csv_path)
            
            # Si no hay acordes o son insuficientes, saltar
            if not m21_chords or len(m21_chords) < self.sequence_length:
                if resuming and reached_resume_row:
                    resuming = False
                continue
            
            # ENVENTANADO ====================================
            
            # Cantidad de ventanas posibles
            windows_amount = len(m21_chords) - self.sequence_length + 1
            
            # Enventanado dinámico
            if windows_amount <= self.max_windows_per_progression:
                # Si hay pocas ventanas, tomamos todas
                starts = list(range(0, windows_amount))
            else:
                # Si hay más ventanas que el máximo, calculamos un stride en base al máximo permitido
                stride_dyn = math.ceil(windows_amount / self.max_windows_per_progression)
                starts = list(range(0, windows_amount, stride_dyn))
                
                # Nos aseguramos de no exceder el máximo
                if len(starts) > self.max_windows_per_progression:
                    starts = starts[:self.max_windows_per_progression]
                last_start = windows_amount - 1
                
                # Aseguramos incluir la última ventana
                if starts and starts[-1] != last_start:
                    if len(starts) == self.max_windows_per_progression:
                        starts[-1] = last_start
                    else:
                        starts.append(last_start)
                        
            # Si reanudando en esta fila, saltar ventanas ya guardadas
            if resuming and reached_resume_row:
                if resume_skip_windows > 0:
                    starts = starts[resume_skip_windows:]
                resuming = False
                
            # Si no quedan ventanas para procesar, saltar                   
            if not starts:
                continue            
            
            # PROCESAMIENTO DE VENTANAS ============================
            
            for i in starts:
                
                sequence = m21_chords[i:i + self.sequence_length] # Secuencia de acordes actual
                
                # Convertimos a piano roll y extraemos símbolos
                piano_roll_sequence = self._chords_to_piano_roll(sequence)
                
                # Aplanamos piano roll a 1D
                piano_roll_flat = piano_roll_sequence.flatten().tolist()
                
                # Extraemos símbolos de acordes
                chord_symbol_sequence = []
                for c in sequence:
                    try:
                        symbol = music21.harmony.chordSymbolFromChord(c).figure                          
                        if symbol == 'Chord Symbol Cannot Be Identified' or not symbol:
                            symbol = c.pitchNames  # No Chord                        
                    except Exception as e:
                        symbol = c.pitchNames  # No Chord
                
                    chord_symbol_sequence.append(symbol)

                # Convertir chord_symbols a JSON string
                chord_symbol_json = json.dumps(chord_symbol_sequence)
                
                # Añadimos al batch actual
                batch.append({
                    'piano_rolls': piano_roll_flat,
                    'chord_symbols': chord_symbol_json,
                    'sequence_length': self.sequence_length,
                    'piano_roll_size': self.piano_size,
                    'idx': row.get('idx'),
                    'artist': str(row.get('singer', 'Unknown')),
                    'song': str(row.get('song', 'Unknown')),
                    'section': str(row.get('section', 'Unknown')),
                    'tonality': str(row.get('tonality', 'Unknown')),
                    'genres': str(row.get('genres', 'Unknown')),
                    'path': str(full_section_path),
                    **emotion_data
                })
                                 
                batch_count += 1 # Incrementamos contador de batch                 
                
                # Si alcanzamos el tamaño del batch, guardamos y reseteamos
                if batch_count == self.batch_size:
                    current_batch_idx += 1  # Incrementamos índice de batch
                    
                    # Guardamos el batch actual
                    df_rows = pd.DataFrame(batch)                    
                    
                    out_path = os.path.join(batch_dir, f'dataset_01_{current_batch_idx}.parquet')
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    
                    df_rows.to_parquet(out_path, compression='snappy')
                    
                    # Limpieza
                    del df_rows
                    del batch[:]
                    
                    # Reseteamos listas y contador
                    batch = []
                    batch_count = 0
                    
        # Guardar batch parcial restante
        if batch_count > 0:
            current_batch_idx += 1
            
            df_rows = pd.DataFrame(batch)
            
            out_path = os.path.join(batch_dir, f'dataset_01_{current_batch_idx}.parquet')
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            df_rows.to_parquet(out_path, compression='snappy')

        print(f'Procesamiento completo. Batches escritos hasta índice {current_batch_idx}.')
        return {

    }

if __name__ == '__main__':
    # --- Configuración ---
    
    # Directorio raíz donde descomprimiste el dataset Popular Hook
    DATASET_ROOT_PATH = '/home/neme/workspace/Data/MIDI/preprocced/Popular-hook'
    
    # nombre del archivo de metadatos principal y su ruta
    INFO_TABLES_FILENAME = 'info_tables.xlsx'
    INFO_TABLES_FILE_PATH = os.path.join(DATASET_ROOT_PATH, INFO_TABLES_FILENAME)
    
    # Ruta de salida para los batches procesados
    OUTPUT_PATH = '/home/neme/workspace/Data/MIDI/preprocced/Popular-hook/'
    
    # --- Parámetros de procesamiento ---
    NUM_PROGRESION = None
    BATCH_SIZE = 5000
    MAX_WINDOWS_PER_PROGRESSION = 10
    
    # --- Reanudación ---
    RESUME_FROM_IDX = None  # Ejemplo: 12345
    RESUME_WINDOWS_DONE = 0
    
    preprocessor = PopularHookPreprocessorResumable(
        dataset_path=DATASET_ROOT_PATH,
        info_tables_file=INFO_TABLES_FILE_PATH,
        num_progresions=NUM_PROGRESION,
        sequence_length=16,
        batch_size=BATCH_SIZE,
        max_windows_per_progression=MAX_WINDOWS_PER_PROGRESSION,
        resume_from_idx=RESUME_FROM_IDX,
        resume_windows_done=RESUME_WINDOWS_DONE
    )
    result = preprocessor.process_dataset(
        save_path=OUTPUT_PATH,
        genre_filter=None
    )