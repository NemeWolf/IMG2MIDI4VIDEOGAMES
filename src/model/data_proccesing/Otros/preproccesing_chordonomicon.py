import pandas as pd
import numpy as np
import re
from music21 import stream, chord, key as m21key, pitch, harmony, analysis as m21analysis
from tqdm import tqdm
import ast
import math

NOTES_AMERICAN = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
NOTES_AMERICAN_SHARP = ['Cs', 'Ds', 'Es', 'Fs', 'Gs', 'As', 'Bs']
NOTES_AMERICAN_FLAT = ['Cb', 'Db', 'Eb', 'Fb', 'Gb', 'Ab', 'Bb']

NOTES_LATIN = ['do', 're', 'mi', 'fa', 'sol', 'la', 'si']
NOTES_LATIN_SHARP= ['dos', 'res', 'mis', 'fas', 'sols', 'las', 'sis']
NOTES_LATIN_FLAT= ['dob', 'reb', 'mib', 'fab', 'solb', 'lab', 'sib']

NOTES = dict(zip(NOTES_LATIN, NOTES_AMERICAN))
NOTES_SHARP = dict(zip(NOTES_LATIN_SHARP, NOTES_AMERICAN_SHARP))
NOTES_FLAT = dict(zip(NOTES_LATIN_FLAT, NOTES_AMERICAN_FLAT))    

class ChordonomiconPreprocessor:
    """
    Procesador avanzado para el dataset Chordonomicon.
    Implementa una representación de piano roll, conserva las inversiones,
    extrae metadatos enriquecidos y analiza la tonalidad de las progresiones.
    """
    def __init__(self, csv_path, save_path, chord_mapping_path, degrees_mapping_path, num_progressions=100000, sequence_length=16, stride = 4, batch_size=5000, max_windows_per_progression=10):
        """
        Inicializa el preprocesador.

        Args:
            csv_path (str): Ruta al archivo CSV de Chordonomicon.
            mapping_path (str): Ruta al archivo CSV de mapeo de acordes (mirex_mapping_v2).
            num_progressions (int): Número de filas a procesar del dataset.
            sequence_length (int): Longitud fija de las secuencias de acordes a generar.
        """
        self.csv_path = csv_path
        self.save_path = save_path
        self.num_progressions = num_progressions
        self.sequence_length = sequence_length
        self.stride = stride
        self.batch_size = batch_size
        self.max_windows_per_progression = max_windows_per_progression
        
        self.chord_mapping = self._load_chord_mapping(chord_mapping_path)
        self.degrees_mapping = self.load_degrees_mapping(degrees_mapping_path)
        
        self.note_names = []
        # Rango de piano roll: 8 octavas desde C0 (MIDI 12) a B7 (MIDI 107) -> 96 notas
        self.piano_roll_size = 84
        self.min_midi_note = 24

    def _load_chord_mapping(self, mapping_path):
        """Carga el mapeo de calidad de acordes a un diccionario."""
        mapping_df = pd.read_csv(mapping_path)
        return dict(zip(mapping_df['Original Symbol'].apply(lambda x: x.replace('"', '')), mapping_df['ChordSymbol_m21'].apply(lambda x: x.replace('"', ''))))

    def load_degrees_mapping(self, degrees_path):
        """Carga el mapeo de acordes a sus grados y notas fundamentales."""
        degrees_df = pd.read_csv(degrees_path)
        return dict(zip(degrees_df['Chords'], degrees_df['Notes'].apply(ast.literal_eval)))
    
    def _clean_chord_string(self, chords_text):
        """
        Limpia la cadena de texto de acordes, eliminando etiquetas de sección.
        args:
            chords_text (str): Cadena de texto con acordes y etiquetas.
        returns:
            list: Lista de acordes limpios.        
        """
        # Elimina etiquetas como <intro_1>, <verse_1>, etc.
        cleaned_text = re.sub(r'<[^>]+>', '', chords_text)
        # Reemplaza múltiples espacios con uno solo
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        return cleaned_text.split(' ')

    def extract_chord_symbol(self,chord, note):
        
        for c in range(len(NOTES_AMERICAN)):                        
            if note == NOTES_LATIN[c]:
                tonic = NOTES[note]
                break
            elif note == NOTES_LATIN_SHARP[c]:
                tonic = NOTES_SHARP[note]
                break
            elif note == NOTES_LATIN_FLAT[c]:
                tonic = NOTES_FLAT[note]
                break        
        symbol = chord[len(tonic):]     
        return tonic, symbol, chord
    
    def _parse_single_chord(self, chord_str):
        """
        Parsea un único acorde de texto a un objeto Chord de music21, conservando la inversión.
        args:
            chord_str (str): Cadena de texto representando un acorde.
        returns:
            music21.chord.Chord or None: Objeto Chord o None si no se puede parsear.
        """

        # Capturar la nota raíz y la nota del bajo si existe
        chord_str = chord_str.split("/")
        
        chord = chord_str[0]
        bass = chord_str[1] if len(chord_str) > 1 else ''
        
        root, quality, _ = self.extract_chord_symbol(chord, self.degrees_mapping[chord][0])
                
        # Reemplazar notación de sostenidos y bemoles para music21
        root = root.replace('s', '#').replace('b', '-')
        bass = bass.replace('s', '#').replace('b', '-') if bass != '' else ''
        
        # Mapear la calidad al formato de music21        
        try:
            m21_quality = ast.literal_eval(self.chord_mapping[quality])[0].replace("'", "") # Capturar el primer mapeo disponible y limpiar comillas

            final_chord_symbol = root + m21_quality
            final_chord_symbol += '/' + bass if bass else ''

            # Crear el objeto acorde de music21
            return harmony.ChordSymbol(final_chord_symbol)
    
        except Exception as e:
            print(f"Error parseando acorde '{chord_str}': {e}")
            return None
    
    def _analyze_key(self, chord_sequence):
        """
        Analiza la tonalidad de una secuencia de objetos Chord de music21.
        Utiliza el algoritmo Krumhansl-Schmuckler.
        """
        s = stream.Stream()
        for c in chord_sequence:
            pitches = c.pitches            
            chord_obj = chord.Chord(pitches)            
            s.append(chord_obj) 
            
        try:
            k = s.analyze('key')            
            return k.tonic.name, k.mode, k.correlationCoefficient
        
        except Exception as e:
            print(f"Error analizando tonalidad: {e}")
            return "Unknown"


    def _to_piano_roll(self, chord_obj):
        """Convierte un objeto Chord de music21 a una representación piano roll."""
        
        piano_roll = np.zeros(self.piano_roll_size, dtype=int)
        
        if chord_obj is not None:
            for p in chord_obj.pitches:
                midi_note = p.midi
                if self.min_midi_note <= midi_note < self.min_midi_note + self.piano_roll_size:
                    piano_roll[midi_note - self.min_midi_note] = 1
        return piano_roll

    def process(self):
        """
        Orquesta el proceso completo de carga, limpieza, parseo, secuenciación y codificación.
        """
        print("Cargando y limpiando datos...")
        
        if self.num_progressions is None:   # Cargar todo el dataset
            df = pd.read_csv(self.csv_path)
        else:
            df = pd.read_csv(self.csv_path, nrows=self.num_progressions) # carga solo las primeras N filas
        
        print(f"Total de progresiones cargadas: {df.shape[0]}")
        
        # Listas para almacenar todas las secuencias y metadatos
        all_sequences_chords = []
        all_sequences_pianoroll = []
        all_metadata = []        
        
        # Listas para batch
        batch_sequences_chords = []
        batch_sequences_pianoroll = []
        batch_metadata = []
        
        batch = 0
        batch_count = 0
        
        print("Procesando progresiones...")
        for index, row in tqdm(df.iterrows(), total=df.shape[0]):   # Iterar sobre cada fila del DataFrame      
            
            # Limpiar y obtener lista de acordes
            chord_list_str = self._clean_chord_string(row['chords'])
                        
            # Convertir toda la canción a objetos Chord de music21
            song_chords = [self._parse_single_chord(cs) for cs in chord_list_str]
            song_chords = [c for c in song_chords if c is not None] # Filtrar nulos            
            
            tonic, mode, coeff = self._analyze_key(song_chords)
            
            # Si la progresión es más corta que la longitud de secuencia, omitir
            if len(song_chords) < self.sequence_length:
                continue

            # Aplicar ventana deslizante para crear secuencias de longitud fija
            windows_amount = len(song_chords) - self.sequence_length + 1

            # Selección de inicios de ventana con tope configurable
            if windows_amount <= self.max_windows_per_progression:
                # Usa todas las ventanas disponibles (stride = 1)
                starts = list(range(0, windows_amount))
            else:
                # Calcula un stride dinámico para no exceder el máximo
                stride_dyn = math.ceil(windows_amount / self.max_windows_per_progression) # Si hay 25 ventanas y el máximo es 10, stride será 3
                starts = list(range(0, windows_amount, stride_dyn))
                # Limitar a 'max_windows_per_progression' y asegurar incluir la última ventana
                if len(starts) > self.max_windows_per_progression: # Starts puede ser mayor si stride no divide exactamente
                    starts = starts[:self.max_windows_per_progression]
                last_start = windows_amount - 1
                if starts and starts[-1] != last_start:
                    if len(starts) == self.max_windows_per_progression:
                        starts[-1] = last_start
                    else:
                        starts.append(last_start)
            
            for i in starts:
                
                batch_count += 1
                
                # Extraer secuencia
                sequence = song_chords[i:i + self.sequence_length]             
             
                # Convertir a piano roll
                pianoroll_sequence = [self._to_piano_roll(c) for c in sequence]
                
                # Almacenar resultados               
                all_sequences_chords.append([c.figure for c in sequence])
                all_sequences_pianoroll.append(np.array(pianoroll_sequence))
                
                # Extraer y almacenar metadatos
                genres = re.findall(r"'(.*?)'", row['genres']) if isinstance(row['genres'], str) else []      
                               
                all_metadata.append({
                    'original_id': row['id'],
                    'artist_id': row['artist_id'],
                    'song_id': row['spotify_song_id'],  
                    'key_tonic': tonic,  
                    'key_mode': mode,   
                    'key_correlation': coeff,
                    'main_genre': row['main_genre'],
                    'genres': genres,
                    'rock_genre': row['rock_genre'],
                    'release_date': row['release_date'],
                    'decade': row['decade']  
                })
                
                if batch_count == self.batch_size:
                    
                    start = batch_count*batch
                    batch_sequences_chords = all_sequences_chords[start:]
                    batch_sequences_pianoroll = all_sequences_pianoroll[start:]
                    batch_metadata = all_metadata[start:]
                    
                    batch += 1

                    processed_data = {
                        'piano_rolls': np.array(batch_sequences_pianoroll),
                        'chord_symbols': batch_sequences_chords,            
                        'metadata': pd.DataFrame(batch_metadata)
                    }

                    # pd.to_pickle(processed_data, f'{self.save_path}batch/dataset_01_{batch}.pkl')

                    batch_count = 0    
                    batch_sequences_chords = []
                    batch_sequences_pianoroll = []
                    batch_metadata = []

        processed_data = {
            'piano_rolls': np.array(all_sequences_pianoroll),
            'chord_symbols': all_sequences_chords,            
            'metadata': pd.DataFrame(all_metadata)
        }
        
        print(f"Procesamiento completo. Se generaron {len(processed_data['piano_rolls'])} secuencias.")
        
       # pd.to_pickle(processed_data, f'{self.save_path}dataset_01.pkl')
        
        return processed_data
        
if __name__ == '__main__':
    # --- Parámetros de configuración ---
    # NOTA: Asegúrate de que las rutas a tus archivos sean correctas.
    CHORDONOMICON_CSV_PATH = '/mnt/c/Users/nehem/OneDrive - Universidad de Chile/Universidad/6to año/Data/MIDI/preprocced/Chordomicon/chordonomicon_v2_filtrered.csv' # Ruta al dataset completo
    MIREX_MAPPING_PATH = '/mnt/c/Users/nehem/OneDrive - Universidad de Chile/Universidad/6to año/Data/MIDI/preprocced/Chordomicon/mirex_mapping_v2.csv' # Ruta al archivo de mapeo
    DEGREE_MAPPING_PATH = '/mnt/c/Users/nehem/OneDrive - Universidad de Chile/Universidad/6to año/Data/MIDI/preprocced/Chordomicon/chords_mapping.csv' 
    OUTPUT_PATH = '/mnt/c/Users/nehem/OneDrive - Universidad de Chile/Universidad/6to año/Data/MIDI/preprocced/Chordomicon/' # Ruta para guardar el dataset procesado
    
    NUM_PROGRESSIONS_TO_PROCESS = 100000 # Usar un número pequeño para pruebas rápidas
    SEQUENCE_LENGTH = 16
    STRIDE = 4
    BATCH_SIZE = 5000
    MAX_WINDOWS_PER_PROGRESSION = 10  # Nuevo parámetro para limitar ventanas por progresión
    
    # --- Ejecución del preprocesador ---
    preprocessor = ChordonomiconPreprocessor(
        csv_path=CHORDONOMICON_CSV_PATH,
        save_path=OUTPUT_PATH,
        chord_mapping_path=MIREX_MAPPING_PATH,
        degrees_mapping_path=DEGREE_MAPPING_PATH,
        num_progressions=NUM_PROGRESSIONS_TO_PROCESS,
        sequence_length=SEQUENCE_LENGTH,
        stride=STRIDE,
        batch_size=BATCH_SIZE,
        max_windows_per_progression=MAX_WINDOWS_PER_PROGRESSION
    )
    
    processed_data = preprocessor.process()

    # --- Verificación de los resultados ---
    print("\n--- Proceso completado ---")
    print(f"Dimensiones del array de piano rolls: {processed_data['piano_rolls'].shape}")

    index = np.random.randint(0, len(processed_data['chord_symbols']))
    print(processed_data['chord_symbols'][index])
    print(f"Metadatos asociados:\n{processed_data['metadata'].iloc[index]}")
    
    data = pd.DataFrame(processed_data['metadata'])
    data['chord_symbols'] = processed_data['chord_symbols']
    data['piano_rolls'] = list(processed_data['piano_rolls'])
