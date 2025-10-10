from __future__ import annotations

import os
import re
import glob
import math
import ast
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple, Optional, Dict, Any
import json 

from music21 import stream, chord, key as m21key, pitch, harmony, analysis as m21analysis


# Note name mappings (latin -> american), mirrors the original script
NOTES_AMERICAN = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
NOTES_AMERICAN_SHARP = ['Cs', 'Ds', 'Es', 'Fs', 'Gs', 'As', 'Bs']
NOTES_AMERICAN_FLAT = ['Cb', 'Db', 'Eb', 'Fb', 'Gb', 'Ab', 'Bb']

NOTES_LATIN = ['do', 're', 'mi', 'fa', 'sol', 'la', 'si']
NOTES_LATIN_SHARP = ['dos', 'res', 'mis', 'fas', 'sols', 'las', 'sis']
NOTES_LATIN_FLAT = ['dob', 'reb', 'mib', 'fab', 'solb', 'lab', 'sib']

NOTES = dict(zip(NOTES_LATIN, NOTES_AMERICAN))
NOTES_SHARP = dict(zip(NOTES_LATIN_SHARP, NOTES_AMERICAN_SHARP))
NOTES_FLAT = dict(zip(NOTES_LATIN_FLAT, NOTES_AMERICAN_FLAT))


class ChordonomiconPreprocessorResumable:
    """Preprocesa el dataset Chordonomicon en batches, con capacidad de reanudación."""

    def __init__(self,
                csv_path: str,  # Ruta al CSV de Chordonomicon
                save_path: str, # Carpeta donde guardar los batches
                chord_mapping_path: str, # Ruta al CSV de mapeo de acordes
                degrees_mapping_path: str, # Ruta al CSV de mapeo de grados
                num_progressions: Optional[int] = 100000, # Número de progresiones a procesar (None = todas)
                sequence_length: int = 16, # Longitud de secuencia de acordes
                batch_size: int = 5000, # Número de secuencias por batch
                max_windows_per_progression: int = 10, # Máximo de ventanas por progresión
                resume_from_original_id: Optional[int] = None, # original_id para reanudar (None = desde el inicio)
                resume_windows_done: int = 0, # Número de ventanas ya procesadas en esa progresión
            ) -> None:
        
        self.csv_path = csv_path
        self.save_path = save_path
        self.num_progressions = num_progressions
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.max_windows_per_progression = max_windows_per_progression

        # Resume parameters
        self.resume_from_original_id = resume_from_original_id
        self.resume_windows_done = int(resume_windows_done or 0)

        # Load mappings
        self.chord_mapping = self._load_chord_mapping(chord_mapping_path)
        self.degrees_mapping = self._load_degrees_mapping(degrees_mapping_path)

        # Piano roll configuration (to match original)
        self.piano_roll_size = 84
        self.min_midi_note = 24

    # ------------------------ Loaders ------------------------
    def _load_chord_mapping(self, mapping_path: str) -> Dict[str, str]:
        """
        Carga el mapeo de acordes desde un CSV para notacion original a ChordSymbol compatible con music21.
        args:
            mapping_path: Ruta al archivo CSV con las columnas 'Original Symbol' y 'ChordSymbol_m21'.
        returns:
            Un diccionario que mapea símbolos originales a símbolos compatibles con music21.        
        """
        df = pd.read_csv(mapping_path)
        return dict(
            zip(
                df['Original Symbol'].apply(lambda x: x.replace('"', '')),
                df['ChordSymbol_m21'].apply(lambda x: x.replace('"', '')),
            )
        )

    def _load_degrees_mapping(self, degrees_path: str) -> Dict[str, List[str]]:
        """ 
        Carga el mapeo de grados desde un CSV de cada tipo de acorde presente en el dataset. 
        args:
            degrees_path: Ruta al archivo CSV con las columnas 'Chords' y 'Notes'.
        returns:
            Un diccionario que mapea símbolos de acordes a listas de notas (grados).
        """
        df = pd.read_csv(degrees_path)
        return dict(zip(df['Chords'], df['Notes'].apply(ast.literal_eval)))

    # ------------------------ Utilities ------------------------
    def _clean_chord_string(self, chords_text: str) -> List[str]:
        """
        Limpia y tokeniza una cadena de acordes.
        """
        cleaned_text = re.sub(r'<[^>]+>', '', chords_text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        return cleaned_text.split(' ')

    def _extract_chord_symbol(self, chord_txt: str, note: str) -> Tuple[str, str, str]:
        """
        Extrae el símbolo de tónica y calidad de un símbolo de acorde.
        Dada la nota raiz, captura la tónica y el resto como calidad.
        args:
            chord_txt: Símbolo completo del acorde (e.g., 'doM7').
            note: Nota raíz en notación latina (e.g., 'do').    
        returns:
            Una tupla (tónica, calidad, símbolo original).
        """        
        tonic = ''
        
        # Detectar tónica en notación latina y convertir a americana
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
                
        # Extraer calidad
        symbol = chord_txt[len(tonic):]
        return tonic, symbol, chord_txt

    def _parse_single_chord(self, chord_str: str) -> Optional[harmony.ChordSymbol]:
        """
        Parsea un símbolo de acorde individual a un objeto music21 ChordSymbol.
        args:
            chord_str: Símbolo del acorde (e.g., 'doM7', 'solm7/b').
        returns:
            Un objeto harmony.ChordSymbol o None si no se puede parsear.
        """
        
       # Separa baja si existe
        parts = chord_str.split('/')
        chord_txt = parts[0]
        bass_txt = parts[1] if len(parts) > 1 else ''

        # Extrae tónica y calidad
        try:
            root_note, quality_txt, _ = self._extract_chord_symbol(
                chord_txt, self.degrees_mapping[chord_txt][0]
            )
        except Exception:
            return None

        # Adaptar notación para music21
        root_note = root_note.replace('s', '#').replace('b', '-')
        bass_txt = bass_txt.replace('s', '#').replace('b', '-') if bass_txt else ''

        # Mapear calidad usando el diccionario de mapeo cargado
        try:
            m21_quality = ast.literal_eval(self.chord_mapping[quality_txt])[0].replace("'", '')
            final_symbol = root_note + m21_quality  # Reconstruir símbolo final
            if bass_txt:
                final_symbol += '/' + bass_txt  # Añadir bajo si existe
                
            # Crear objeto ChordSymbol de music21
            return harmony.ChordSymbol(final_symbol)
        except Exception:
            return None

    def _analyze_key(self, chord_sequence: List[harmony.ChordSymbol]) -> Tuple[str, str, float]:
        """
        Analiza la tonalidad de una secuencia de acordes usando music21.
        args:
            chord_sequence: Lista de objetos harmony.ChordSymbol.
        returns:
            Una tupla (tónica, modo, coeficiente de correlación).
        """
        # Construir un stream temporal
        s = stream.Stream()

        # El metodo analyze tiene mejores resultados con objetos music21.chord.Chord que con 
        # music21.harmony.ChordSymbol, por lo que se convierten a music21.chord.Chord antes de analizar.
        
        # Añadir acordes al stream
        for cobj in chord_sequence:
            if cobj is None:
                continue
            s.append(chord.Chord(cobj.pitches))          
            
        try:
            # Analizar la tonalidad
            k = s.analyze('key')
            return k.tonic.name, k.mode, k.correlationCoefficient
        except Exception:
            return 'Unknown', 'unknown', float('nan')

    def _to_piano_roll(self, chord_obj: harmony.ChordSymbol) -> np.ndarray:
        """
        Convierte un objeto harmony.ChordSymbol a una representación de piano roll binaria.
        args:
            chord_obj: Un objeto harmony.ChordSymbol.
        returns:
            Un array numpy de tamaño (84,) con 1s en las posiciones de las notas presentes.
        """
        
        roll = np.zeros(self.piano_roll_size, dtype=int)    # Inicializar piano roll vacío
        
        if chord_obj is not None:
            for p in chord_obj.pitches:  # Iterar sobre las notas del acorde
                midi_note = p.midi 
                # Mapear a rango 24-107 (84 notas)
                if self.min_midi_note <= midi_note < self.min_midi_note + self.piano_roll_size:
                    roll[midi_note - self.min_midi_note] = 1
        return roll

    # ------------------------ Resume helpers ------------------------
    def _next_existing_batch_index(self) -> int:
        """
        Encuentra el índice del último batch existente en disco para continuar la numeración.
        returns:
            El índice del último batch existente, o 0 si no hay batches.
        """
        # Buscar archivos batch existentes
        batch_dir = os.path.join(self.save_path, 'batch')
        os.makedirs(batch_dir, exist_ok=True)
        files = glob.glob(os.path.join(batch_dir, 'dataset_01_*.parquet'))
        
        # Extraer índices y encontrar el máximo
        if not files:
            return 0    # Si no hay archivos, empezar desde 0
        nums = []

        # Extraer números de los nombres de archivo
        for p in files:
            m = re.search(r'_(\d+)\.parquet$', os.path.basename(p))
            if m:
                nums.append(int(m.group(1)))
        return max(nums) if nums else 0

    # ------------------------ Main processing ------------------------
    def process(self) -> Dict[str, Any]:
        """
        Procesa el dataset en batches, con capacidad de reanudación.
        returns:
            Un diccionario con el contenido del último batch procesado (parcial).
        """
        
        print('Cargando y limpiando datos...')
        
        # CARGAR DATOS ============================
        if self.num_progressions is None:
            df = pd.read_csv(self.csv_path)
        else:
            df = pd.read_csv(self.csv_path, nrows=self.num_progressions)

        print(f'Total de progresiones cargadas: {df.shape[0]}')

        # Buffer para batch
        batch = []       
        
        batch_count = 0

        # Índice de batch actual (para nombrar archivos)
        current_batch_idx = self._next_existing_batch_index()
        
        # Flags de reanudación
        resuming = self.resume_from_original_id is not None # Si se debe reanudar
        reached_resume_row = False
        resume_skip_windows = int(self.resume_windows_done) if resuming else 0  # Ventanas a saltar en la progresión de reanudación

        # PROCESAMIENTO PRINCIPAL ============================
        
        print('Procesando progresiones...')
        for _, row in tqdm(df.iterrows(), total=df.shape[0]):
            # Si se está reanudando, saltar filas hasta alcanzar el original_id objetivo (almacenado en 'id' en CSV)
            if resuming and not reached_resume_row:
                if row.get('id') != self.resume_from_original_id:
                    continue
                reached_resume_row = True

            # Limpiar y parsear acordes
            chord_tokens = self._clean_chord_string(row['chords'])
            song_chords = [self._parse_single_chord(cs) for cs in chord_tokens]
            
            # Filtrar acordes no parseados (None)
            song_chords = [c for c in song_chords if c is not None]

            # Si la progresión es demasiado corta, saltarla
            if len(song_chords) < self.sequence_length:               
                # Si se está reanudando y se alcanzó la fila objetivo, desactivar reanudación y continuar normalmente
                if resuming and reached_resume_row:
                    resuming = False
                continue
            
            # ENVENTANADO DE LA PROGRESIÓN ============================
            
            # Calcular número de ventanas posibles
            windows_amount = len(song_chords) - self.sequence_length + 1

            # Construir inicios limitados por max_windows_per_progression
            if windows_amount <= self.max_windows_per_progression:
                starts = list(range(0, windows_amount))
            else:
                # Si hay más ventanas que el máximo, calculamos un stride en base al máximo permitido                
                stride_dyn = math.ceil(windows_amount / self.max_windows_per_progression)
                starts = list(range(0, windows_amount, stride_dyn))
                
                # Nos aseguramos de no exceder el máximo
                if len(starts) > self.max_windows_per_progression:
                    starts = starts[: self.max_windows_per_progression]
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
                resuming = False  # Continuar normalmente después de esta fila

            if not starts:
                continue

            # Calcular tonalidad una vez (heurística opcional: primera ventana)
            tonic, mode, coeff = self._analyze_key(song_chords[: self.sequence_length])

            # PROCESAMIENTO DE VENTANAS ============================
            for i in starts:                
                seq = song_chords[i : i + self.sequence_length] # Secuencia de acordes actual
                pr_seq = [self._to_piano_roll(c) for c in seq]  # Secuencia de piano rolls
                chord_symbol_sequence = [c.figure for c in seq] # Secuencia de símbolos de acordes

                # Aplanar piano rolls 
                pr_seq_flat = np.array(pr_seq).flatten().tolist()
                
                # Convertir chord_symbols a JSON string
                chord_symbol_json = json.dumps(chord_symbol_sequence)

                # Metadata
                genres = re.findall(r"'(.*?)'", row['genres']) if isinstance(row['genres'], str) else [] # Extraer lista de géneros
    
                # Añadir al batch
                batch.append(
                    {
                        'chord_symbols': chord_symbol_json,
                        'piano_rolls': pr_seq_flat,
                        'sequence_length': self.sequence_length,
                        'piano_roll_size': self.piano_roll_size,
                        'original_id': row['id'],
                        'artist_id': row.get('artist_id'),
                        'song_id': row.get('spotify_song_id'),
                        'key_tonic': tonic,
                        'key_mode': mode,
                        'key_correlation': coeff,
                        'main_genre': row.get('main_genre'),
                        'genres': genres,
                        'rock_genre': row.get('rock_genre'),
                        'release_date': row.get('release_date'),
                        'decade': row.get('decade'),
                    }
                )
                
                batch_count += 1 # Incrementar contador de batch
                
                # Si el batch está lleno, guardarlo
                if batch_count == self.batch_size:
                    current_batch_idx += 1 # Incrementar índice de batch
                    
                    df_rows = pd.DataFrame(batch)                   
                   
                    out_path = os.path.join(self.save_path, 'batch', f'dataset_01_{current_batch_idx}.parquet')
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    
                    df_rows.to_parquet(out_path, compression='snappy')
                    
                    # Limpieza
                    del df_rows
                    del batch[:]
                    
                    # Resetear buffers
                    batch = []
                    batch_count = 0

        # Guardar cualquier batch parcial restante
        if batch_count > 0:
            current_batch_idx += 1
            df_row = pd.DataFrame(batch)
            
            out_path = os.path.join(self.save_path, 'batch', f'dataset_01_{current_batch_idx}.parquet')
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            df_row.to_parquet(out_path, compression='snappy')

        print(f'Procesamiento completado. Batches escritos hasta índice {current_batch_idx}.')
        # Retornar contenido del último batch para inspección rápida
        return {}

if __name__ == '__main__':
    # --- Configuración ---
    CHORDONOMICON_CSV_PATH = '/home/neme/workspace/Data/MIDI/preprocced/Chordomicon/chordonomicon_v2_filtrered.csv'
    MIREX_MAPPING_PATH = '/home/neme/workspace/Data/MIDI/preprocced/Chordomicon/mirex_mapping_v2.csv'
    DEGREE_MAPPING_PATH = '/home/neme/workspace/Data/MIDI/preprocced/Chordomicon/chords_mapping.csv'
    OUTPUT_PATH = '/home/neme/workspace/Data/MIDI/preprocced/Chordomicon/'

    NUM_PROGRESSIONS_TO_PROCESS = 100000
    SEQUENCE_LENGTH = 16
    BATCH_SIZE = 5000
    MAX_WINDOWS_PER_PROGRESSION = 10

    # Reanudación concreta para el caso reportado
    RESUME_FROM_ORIGINAL_ID = None # último original_id visto (None = desde el inicio)
    RESUME_WINDOWS_DONE = 0          # ya guardadas 2 ventanas de esa progresión

    preprocessor = ChordonomiconPreprocessorResumable(
        csv_path=CHORDONOMICON_CSV_PATH,
        save_path=OUTPUT_PATH,
        chord_mapping_path=MIREX_MAPPING_PATH,
        degrees_mapping_path=DEGREE_MAPPING_PATH,
        num_progressions=NUM_PROGRESSIONS_TO_PROCESS,
        sequence_length=SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE,
        max_windows_per_progression=MAX_WINDOWS_PER_PROGRESSION,
        resume_from_original_id=RESUME_FROM_ORIGINAL_ID,
        resume_windows_done=RESUME_WINDOWS_DONE,
    )

    result = preprocessor.process()
