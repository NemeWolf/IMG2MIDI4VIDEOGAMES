"""
Resumable preprocessor for the Chordonomicon dataset.

This script is a non-destructive variant of `preproccesing_chordonomicon.py` that:
- Keeps the original file untouched.
- Limits windows per progression via `max_windows_per_progression` with dynamic stride.
- Fixes batching (writes exactly `batch_size` samples per file, without re-slicing the full arrays).
- Supports resume-from-checkpoint by original_id and number of windows already saved for that row.

Outputs batches to: <save_path>/batch/dataset_01_<n>.pkl

Adjust the constants in the __main__ section as needed.
"""

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
    """Preprocessor with dynamic window limiting and resume support."""

    def __init__(
        self,
        csv_path: str,
        save_path: str,
        chord_mapping_path: str,
        degrees_mapping_path: str,
        num_progressions: Optional[int] = 100000,
        sequence_length: int = 16,
        batch_size: int = 5000,
        max_windows_per_progression: int = 10,
        resume_from_original_id: Optional[int] = None,
        resume_windows_done: int = 0,
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
        df = pd.read_csv(mapping_path)
        return dict(
            zip(
                df['Original Symbol'].apply(lambda x: x.replace('"', '')),
                df['ChordSymbol_m21'].apply(lambda x: x.replace('"', '')),
            )
        )

    def _load_degrees_mapping(self, degrees_path: str) -> Dict[str, List[str]]:
        df = pd.read_csv(degrees_path)
        return dict(zip(df['Chords'], df['Notes'].apply(ast.literal_eval)))

    # ------------------------ Utilities ------------------------
    def _clean_chord_string(self, chords_text: str) -> List[str]:
        cleaned_text = re.sub(r'<[^>]+>', '', chords_text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        return cleaned_text.split(' ')

    def _extract_chord_symbol(self, chord_txt: str, note: str) -> Tuple[str, str, str]:
        tonic = ''
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
        symbol = chord_txt[len(tonic):]
        return tonic, symbol, chord_txt

    def _parse_single_chord(self, chord_str: str) -> Optional[harmony.ChordSymbol]:
        # Split possible bass (slash chord)
        parts = chord_str.split('/')
        chord_txt = parts[0]
        bass_txt = parts[1] if len(parts) > 1 else ''

        try:
            root_note, quality_txt, _ = self._extract_chord_symbol(
                chord_txt, self.degrees_mapping[chord_txt][0]
            )
        except Exception:
            return None

        # Adapt notation for music21
        root_note = root_note.replace('s', '#').replace('b', '-')
        bass_txt = bass_txt.replace('s', '#').replace('b', '-') if bass_txt else ''

        try:
            m21_quality = ast.literal_eval(self.chord_mapping[quality_txt])[0].replace("'", '')
            final_symbol = root_note + m21_quality
            if bass_txt:
                final_symbol += '/' + bass_txt
            return harmony.ChordSymbol(final_symbol)
        except Exception:
            return None

    def _analyze_key(self, chord_sequence: List[harmony.ChordSymbol]) -> Tuple[str, str, float]:
        s = stream.Stream()
        for cobj in chord_sequence:
            if cobj is None:
                continue
            s.append(chord.Chord(cobj.pitches))
        try:
            k = s.analyze('key')
            return k.tonic.name, k.mode, k.correlationCoefficient
        except Exception:
            return 'Unknown', 'unknown', float('nan')

    def _to_piano_roll(self, chord_obj: harmony.ChordSymbol) -> np.ndarray:
        roll = np.zeros(self.piano_roll_size, dtype=int)
        if chord_obj is not None:
            for p in chord_obj.pitches:
                midi_note = p.midi
                if self.min_midi_note <= midi_note < self.min_midi_note + self.piano_roll_size:
                    roll[midi_note - self.min_midi_note] = 1
        return roll

    # ------------------------ Resume helpers ------------------------
    def _next_existing_batch_index(self) -> int:
        batch_dir = os.path.join(self.save_path, 'batch')
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

    # ------------------------ Main processing ------------------------
    def process(self) -> Dict[str, Any]:
        print('Cargando y limpiando datos...')
        if self.num_progressions is None:
            df = pd.read_csv(self.csv_path)
        else:
            df = pd.read_csv(self.csv_path, nrows=self.num_progressions)

        print(f'Total de progresiones cargadas: {df.shape[0]}')

        # Per-batch buffers
        batch_chords: List[List[str]] = []
        batch_rolls: List[np.ndarray] = []
        batch_meta: List[Dict[str, Any]] = []
        batch_count = 0

        # Continue numbering from last existing batch on disk
        current_batch_idx = self._next_existing_batch_index()

        # Resume flags
        resuming = self.resume_from_original_id is not None
        reached_resume_row = False
        resume_skip_windows = int(self.resume_windows_done) if resuming else 0

        print('Procesando progresiones...')
        for _, row in tqdm(df.iterrows(), total=df.shape[0]):
            # If resuming, skip rows until we hit the target original_id (stored under 'id' in CSV)
            if resuming and not reached_resume_row:
                if row.get('id') != self.resume_from_original_id:
                    continue
                reached_resume_row = True

            # Clean chord text and parse
            chord_tokens = self._clean_chord_string(row['chords'])
            song_chords = [self._parse_single_chord(cs) for cs in chord_tokens]
            song_chords = [c for c in song_chords if c is not None]

            if len(song_chords) < self.sequence_length:
                if resuming and reached_resume_row:
                    resuming = False  # resume done even if sequence is too short
                continue

            windows_amount = len(song_chords) - self.sequence_length + 1

            # Build starts limited by max_windows_per_progression
            if windows_amount <= self.max_windows_per_progression:
                starts = list(range(0, windows_amount))
            else:
                stride_dyn = math.ceil(windows_amount / self.max_windows_per_progression)
                starts = list(range(0, windows_amount, stride_dyn))
                if len(starts) > self.max_windows_per_progression:
                    starts = starts[: self.max_windows_per_progression]
                last_start = windows_amount - 1
                if starts and starts[-1] != last_start:
                    if len(starts) == self.max_windows_per_progression:
                        starts[-1] = last_start
                    else:
                        starts.append(last_start)

            # If resuming on this row, drop first N windows already saved
            if resuming and reached_resume_row:
                if resume_skip_windows > 0:
                    starts = starts[resume_skip_windows:]
                resuming = False  # from now on, continue normally

            if not starts:
                continue

            # Compute key once (optional heuristic: first window)
            tonic, mode, coeff = self._analyze_key(song_chords[: self.sequence_length])

            for i in starts:
                seq = song_chords[i : i + self.sequence_length]
                pr_seq = [self._to_piano_roll(c) for c in seq]

                batch_chords.append([c.figure for c in seq])
                batch_rolls.append(np.array(pr_seq))

                genres = re.findall(r"'(.*?)'", row['genres']) if isinstance(row['genres'], str) else []
                batch_meta.append(
                    {
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

                batch_count += 1
                
                if batch_count == self.batch_size:
                    current_batch_idx += 1
                    out = {
                        'piano_rolls': np.array(batch_rolls),
                        'chord_symbols': batch_chords,
                        'metadata': pd.DataFrame(batch_meta),
                    }
                    out_path = os.path.join(self.save_path, 'batch', f'dataset_01_{current_batch_idx}.pkl')
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    pd.to_pickle(out, out_path)

                    # reset buffers
                    batch_chords, batch_rolls, batch_meta = [], [], []
                    batch_count = 0

        # Flush any remaining partial batch
        if batch_count > 0:
            current_batch_idx += 1
            out = {
                'piano_rolls': np.array(batch_rolls),
                'chord_symbols': batch_chords,
                'metadata': pd.DataFrame(batch_meta),
            }
            out_path = os.path.join(self.save_path, 'batch', f'dataset_01_{current_batch_idx}.pkl')
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            pd.to_pickle(out, out_path)

        print(f'Procesamiento completado. Batches escritos hasta índice {current_batch_idx}.')
        # Return last batch content for quick inspection
        return {
            'piano_rolls': np.array(batch_rolls),
            'chord_symbols': batch_chords,
            'metadata': pd.DataFrame(batch_meta),
        }


if __name__ == '__main__':
    # --- Configuración ---
    CHORDONOMICON_CSV_PATH = '/mnt/c/Users/nehem/OneDrive - Universidad de Chile/Universidad/6to año/Data/MIDI/preprocced/Chordomicon/chordonomicon_v2_filtrered.csv'
    MIREX_MAPPING_PATH = '/mnt/c/Users/nehem/OneDrive - Universidad de Chile/Universidad/6to año/Data/MIDI/preprocced/Chordomicon/mirex_mapping_v2.csv'
    DEGREE_MAPPING_PATH = '/mnt/c/Users/nehem/OneDrive - Universidad de Chile/Universidad/6to año/Data/MIDI/preprocced/Chordomicon/chords_mapping.csv'
    OUTPUT_PATH = '/mnt/c/Users/nehem/OneDrive - Universidad de Chile/Universidad/6to año/Data/MIDI/preprocced/Chordomicon/'

    NUM_PROGRESSIONS_TO_PROCESS = 100000
    SEQUENCE_LENGTH = 16
    BATCH_SIZE = 5000
    MAX_WINDOWS_PER_PROGRESSION = 10

    # Reanudación concreta para el caso reportado
    RESUME_FROM_ORIGINAL_ID = 67407  # último original_id visto
    RESUME_WINDOWS_DONE = 2          # ya guardadas 2 ventanas de esa progresión

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
    print('\n--- Último batch (parcial) ---')
    print(f"piano_rolls: {result['piano_rolls'].shape}")
    if len(result['chord_symbols']) > 0:
        print(result['chord_symbols'][0])
        print(result['metadata'].head(1))
