import pandas as pd
import numpy as np
import os
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from collections import Counter

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
    
    def _reconstruct_data_from_parquet(self, df_raw: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Reconstruye las estructuras de datos originales desde el formato parquet."""
        print(f"Reconstruyendo datos de {dataset_name}...")
        
        # Crear listas para los datos reconstruidos
        piano_rolls_reconstructed = []
        chord_symbols_reconstructed = []
        
        print(f"  Reconstruyendo piano rolls y chord symbols...")
        
        for idx, row in tqdm(df_raw.iterrows(), total=len(df_raw), desc=f"Procesando {dataset_name}"):
            # Reconstruir piano roll
            piano_roll_flat = row['piano_roll']
            if isinstance(piano_roll_flat, list):
                piano_roll = np.array(piano_roll_flat).reshape(
                    row['sequence_length'], 
                    row.get('piano_roll_size', row.get('piano_size', 84))  # Compatibilidad con ambos nombres
                )
            else:
                # Si ya es numpy array, mantenerlo
                piano_roll = piano_roll_flat
            
            piano_rolls_reconstructed.append(piano_roll)
            
            # Reconstruir chord_symbols desde JSON
            if isinstance(row['chord_symbols'], str):
                try:
                    chord_symbols = json.loads(row['chord_symbols'])
                    chord_symbols_reconstructed.append(chord_symbols)
                except json.JSONDecodeError:
                    print(f"Error decodificando chord_symbols en fila {idx}")
                    chord_symbols_reconstructed.append([])
            else:
                chord_symbols_reconstructed.append(row['chord_symbols'])
        
        # Crear DataFrame reconstruido
        df_reconstructed = df_raw.copy()
        df_reconstructed['piano_roll'] = piano_rolls_reconstructed
        df_reconstructed['chord_symbols'] = chord_symbols_reconstructed
        
        # Limpiar columnas auxiliares de parquet
        cols_to_drop = ['sequence_length', 'piano_roll_size', 'piano_size']
        for col in cols_to_drop:
            if col in df_reconstructed.columns:
                df_reconstructed.drop(columns=[col], inplace=True)
        
        print(f"  {dataset_name} reconstruido: {len(df_reconstructed)} secuencias")
        return df_reconstructed
    
    def _verify_data_integrity(self, df: pd.DataFrame, dataset_name: str):
        """Verifica la integridad de los datos cargados."""
        print(f"Verificando integridad de {dataset_name}...")
        
        # Verificar columnas esenciales
        required_cols = ['piano_roll', 'chord_symbols']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"  ⚠️ Columnas faltantes: {missing_cols}")
        
        # Verificar tipos de datos en una muestra
        sample_size = min(5, len(df))
        for i in range(sample_size):
            row = df.iloc[i]
            piano_roll = row['piano_roll']
            chord_symbols = row['chord_symbols']
            
            # Verificar piano_roll
            if not isinstance(piano_roll, np.ndarray):
                print(f"  ⚠️ Piano roll en fila {i} no es numpy array: {type(piano_roll)}")
            elif piano_roll.ndim != 2:
                print(f"  ⚠️ Piano roll en fila {i} no es 2D: shape {piano_roll.shape}")
            
            # Verificar chord_symbols
            if not isinstance(chord_symbols, list):
                print(f"  ⚠️ Chord symbols en fila {i} no es lista: {type(chord_symbols)}")
        
        print(f"  ✅ {dataset_name} verificado")
        return True
    
    def _create_cv_splits(self, df: pd.DataFrame, cv_method: str, n_splits: int):
        """Crea los splits de cross-validation."""
        print(f"Creando {n_splits} folds usando {cv_method} cross-validation...")
        
        if cv_method == 'stratified':
            # Usar tonalidad como estratificación (más relevante para música)
            if 'tonality' in df.columns:
                stratify_col = df['tonality']
                print("Estratificando por tonalidad...")
            elif 'genres' in df.columns:
                # Si no hay tonalidad, usar primer género
                stratify_col = df['genres'].str.split(',').str[0].fillna('Unknown')
                print("Estratificando por género principal...")
            else:
                print("⚠️  No se encontró columna para estratificar, usando KFold estándar")
                cv_method = 'standard'
                stratify_col = None
        else:
            stratify_col = None
        
        if cv_method == 'stratified' and stratify_col is not None:
            # Verificar que hay suficientes muestras por clase
            value_counts = stratify_col.value_counts()
            min_samples = value_counts.min()
            if min_samples < n_splits:
                print(f"⚠️  Algunas clases tienen menos de {n_splits} muestras, reduciendo a {min_samples} folds")
                n_splits = min(n_splits, min_samples)
            
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            splits = list(cv.split(df, stratify_col))
        else:
            # KFold estándar
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            splits = list(cv.split(df))
        
        # Crear folds
        folds = []
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            train_fold = df.iloc[train_idx].copy()
            val_fold = df.iloc[val_idx].copy()
            
            folds.append({
                'fold': fold_idx,
                'train': train_fold,
                'validation': val_fold
            })
            
            print(f"  Fold {fold_idx}: {len(train_fold)} train, {len(val_fold)} validation")
        
        return folds
    
    def _save_cv_splits(self, cv_folds: list, test_df: pd.DataFrame, finetune_df: pd.DataFrame, n_splits: int):
        """Guarda los splits de cross-validation."""
        print(f"\\n--- Resumen de Cross-Validation ({n_splits} folds) ---")
        
        # Crear directorio para CV
        cv_dir = os.path.join(self.output_dir, 'cross_validation')
        os.makedirs(cv_dir, exist_ok=True)
        
        # Guardar cada fold
        for fold_data in cv_folds:
            fold_idx = fold_data['fold']
            train_fold = fold_data['train']
            val_fold = fold_data['validation']
            
            print(f"Fold {fold_idx}: {len(train_fold)} train, {len(val_fold)} validation")
            
            # Guardar parquet
            train_fold.to_parquet(
                os.path.join(cv_dir, f'fold_{fold_idx}_train.parquet'), 
                compression='snappy'
            )
            val_fold.to_parquet(
                os.path.join(cv_dir, f'fold_{fold_idx}_validation.parquet'), 
                compression='snappy'
            )
            
            # Guardar pickle para compatibilidad
            train_fold.to_pickle(os.path.join(cv_dir, f'fold_{fold_idx}_train.pkl'))
            val_fold.to_pickle(os.path.join(cv_dir, f'fold_{fold_idx}_validation.pkl'))
        
        # Guardar test set (común para todos los folds)
        print(f"Test set: {len(test_df)} secuencias")
        test_df.to_parquet(os.path.join(self.output_dir, 'test_set.parquet'), compression='snappy')
        test_df.to_pickle(os.path.join(self.output_dir, 'test_set.pkl'))
        
        # Guardar dataset de fine-tuning
        print(f"Fine-tuning set: {len(finetune_df)} secuencias")
        finetune_df.to_parquet(os.path.join(self.output_dir, 'dataset_finetune_music.parquet'), compression='snappy')
        finetune_df.to_pickle(os.path.join(self.output_dir, 'dataset_finetune_music.pkl'))
        
        # Crear archivo de configuración de CV
        cv_config = {
            'n_splits': n_splits,
            'total_folds': len(cv_folds),
            'test_size': len(test_df),
            'finetune_size': len(finetune_df),
            'fold_sizes': [{'fold': f['fold'], 'train': len(f['train']), 'val': len(f['validation'])} 
                          for f in cv_folds]
        }
        
        import json
        with open(os.path.join(cv_dir, 'cv_config.json'), 'w') as f:
            json.dump(cv_config, f, indent=2)
        
        print(f"\\n✅ Cross-validation configurado en: {cv_dir}")
        return cv_folds
    
    def _save_standard_split(self, train_df: pd.DataFrame, dev_df: pd.DataFrame, 
                           test_df: pd.DataFrame, finetune_df: pd.DataFrame):
        """Guarda la división estándar train/dev/test."""
        print("\\n--- Resumen de Datasets Finales ---")
        print(f"Set de Entrenamiento (Train): {len(train_df)} secuencias")
        print(f"Set de Desarrollo (Dev/Validation): {len(dev_df)} secuencias")
        print(f"Set de Prueba (Test): {len(test_df)} secuencias")
        print(f"Dataset de Fine-Tuning (Video Game Music): {len(finetune_df)} secuencias")
        
        # Guardar en formato parquet
        print("Guardando datasets finales en formato parquet...")
        train_df.to_parquet(os.path.join(self.output_dir, 'train_set.parquet'), compression='snappy')
        dev_df.to_parquet(os.path.join(self.output_dir, 'dev_set.parquet'), compression='snappy')
        test_df.to_parquet(os.path.join(self.output_dir, 'test_set.parquet'), compression='snappy')
        finetune_df.to_parquet(os.path.join(self.output_dir, 'dataset_finetune_music.parquet'), compression='snappy')
        
        # Guardar en formato pickle para compatibilidad
        print("Guardando datasets finales en formato pickle para compatibilidad...")
        train_df.to_pickle(os.path.join(self.output_dir, 'train_set.pkl'))
        dev_df.to_pickle(os.path.join(self.output_dir, 'dev_set.pkl'))
        test_df.to_pickle(os.path.join(self.output_dir, 'test_set.pkl'))
        finetune_df.to_pickle(os.path.join(self.output_dir, 'dataset_finetune_music.pkl'))

    def load_data(self):
        """Carga los dos datasets procesados desde archivos .parquet."""
        print("Cargando datasets pre-procesados...")
        try:
            # Cargar archivos parquet
            print("Cargando Chordonomicon...")
            df_chordonomicon_raw = pd.read_parquet(self.chordonomicon_path)
            print("Cargando Popular Hook...")
            df_popular_hook_raw = pd.read_parquet(self.popular_hook_path)
            
            # Reconstruir estructuras de datos desde parquet
            print("Reconstruyendo estructuras de datos...")
            self.df_chordonomicon = self._reconstruct_data_from_parquet(df_chordonomicon_raw, "Chordonomicon")
            self.df_popular_hook = self._reconstruct_data_from_parquet(df_popular_hook_raw, "Popular Hook")
            
            # Verificar integridad de los datos
            self._verify_data_integrity(self.df_chordonomicon, "Chordonomicon")
            self._verify_data_integrity(self.df_popular_hook, "Popular Hook")

            print(f"Datos cargados: {len(self.df_chordonomicon)} secuencias de Chordonomicon, "
                  f"{len(self.df_popular_hook)} secuencias de Popular Hook.")
            
        except FileNotFoundError as e:
            print(f"Error: No se pudo encontrar un archivo .parquet. Asegúrate de que las rutas son correctas.")
            raise e

    def clean_data(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        """Realiza la limpieza de un DataFrame: elimina nulos y duplicados."""
        print(f"Limpiando el dataset '{name}'...")
        
        # Eliminar filas donde el piano_roll pueda ser nulo o vacío
        initial_rows = len(df)
        df_clean = df.dropna(subset=['piano_roll']).copy()
        
        # Verificar que piano_rolls sean arrays válidos
        valid_piano_rolls = []
        for idx, piano_roll in enumerate(df_clean['piano_roll']):
            try:
                if isinstance(piano_roll, np.ndarray) and piano_roll.size > 0:
                    valid_piano_rolls.append(True)
                elif isinstance(piano_roll, list) and len(piano_roll) > 0:
                    valid_piano_rolls.append(True)
                else:
                    valid_piano_rolls.append(False)
            except:
                valid_piano_rolls.append(False)
        
        df_clean = df_clean[valid_piano_rolls].reset_index(drop=True)
        
        # Eliminar duplicados exactos en las progresiones de piano roll
        # Para hacer esto, convertimos temporalmente el array a bytes para que sea "hashable"
        try:
            df_clean['piano_roll_hash'] = df_clean['piano_roll'].apply(
                lambda x: x.tobytes() if isinstance(x, np.ndarray) else np.array(x).tobytes()
            )
            df_clean.drop_duplicates(subset=['piano_roll_hash'], keep='first', inplace=True)
            df_clean.drop(columns=['piano_roll_hash'], inplace=True)
        except Exception as e:
            print(f"Advertencia: No se pudieron eliminar duplicados por piano_roll: {e}")
        
        final_rows = len(df_clean)
        
        print(f"Se eliminaron {initial_rows - final_rows} secuencias duplicadas o vacías. "
              f"Tamaño final: {final_rows} secuencias.")
        return df_clean

    def create_final_datasets(self, key_correlation_threshold=0.75, test_size=0.1, dev_size=0.1, cv_method='stratified', n_splits=5):
        """
        Ejecuta el pipeline completo de carga, limpieza, división y guardado.
        
        Args:
            key_correlation_threshold: Umbral para filtrar por confianza de tonalidad
            test_size: Proporción del conjunto de test
            dev_size: Proporción del conjunto de desarrollo
            cv_method: Método de cross-validation ('stratified', 'standard', 'none')
            n_splits: Número de folds para cross-validation
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
        print(f"Popular Hook dividido en {len(df_ph_other)} secuencias (otros géneros) y " 
              f"{len(df_ph_videogame)} secuencias (videojuegos).")
        
        
        # 5. Ensamblar los Datasets
        # Dataset principal: Chordonomicon (filtrado) + Popular Hook (no videojuegos)
        dataset_vae_main = pd.concat([df_chordonomicon_final, df_ph_other[COMMON_COLS]], ignore_index=True)
        
        # Dataset de fine-tuning: Solo videojuegos, conservando la emoción
        FINETUNE_COLS = COMMON_COLS + ['midi_emotion_predected']
        dataset_finetune_music = df_ph_videogame[FINETUNE_COLS]
        
        # 6. Dividir en Train/Dev/Test o Cross-Validation
        if cv_method == 'none':
            # División estándar train/dev/test
            print("Dividiendo el dataset principal en train/dev/test...")
            train_val_df, test_df = train_test_split(dataset_vae_main, test_size=test_size, random_state=42)
            
            # Ajustar el dev_size para que sea sobre el total original
            dev_size_adjusted = dev_size / (1 - test_size)
            train_df, dev_df = train_test_split(train_val_df, test_size=dev_size_adjusted, random_state=42)
            
            # Guardar división estándar
            self._save_standard_split(train_df, dev_df, test_df, dataset_finetune_music)
            
        else:
            # Configurar cross-validation
            print(f"Configurando {cv_method} cross-validation con {n_splits} folds...")
            
            # Separar test set primero (para evaluación final independiente)
            train_val_df, test_df = train_test_split(dataset_vae_main, test_size=test_size, random_state=42)
            
            # Configurar cross-validation en el conjunto train+validation
            cv_folds = self._create_cv_splits(train_val_df, cv_method, n_splits)
            
            # Guardar folds de cross-validation
            self._save_cv_splits(cv_folds, test_df, dataset_finetune_music, n_splits)
        
        print(f"\n¡Proceso completado! Archivos guardados en '{self.output_dir}'.")

if __name__ == '__main__':
    # --- Configuración ---
    # Directorio donde se encuentran los archivos .parquet concatenados
    CHORDONOMICON_PATH = '/home/neme/workspace/Data/MIDI/preprocced/Chordomicon/dataset_01_full.parquet'
    POPULARHOOK_PATH = '/home/neme/workspace/Data/MIDI/preprocced/Popular-hook/dataset_01_full.parquet'
    # Directorio donde se guardarán los datasets finales
    PROCESSED_DATA_DIR = '/home/neme/workspace/Data/MIDI/final_datasets/'
        
    # --- Configuración de Cross-Validation ---
    USE_CROSS_VALIDATION = True  # Cambiar a False para división estándar
    CV_METHOD = 'stratified'  # 'stratified', 'standard', o 'none'
    N_SPLITS = 5  # Número de folds
    
    # --- Ejecución ---
    processor = FinalDatasetProcessor(
        chordonomicon_path=CHORDONOMICON_PATH,
        popular_hook_path=POPULARHOOK_PATH,
        output_dir=PROCESSED_DATA_DIR
    )
    
    if USE_CROSS_VALIDATION:
        processor.create_final_datasets(
            cv_method=CV_METHOD,
            n_splits=N_SPLITS,
            test_size=0.15  # Un poco más grande para test final
        )
    else:
        processor.create_final_datasets(cv_method='none')