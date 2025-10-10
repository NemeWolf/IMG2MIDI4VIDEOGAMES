# Reconstruir datos
import pandas as pd
import numpy as np
from tqdm import tqdm

def _reconstruct_data_from_parquet(df_raw: pd.DataFrame) -> pd.DataFrame:

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