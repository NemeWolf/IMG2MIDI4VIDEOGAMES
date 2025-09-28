import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable
from music21 import stream, chord, note, duration
import numpy as np

# Custom metrics
def BCE(y_true, y_pred):
    return tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
    
def MAE(y_true_duration, y_pred_duration):
    return tf.reduce_mean(tf.abs(y_true_duration - y_pred_duration))

def MSE(y_true_notes, y_pred_notes):
    return tf.reduce_mean(tf.square(y_true_notes - y_pred_notes))
# =================================================================================================
# Custom metrics and loss functions
# =================================================================================================
# kl_loss calculation -------------------------------------------------------------------------------------
@register_keras_serializable(package='Custom', name='_kl_loss')
@tf.function
def _kl_loss(mu, log_var):         
    log_var = tf.clip_by_value(log_var, -5.0, 5.0) # --> to cnn
    kl_loss = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mu) - tf.exp(log_var))
    return kl_loss
    
# reconstruction_loss calculation for LSTM ----------------------------------------------------------------
@register_keras_serializable(package='Custom', name='_reconstruction_loss_lstm')
@tf.function
def _reconstruction_loss_lstm(y_true, y_pred):
    return BCE(y_true, y_pred)
  
# reconstruction_loss calculation for CNN ----------------------------------------------------------------    
@register_keras_serializable(package='Custom', name='_reconstruction_loss_cnn')
def _reconstruction_loss_cnn(y_true, y_pred): 
    #y_true = tf.clip_by_value(y_true, 0.0, 1.0) 
    #y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    ssim = tf.reduce_mean(1 - tf.image.ssim(y_true, y_pred, max_val=1.0))
    return 0.7 * mse + 0.3 * ssim
    #return mse 

def harmonic_progressions_to_midi(progression, original, output_file, _save=False, threshold=0.5):
    """
    Convierte progresiones armónicas codificadas en archivos MIDI usando music21.

    :param progressions: Lista de progresiones armónicas. Cada progresión es una lista de acordes.
                         Cada acorde es una lista de 13 elementos (12 notas + duración).
    :param output_file: Nombre del archivo MIDI de salida.
    """     
    #error = _reconstruction_loss_lstm(progression, original)       
    
    if original is None:
        original = progression
    
    count=0
    for chords, original_chords in zip(progression, original):      
                
        # Si la progresión tiene duración, extraerla
        if progression.shape[2] == 13:
            _chords = chords[:, :-1]
            duration_chords = chords[:, -1]   
                    
            _original_chords = original_chords[:, :-1]
            duration_original = original_chords[:, -1]
            
            # desnormalizar duracion
            duration_max = 16
            duration_min = 0
            
            duration_chords = duration_chords * (duration_max - duration_min) + duration_min
            duration_original = duration_original * (duration_max - duration_min) + duration_min
        
        elif progression.shape[2] == 12:
            _chords = chords
            _original_chords = original_chords 
            
        # Codificar progresion generada -----------------------------------------
        s = stream.Stream()        
        s_count = 0
        for _chord in _chords:            
            # Crear acorde
            chord_notes = []        
            for i, note_present in enumerate(_chord):
                if note_present > 0.5: # --> sigmoid
                #if note_present > 0.125: # --> softmax
                    chord_notes.append(note.Note(60 + i))  # 60 es el MIDI number para C4                    
                
            c = chord.Chord(chord_notes)
            
            # Asignar duracion al acorde
            if progression.shape[2] == 13:                
                duration_chord = duration_chords[s_count]                
                if duration_chord == 0:
                    duration_chord = 4.0                    
                c.duration = duration.Duration(duration_chord)                
            elif progression.shape[2] == 12:
                c.duration = duration.Duration(4.0)      
            s.append(c)                
            s_count += 1
        
        # Codificar progresion original -----------------------------------------
        o = stream.Stream()
        o_count = 0
        for _chord in _original_chords:
            # Crear acorde 
            chord_notes = []        
            for i, note_present in enumerate(_chord):
                if note_present == 1 : # --> softmax
                    chord_notes.append(note.Note(60 + i))
            c = chord.Chord(chord_notes)
            
            # Asignar duracion al acorde
            if progression.shape[2] == 13:
                duration_chord = duration_original[o_count]
                if duration_chord == 0:
                    duration_chord = 4.0    
                c.duration = duration.Duration(duration_chord)
            elif progression.shape[2] == 12:
                c.duration = duration.Duration(4.0)
            o.append(c)
            o_count += 1
        
        preogresion_file = output_file + '\\' + str(count) + '_reconstructed_progression.mid'
        original_file = output_file + '\\' + str(count) + '_original_progression.mid'
        
        count += 1
        
        if _save:
            # Guardar archivos MIDI
            s.write('midi', fp=preogresion_file)
            # o.write('midi', fp=original_file)        
            # print(f"Archivo MIDI guardado en {preogresion_file}")
            # print(f"Archivo MIDI guardado en {original_file}")
        
       # print(f"Perdida de reconstruccion: {error}")

def most_common_progression_duration(progressions):
    progression_durations = []
    for progression in progressions:
        total_duration = sum(chord_data[12] for chord_data in progression)
        progression_durations.append(total_duration)
    
    # Convertir a numpy array para usar np.unique con return_counts
    progression_durations = np.array(progression_durations)
    unique_durations, counts = np.unique(progression_durations, return_counts=True)
    
    # Encontrar la duración más común
    most_common = unique_durations[np.argmax(counts)]
    
    return most_common

def calculate_percent_succes_notes(original,generate): 
    # Calcular porcentaje de aciertos en notas
    count = 0
    total_count_notes = 0 
    for i in range(original.shape[0]):        
        for j in range(original.shape[1]):            
            for k in range(original.shape[2]):
                if original[i,j,k] == generate[i,j,k]:
                    count += 1    
                total_count_notes += 1    
    percent_notes = (count/total_count_notes)*100
    
    # Calcular porcentaje de aciertos en acordes
    success_count = 0
    total_count_chords = 0
    for i in range(original.shape[0]):        
        for j in range(original.shape[1]):            
            if np.array_equal(original[i][j], generate[i][j]):
                success_count += 1
            total_count_chords += 1            
    percent_chords = (success_count / total_count_chords) * 100
    
    # Imprimir resultados
    print(f"Porcentaje de aciertos en notas: {percent_notes}")
    print(f"Porcentaje de aciertos en acordes: {percent_chords}")

def calculate_duration_error(progression, original):
    # Calcular error en duraciones
    total_error = 0
    for i in range(progression.shape[0]):
        for j in range(progression.shape[1]):
            error = abs(progression[i][j][-1] - original[i][j][-1])
            total_error += error
    
    error= total_error/(progression.shape[0]*progression.shape[1])
    print(f"Error en duraciones: {error}")