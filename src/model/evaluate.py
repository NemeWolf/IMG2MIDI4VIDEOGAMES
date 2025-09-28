from models.lstm_vae_model import Sampling, VAELossLayer
from utils.metrics import _reconstruction_loss_lstm, _kl_loss, _reconstruction_loss_cnn, harmonic_progressions_to_midi, calculate_percent_succes_notes, calculate_duration_error
from sklearn.model_selection import train_test_split 

import os
import pickle
import json

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import load_model

# Models paths
MODEL_PATH_LSTM = 'deepGame4Music\\models\\LSTM_VAE_model\\test_08'
MODEL_PATH_CNN = 'deepGame4Music\\models\\CNN_VAE_model\\test_12'

# Datasets paths
DATASET_LSTM = 'data\\MIDI\\dataset\\all_progressions.npy'
DATASET_CNN = "data\\IMG\\dataset\\procceced_frames.npy"
TRAIN_DATA_LSTM = 'deepGame4Music\\models\\LSTM_VAE_model\\test_08\\train_data_lstm_01.npy'
TEST_DATA_LSTM = 'data\\IMG\\test_data\\test_04.npy'

# methods =============================================================================================
def align_latent_params(mu_cnn, log_var_cnn, mu_lstm, log_var_lstm):
    """ 
    Align the latent parameters of the CNN and LSTM models
    
    input:
        - mu_cnn: mean of the latent space of the CNN model
        - log_var_cnn: log variance of the latent space of the CNN model
        - mu_lstm: mean of the latent space of the LSTM model
        - log_var_lstm: log variance of the latent space of the LSTM model
    output:
        - mu_normalized: normalized mean of the latent space of the CNN model
        - log_var_scaled: scaled log variance of the latent space of the CNN model
    """
    mu_lstm_mean = np.mean(mu_lstm, axis=0)
    mu_lstm_std = np.std(mu_lstm, axis=0)
            
    var_lstm_mean = np.mean(np.exp(log_var_lstm))
    var_cnn_mean = np.mean(np.exp(log_var_cnn))

    # normalize mu
    mu_normalized = (mu_cnn - mu_lstm_mean) / mu_lstm_std
    
    # scale log_var
    scale_factor = np.log(var_lstm_mean / var_cnn_mean)
    log_var_scaled = log_var_cnn + scale_factor
    return mu_normalized, log_var_scaled

def sample_z(mu_normalized, log_var_scaled):
    epsilon = np.random.normal(size=mu_normalized.shape)
    z = mu_normalized + np.exp(0.5 * log_var_scaled) * epsilon
    return z

def remove_train_data_from_test(test_data, train_data):
    # Convert to tuples to facilitate comparison
    train_data_tuples = [tuple(map(tuple, progression)) for progression in train_data]
    test_data_tuples = [tuple(map(tuple, progression)) for progression in test_data]

    # Identify the progressions in test_data that are not in train_data
    filtered_test_data_tuples = [progression for progression in test_data_tuples if progression not in train_data_tuples]

    # Convert back to numpy array
    filtered_test_data = np.array([np.array(progression) for progression in filtered_test_data_tuples])

    return filtered_test_data

def save_images_grid(images, rows, cols, filename):
    """
    Save a grid of images to a file
    
    input:
        - images: images to save in a grid
        - rows: number of rows in the grid
        - cols: number of columns in the grid
        - filename: name of the file to save the grid
    output:
        - Nones
    """
    images = images.squeeze() # --> # Ensure that the images have the correct shape
    grid_image = Image.new('L', (cols * images.shape[2], rows * images.shape[1])) # --> Create a blank image to paste the images on
    
    for i in range(rows):
        for j in range(cols):
            index = i * cols + j
            if index < images.shape[0]:
                img = Image.fromarray((images[index] * 255).astype(np.uint8))
                grid_image.paste(img, (j * images.shape[2], i * images.shape[1]))
    
    grid_image.save(f"data\\IMG\\test_data\\test_04_png\\{filename}")

def save_images_separately(images, prefix):
    """
    Save a set of images to separate files
    input:
        - images: images to save
        - prefix: prefix for the filenames
    output:
        - Nones
    """
    images = images.squeeze()
    for i in range(images.shape[0]):
        img = Image.fromarray((images[i] * 255).astype(np.uint8))
        img.save(f"data\\IMG\\test_data\\test_04_png\\{prefix}_{i + 1}.png")

# =================================================================================================
# Custom objects for loading models with custom layers and metrics 
custom_objects = {
    'VAELossLayer': VAELossLayer,
    'Sampling': Sampling,
    '_reconstruction_loss_lstm': _reconstruction_loss_lstm,
    '_kl_loss': _kl_loss,
    '_reconstruction_loss_cnn': _reconstruction_loss_cnn
}
# =================================================================================================
# Evaluate class 
class Evaluate():
    def __init__(self,model_path, _type, test_data=None):
        # Models
        self.model = None        
        self.encoder = None
        self.decoder = None
        
        self.model_path = model_path 
        self.test_data = test_data
        
        # Model variables
        self.encoder_output = None
        self.prediction_model = None
        self.reconstructed_data = None
        self.new_data = None
        self.mu = None
        self.log_var = None

        # History                                
        self.history = None   
        self.type = _type     
        self.call()
        
    def call(self):
        self.load_model()        
    # =================================================================================================    
    def load_model(self):
        """
        Load complete model from a folder
        """
        #load parameters ---------------------------------------------------------
        paramemetes_path = os.path.join(self.model_path, 'parameters_vae.pkl')
        with open(paramemetes_path, 'rb') as f:
            parameters = pickle.load(f)

        print('==================== PARAMETERS ====================')
        print('Input shape:', parameters[0])
        print('latent_space_dim:', parameters[1])
        print('reconstruction_loss_weights:', parameters[2])
        print('kl_loss_weights:', parameters[3])
        print('learning_rate:', parameters[4])
        print('===================================================')
        
        # Load model -------------------------------------------------------------
        encoder_path = os.path.join(self.model_path, 'encoder_model')
        decoder_path = os.path.join(self.model_path, 'decoder_model')
        model_path = os.path.join(self.model_path, 'vae_model')
 
        self.encoder = load_model(encoder_path, custom_objects=custom_objects)
        self.decoder = load_model(decoder_path, custom_objects=custom_objects)
        self.model = load_model(model_path, custom_objects=custom_objects)
        
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

        # Load history -----------------------------------------------------------
        history_path = os.path.join(self.model_path, 'training_history.json')
        with open(history_path, 'r') as f:
            self.history = json.load(f)  
    
    # =================================================================================================
    # LSTM
    def reconstruct_encoder_lstm(self, progression_input=None):
        '''
        Codifica la progresión armonica de entrada y muestra el espacio latente
        input:
            - progresion armonica (num_samples, num_frames, num_notes) 
        '''
        
        if progression_input is not None:
            self.test_data = progression_input
            
        # Codificar progresion armonica --------------------------------------------
        self.encoder_output, self.mu, self.log_var = self.encoder.predict(self.test_data)        
        # Imprimir resultados ------------------------------------------------------
        print("\n ENCODER")
        print(f"\n Progresion Original:\n{self.test_data}")
        print(f"\n Encoder output:\n{self.encoder_output}")
        print(f"\n Forma del espacio latente:\n{self.encoder_output.shape}")
        
    def reconstruc_decoder_lstm(self, progression_input=None, encoder_output=None, save=False):
        '''
        decode a sample in the latent space, convert it to midi and show the generated progression generada
        input:
            - Chords progressions (num_samples, num_frames, num_notes) 
            - Sample of the latent space (num_samples, latent_space_dim)
            - Save MIDI file (True or False)
        output:
            - None
        '''
        # Update variables --------------------------------------------------------
        if encoder_output is not None:
            self.encoder_output = encoder_output
        
        if progression_input is not None:
            self.test_data = progression_input
        
        # Decode progression ------------------------------------------------------
        self.prediction_model = self.decoder.predict(self.encoder_output)        
        
        # Process generated progression -------------------------------------------
        if self.prediction_model.shape[2] == 13:
            notes = self.prediction_model[:, :, :-1]
            duration = self.prediction_model[:, :, -1]
            duration = np.expand_dims(duration, axis=2)

            self.reconstructed_data = np.where(notes > 0.5, 1, 0) # --> sigmoid
            self.reconstructed_data = np.concatenate((self.reconstructed_data, duration), axis=2)
                            
        elif self.prediction_model.shape[2] == 12:   
            self.reconstructed_data = np.where(self.prediction_model > 0.5, 1, 0)  

        # Save generated progression to MIDI --------------------------------------
        harmonic_progressions_to_midi(self.reconstructed_data, self.test_data, 'data\\MIDI\\midi_outputs', _save=save) 
                        
        # Imprimir resultados ------------------------------------------------------
        print("\n DECODER") 
        print(f"\n Progresion Generada procesada:\n{self.reconstructed_data}")
        print(f"\n Forma de la progresion generada:\n{self.reconstructed_data.shape}")
    
    def plot_latent_space_histogram(self, laten_space=None):
        """
        Plot the histogram of the latent space components
        input:
            - laten_space: latent space representation
        output:
            - None
        """
        if laten_spaces is not None:
            self.encoder_output = laten_spaces        
            
        for latent_space in self.encoder_output:
            # Ensure that the latent space has the correct shape
            if len(latent_space.shape) == 2 and latent_space.shape[0] == 1:
                latent_space = latent_space[0]
                    
            plt.figure(figsize=(12, 6))
            plt.bar(range(len(latent_space)), latent_space, color='skyblue', width=0.8, edgecolor='black', linewidth=1.5, alpha=0.7)
            plt.title('Latent Space Components')
            plt.xlabel('Component')
            plt.ylabel('Value')
            plt.xticks(range(len(latent_space)))
            
            # Config the layout
            plt.subplots_adjust(
                left=0.075,  # Espacio desde el borde izquierdo
                right=0.95,  # Espacio desde el borde derecho
                top=0.925,  # Espacio desde el borde superior
                bottom=0.1,  # Espacio desde el borde inferior
                hspace=0.4,  # Espacio horizontal entre subplots
                wspace=0.4  # Espacio vertical entre subplots
            )
                
            plt.show()
    # ================================================================================================= 
    # EVALUATE CNN
    def reconstruct_cnn(self, frames= None, num_samples=5, random=False):
        """
        Reconstruct images from the latent space
        input:
            - frames: images to reconstruct
            - num_samples: number of samples to reconstruct
            - random: random samples
        output:
            - None
        """
        if frames is not None:                             
            self.test_data = frames
            
        if num_samples > self.test_data.shape[0]:
            num_samples = self.test_data.shape[0]
        
        if random:
            random_index = np.random.choice(self.test_data.shape[0], num_samples, replace=False)
            self.test_data = self.test_data[random_index]

        self.encoder_output, self.mu, self.log_var = self.encoder.predict(self.test_data)
        self.reconstructed_data = self.decoder.predict(self.encoder_output)
                            
        # Visualize original and reconstructed images
        plt.figure(figsize=(10, 4))
        for i in range(num_samples):
            # Original
            plt.subplot(2, num_samples, i + 1)
            plt.imshow(self.test_data[i].squeeze(), cmap='gray')
            plt.title('Original')
            plt.axis('off')
            
            # Reconstructed
            plt.subplot(2, num_samples, num_samples + i + 1)
            plt.imshow(self.reconstructed_data[i].squeeze(), cmap='gray')
            plt.title('Reconstruida')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()

    def reconstruct_encoder_cnn(self, frames=None, num_samples = 5, random=False):
        """
        Encode images and show the latent space
        input:
            - frames: images to encode
            - num_samples: number of samples to encode
            - random: random samples
        output:
            - None
        """
        if frames is not None:
            self.test_data = frames
            
        if self.test_data.shape[0] < num_samples:
            num_samples = self.test_data.shape[0]
                                    
        if random:
            random_index = np.random.choice(self.test_data.shape[0], num_samples, replace=False)
            self.test_data = self.test_data[random_index]
        
        self.encoder_output, self.mu, self.log_var = self.encoder.predict(self.test_data)
        
        plt.figure(figsize=(10, 4))
        for i in range(num_samples):
            plt.subplot(1, num_samples, i + 1)
            plt.imshow(self.test_data[i].squeeze(), cmap='gray')
            plt.title('Original')
            plt.axis('off')           
        
        plt.tight_layout()
        plt.show()
        
    def reconstruc_decoder_cnn(self, encoder_output=None, num_samples= 5):
        """
        Reconstruct images from the latent space
        input:
            - encoder_output: latent space representation
            - num_samples: number of samples to reconstruct
        output:
            - None
        """
        if encoder_output is not None:
            self.encoder_output = encoder_output
        
        if self.test_data.shape[0] < num_samples:
            num_samples = self.test_data.shape[0]
        
        self.reconstructed_data = self.decoder.predict(self.encoder_output)
        
        plt.figure(figsize=(10, 4))
        for i in range(num_samples):
            plt.subplot(1, num_samples, i + 1)
            plt.imshow(self.reconstructed_data[i].squeeze(), cmap='gray')
            plt.title('Reconstruida')
            plt.axis('off')        
        plt.tight_layout()
        plt.show()
        
    # =================================================================================================
    def plot_history(self):
        fig, axs = plt.subplots(3)
        history_dict = self.history
        
        axs[0].plot(history_dict['loss'], label='train loss')
        axs[0].plot(history_dict['val_loss'], label='val loss')
        axs[0].set_ylabel('Loss')
        axs[0].set_xlabel('Epoch')
        axs[0].legend(loc='upper right')
        axs[0].set_title('Loss eval')

        # create KL loss subplot
        axs[1].plot(history_dict['_kl_loss'], label='KL loss')
        axs[1].plot(history_dict['val__kl_loss'], label='val KL loss')
        axs[1].set_ylabel('KL Loss')
        axs[1].set_xlabel('Epoch')
        axs[1].legend(loc='upper right')
        axs[1].set_title('KL Loss eval')
        
        # create reconstruction loss subplot
                
        # En el subplot de reconstrucción:
        if self.type == 'cnn':
            axs[2].plot(history_dict['_reconstruction_loss_cnn'], label='Reconstruction loss')
            axs[2].plot(history_dict['val__reconstruction_loss_cnn'], label='val Reconstruction loss')
        elif self.type == 'lstm':
            axs[2].plot(history_dict['_reconstruction_loss_lstm'], label='Reconstruction loss')
            axs[2].plot(history_dict['val__reconstruction_loss_lstm'], label='val Reconstruction loss')
            
        axs[2].set_ylabel('Reconstruction Loss')
        axs[2].set_xlabel('Epoch')
        axs[2].legend(loc='upper right')
        axs[2].set_title('Reconstruction Loss eval')

        plt.show()

if __name__ == "__main__":    
    
    # load dataset lstm ===================================================================================     
    test_data_lstm = np.load(DATASET_LSTM)    
    # data used for training
    train_data_lstm = np.load(TRAIN_DATA_LSTM)    
    
    # normalizar duracion min max scaler
    max_duration = 16
    min_duration = 0          
    test_data_lstm[:, :, -1] = (test_data_lstm[:, :, -1] - min_duration) / (max_duration - min_duration)    
    
    # Cut chords amount and/or remove duration
    test_data_lstm = test_data_lstm[:, :4, :-1]
        
    # data unused for training
    test_data_lstm = remove_train_data_from_test(test_data_lstm, train_data_lstm)
    
    # Seleccionar secuencias aleatorias
    random_index = np.random.choice(test_data_lstm.shape[0], 1, replace=False)
    test_data_lstm_samples = test_data_lstm[random_index]
    
# load dataset cnn ====================================================================================
    
    test_data_cnn = np.load(TEST_DATA_LSTM)

    # resize images
    test_data_cnn = tf.image.resize(test_data_cnn, (160, 120)).numpy()
    
    # Seleccionar secuencias aleatorias
    random_index = np.random.choice(test_data_cnn.shape[0], 1, replace=False)        
    test_data_cnn_samples = test_data_cnn[random_index]

# Evaluate Models =====================================================================================
    # Evaluate LSTM --------------------------------------------------------------------------------------------
    # evaluate_lstm = Evaluate(MODEL_PATH_LSTM, 'lstm', test_data_lstm_samples)
    # evaluate_lstm.reconstruct_encoder_lstm()
    #random_sample_latent_space = np.random.normal(size=(1, 16))
    #evaluate_lstm.reconstruc_decoder_lstm(encoder_output=random_sample_latent_space,save=False)  
    # evaluate_lstm.reconstruc_decoder_lstm()  
    #evaluate_lstm.plot_latent_space_histogram() 
      
    # calculate_percent_succes_notes(evaluate_lstm.test_data[:, :, :-1], evaluate_lstm.reconstructed_data_lstm[:, :, :-1])   
    # calculate_duration_error(evaluate_lstm.test_data[:, :, :], evaluate_lstm.reconstructed_data_lstm[:, :, :])
    
    # evaluate_lstm.plot_history()    
    
    # Evaluate CNN --------------------------------------------------------------------------------------------
    # evaluate_cnn = Evaluate(MODEL_PATH_CNN, 'cnn', test_data_cnn_samples)
    # evaluate_cnn.reconstruct_cnn()
    # evaluate_cnn.reconstruct_encoder_cnn()
    # evaluate_cnn.reconstruc_decoder_cnn()
    # evaluate_cnn.plot_history()
    
# Combine both models =================================================================================
    evaluate_lstm = Evaluate(MODEL_PATH_LSTM, 'lstm', test_data_lstm_samples)
    evaluate_cnn = Evaluate(MODEL_PATH_CNN, 'cnn', test_data_cnn_samples)
    
    # # cnn encoder
    evaluate_cnn.reconstruct_encoder_cnn()
    
    # # lstm decoder
    evaluate_lstm.reconstruc_decoder_lstm(evaluate_cnn.encoder_output_cnn)
    
# Combine both models with normalized latent space ====================================================  
    # evaluate_lstm = Evaluate(model_path=MODEL_PATH_LSTM, _type='lstm')
    # evaluate_cnn = Evaluate(model_path=MODEL_PATH_CNN, _type='cnn')                    
    
    # load mu and log_var from lstm 
    # mu_lstm = np.load(f'{MODEL_PATH_LSTM}\\mu_lstm.npy')
    # log_var_lstm = np.load(f'{MODEL_PATH_LSTM}\\log_var_lstm.npy')
    
    # to visualize original image
    # evaluate_cnn.reconstruct_encoder_cnn(frames=test_data_cnn)
    
    # get mu and log_var from cnn
    # encoder_output, mu, log_var = evaluate_cnn.encoder_output, evaluate_cnn.mu, evaluate_cnn.log_var
    
    # to visualize reconstructed image
    # evaluate_cnn.reconstruc_decoder_cnn()
        
    # align latent space
    # mu_normalized, log_var_scaled = align_latent_params(mu, log_var, mu_lstm, log_var_lstm)
    
    # sample z
    # z_normalized = sample_z(mu_normalized, log_var_scaled)
        
    # lstm decoder
    # evaluate_lstm.reconstruc_decoder_lstm(encoder_output=z_normalized, save=True)