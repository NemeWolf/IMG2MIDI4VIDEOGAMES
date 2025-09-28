import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, LSTM, Dense, Layer, RepeatVector, TimeDistributed, GaussianNoise, Dropout
from tensorflow.keras.utils import register_keras_serializable
from utils.metrics import _reconstruction_loss_lstm, _kl_loss
import numpy as np

# Vae Loss Layer ==========================================================
@register_keras_serializable(package='Custom', name='VAELossLayer')
class VAELossLayer(Layer):
    """
    Loss layer for VAE to calculate the reconstruction loss and KL divergence loss
    input  --> y_true, y_pred, mu, log_var
    output --> y_pred (para mantener la compatibilidad con Keras)
    """
    def __init__(self, reconstruction_loss_weight=1, kl_loss_weight=1, **kwargs):
        super(VAELossLayer, self).__init__(**kwargs)
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.kl_loss_weight = kl_loss_weight
        
    def call(self, inputs):
        y_true, y_pred, mu, log_var = inputs

        # calculate reconstruction loss and kl loss 
        self.reconstruction_loss = _reconstruction_loss_lstm(y_true, y_pred)  
        self.kl_loss = _kl_loss(mu, log_var)      
        
        # Calculate total loss
        loss = self.reconstruction_loss_weight * self.reconstruction_loss + self.kl_loss_weight * self.kl_loss
        self.add_loss(loss)
                        
        return y_pred

    def get_config(self):
        config = super().get_config()
        config.update({
            "reconstruction_loss_weight": self.reconstruction_loss_weight,
            "kl_loss_weight": self.kl_loss_weight,
        })
        return config
      

# SAMPLE POINT FROM NORMAL DISTRIBUTION =========================================
@register_keras_serializable(package='Custom', name='Sampling')
class Sampling(Layer):
    """
    Show points from the normal distribution and scale them with the standard deviation
    input  --> z_mean, z_log_var
    output --> z
    """
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# # CONCAPTENATE LAYERS ===========================================================
@register_keras_serializable(package='Custom', name='ConcatenateLayer')
class ConcatenateLayer(Layer):
    """Concatenate two layers"""
    def call(self, inputs):
        x, y = inputs
        return K.concatenate([x, y])

# VARIATIONAL AUTOENCODER ===================================================================
class LSTM_VAE:

  def __init__(self, input_shape, latent_space_dim, reconstruction_loss_weights=1, kl_loss_weights=1):

    self.input_shape = input_shape            
    self.latent_space_dim = latent_space_dim 
    self.reconstruction_loss_weights = reconstruction_loss_weights 
    self.kl_loss_weights = kl_loss_weights
    
    self.encoder = None
    self.decoder = None
    self.model = None 
    
    self.sampling = Sampling()
    
    self._shape_before_bottleneck = None
    self._model_input = None

    self._build() # --> construye el modelo completo del autoencoder
                                  
  # BUILD ENCODER, DECODER AND VAE ===================================================
  def _build(self):
    """Build the entire autoencoder model"""
    self._build_encoder()
    self._build_decoder()
    self._build_lstm_vae()
    
  # CONSTRUMOS VAE ========================================================
  def _build_lstm_vae(self):
      """Build the VAE model"""          
      model_input = self._model_input
      encoder_output, mu, log_var = self.encoder(model_input)
      decoder_output = self.decoder(encoder_output)

      vae_loss_layer = VAELossLayer(reconstruction_loss_weight=self.reconstruction_loss_weights, kl_loss_weight=self.kl_loss_weights,  name="vae_loss_layer")      
      model_output= vae_loss_layer([model_input, decoder_output, mu, log_var])

      self.model = Model(model_input, model_output, name="LSTM_VAE")   
    
  # CONSTRUIMOS ENCODER ============================================================    
  def _build_encoder(self):
    """
    Build the encoder of the autoencoder
    encoder inpput --> Chords progression
    encoder output --> mu
                   --> log_var
                   --> normal distribution sample
    """
    # input layer
    encoder_inputs = Input(shape=self.input_shape, name="encoder_input")
    
    # hidden layers
    x = LSTM(256, return_sequences=True, dropout=0.2, name="encoder_lstm_1")(encoder_inputs)        
    x = LSTM(128, return_sequences=False, dropout=0.2, name="encoder_lstm_2")(x)    
    self._shape_before_bottleneck = K.int_shape(x)[1:]        
    x = Dense(64, activation='relu',  name="encoder_dense")(x)
    
    # output layers
    mu = Dense(self.latent_space_dim, name="mu")(x)    
    log_var = Dense(self.latent_space_dim, name="log_var")(x) 
    
    # sampling layer
    encoder_outputs = Sampling()([mu, log_var])  
      
    self._model_input = encoder_inputs     
    self.encoder = Model(encoder_inputs, [encoder_outputs, mu, log_var], name="encoder") 
  
  # CONSTRUIMOS DECODER============================================================  
  def _build_decoder(self):
    """
    Build the decoder of the autoencoder
    decoder input  --> normal distribution sample
    decoder output --> chords progression
    """
    # input layer
    decoder_inputs = Input(shape=(self.latent_space_dim,), name="decoder_input") 
     
    # hidden layers
    x = Dense(64, activation='relu', name="decoder_dense")(decoder_inputs)
    x = Dense(np.prod(self._shape_before_bottleneck), activation='relu', name='decoder_dense_lstm_1')(x)
    x = RepeatVector(self.input_shape[0])(x)    
    x = LSTM(256, return_sequences=True, dropout=0.2, name="decoder_lstm_2")(x)
   
    # output layer
    decoder_output = TimeDistributed(Dense(self.input_shape[1], activation='sigmoid'), name='decoder_output')(x)

    self.decoder = Model(decoder_inputs, decoder_output, name="decoder")
# AUTOENCODER ===================================================================
if __name__ == "__main__":  
  # to ejecute this code, you need to use python -m models.lstm_vae_model  

  lstm_vae = LSTM_VAE(
      input_shape=(8, 12),
      latent_space_dim=32
  )  
  lstm_vae.encoder.summary()
  lstm_vae.decoder.summary()
  lstm_vae.model.summary()