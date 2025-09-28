import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Flatten, Dense, Reshape, Conv2DTranspose, Activation, Lambda, Layer, Dropout
from tensorflow.keras.utils import register_keras_serializable
from utils.metrics import _reconstruction_loss_cnn, _kl_loss
import numpy as np

# VAE LOSS LAYER ==========================================================

class VAELossLayer(Layer):
    def __init__(self, reconstruction_loss_weight=1000, kl_loss_weight=1, **kwargs):
        super(VAELossLayer, self).__init__(**kwargs)
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.kl_loss_weight_var = tf.Variable(kl_loss_weight, trainable=False, dtype=tf.float32)
        
    def call(self, inputs):
        y_true, y_pred, mu, log_var = inputs

        self.reconstruction_loss = _reconstruction_loss_cnn(y_true, y_pred)
        self.kl_loss = _kl_loss(mu, log_var)

        # Comine both losses
        loss = self.reconstruction_loss_weight * self.reconstruction_loss + self.kl_loss_weight_var * self.kl_los
        self.add_loss(loss)
        return y_pred
        

# SAMPLE POINT FROM NORMAL DISTRIBUTION =========================================
class Sampling(Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        z_log_var = tf.clip_by_value(z_log_var, -5.0, 5.0) # --> Clip_by_value to avoid NaN values
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# AUTOENCODER ===================================================================

class CNN_VAE:

  def __init__(self, input_shape, latent_space_dim, reconstruction_loss_weights=15, kl_loss_weights=1):

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

    self._build() # --> Construye el modelo completo del autoencoder
                                  
  # BUILD ENCODER, DECODER AND VAE ===================================================
  def _build(self):
    """Builds the encoder, decoder and the VAE model"""
    self._build_encoder()
    self._build_decoder()
    self._build_cnn_vae()
    
  # BUILD VAE ========================================================
  def _build_cnn_vae(self):
          
      model_input = self._model_input
      encoder_output, mu, log_var = self.encoder(model_input)
      decoder_output = self.decoder(encoder_output)

      vae_loss_layer = VAELossLayer(reconstruction_loss_weight=self.reconstruction_loss_weights, 
                                    kl_loss_weight=self.kl_loss_weights,  
                                    name="vae_loss_layer")([model_input, decoder_output, mu, log_var ])
            
      self.model = Model(
        inputs=model_input,
        outputs=vae_loss_layer,
        name="CNN_VAE"
    )
      
  # BUILD ENCODER ============================================================    
  def _build_encoder(self):
    """Build encoder of the autoencoder"""
    # Input layer    
    encoder_inputs = Input(shape=self.input_shape, name="encoder_input")
    # Convolutional layers
    x = Conv2D(filters=64, kernel_size=3, strides=1, padding="same", activation="relu", name="encoder_conv_layer_1")(encoder_inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.05)(x)
    x = Conv2D(filters=64, kernel_size=3, strides=2, padding="same", activation="relu", name="encoder_conv_layer_2")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.05)(x)
    x = Conv2D(filters=32, kernel_size=3, strides=2, padding="same", activation="relu", name="encoder_conv_layer_3")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.05)(x)
    x = Conv2D(filters=32, kernel_size=3, strides=2, padding="same", activation="relu", name="encoder_conv_layer_4")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.05)(x)
    x = Conv2D(filters=16, kernel_size=3, strides=1, padding="same", activation="relu", name="encoder_conv_layer_5")(x)
    x = Dropout(0.05)(x)
    # Fully connected
    self._shape_before_bottleneck = K.int_shape(x)[1:] # --> Shape before bottleneck to use in the decoder
    x = Flatten(name="flatten")(x) 
    x = Dense(64, activation='relu', name="encoder_dense")(x)
    # Output layers
    mu = Dense(self.latent_space_dim, name="mu")(x)
    log_var = Dense(self.latent_space_dim, name="log_var")(x)
    encoder_outputs = Sampling()([mu, log_var]) #--> Sampling layer    
    self._model_input = encoder_inputs  
    self.encoder = Model(encoder_inputs, [encoder_outputs, mu, log_var], name="encoder") 
  
  # BUILD DECODER =======================================================================
  def _build_decoder(self):
    """Build the decoder of the autoencoder"""
    # Input layer
    decoder_inputs = Input(shape=(self.latent_space_dim,), name="decoder_input")
    # Fully connected
    x = Dense(64, activation='relu', name="decoder_dense")(decoder_inputs)
    x = Dense(np.prod(self._shape_before_bottleneck), name="flatten")(x) 
    x = Reshape(self._shape_before_bottleneck, name="decoder_conv_5_before")(x)
    # Transposed convolutional layers
    x = Conv2DTranspose(filters=16, kernel_size=3, strides=1, padding="same", name="decoder_conv_5")(x) 
    x = ReLU()(x)  
    x = BatchNormalization()(x)    
    x = Dropout(0.05)(x)    
    x = Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding="same", name="decoder_conv4")(x)
    x = ReLU()(x)
    x = BatchNormalization()(x)    
    x = Dropout(0.05)(x)    
    x = Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding="same", name="decoder_conv3")(x)
    x = ReLU()(x)
    x = BatchNormalization()(x)    
    x = Dropout(0.05)(x)    
    x = Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding="same", name="decoder_conv_2")(x)
    x = ReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.05)(x)    
    x = Conv2DTranspose(filters=64, kernel_size=3, strides=1, padding="same", name="decoder_conv_1")(x)
    x = ReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.05)(x)
    # Output layer
    decoder_outputs = Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding="same", 
                                      activation="sigmoid", dtype='float32', name="decoder_output")(x)
    
    self.decoder = Model(decoder_inputs, decoder_outputs, name="decoder")

# AUTOENCODER ===================================================================
if __name__ == "__main__":
  
    # to ejecute this code, you need to use python -m models.cnn_vae_model  

  vae = CNN_VAE(
      input_shape=(160, 120, 1),
      latent_space_dim=32
  )
  
  vae.encoder.summary()
  vae.decoder.summary()
  vae.model.summary()