import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import matplotlib.pyplot as plt

class AutoregressiveDecoderLayer(layers.Layer):
    """
    Capa personalizada que encapsula el bucle de decodificación autorregresiva.
    """
    def __init__(self, features_dim, conductor_lstm_units, decoder_lstm_units, sequence_length, name="autoregressive_decoder"):
        super().__init__(name=name)
        self.features_dim = features_dim
        self.conductor_lstm_units = conductor_lstm_units
        self.decoder_lstm_units = decoder_lstm_units
        self.sequence_length = sequence_length

        # Las capas internas se definen UNA VEZ aquí, en el constructor.
        self.decoder_lstm = layers.LSTM(self.decoder_lstm_units, return_sequences=False, return_state=True, name="decoder_lstm_cell")
        self.output_dense_layer = layers.Dense(self.features_dim, activation='sigmoid', name="output_projection")

    def call(self, inputs):
        """
        Aquí ocurre la lógica del bucle for.
        """
        conductor_lstm_output, decoder_teacher_inputs, initial_state_h, initial_state_c = inputs

        all_outputs = []
        previous_chord = tf.zeros_like(decoder_teacher_inputs[:, 0, :])
        current_states = [initial_state_h, initial_state_c]

        for t in range(self.sequence_length):
            plan_t = conductor_lstm_output[:, t, :]

            if t > 0:
                previous_chord = decoder_teacher_inputs[:, t-1, :]

            lstm_input = layers.concatenate([previous_chord, plan_t], axis=-1)
            lstm_input = tf.expand_dims(lstm_input, 1)

            lstm_output, h, c = self.decoder_lstm(lstm_input, initial_state=current_states)
            current_states = [h, c]

            output_t = self.output_dense_layer(lstm_output)
            all_outputs.append(output_t)

        output_sequence = tf.stack(all_outputs, axis=1)

        return output_sequence, current_states[0], current_states[1]

class MusicVAEModel(keras.Model):
    """
    Modelo VAE personalizado que sobrescribe train_step para manejar métricas personalizadas.
    """
    def __init__(self, encoder, decoder, kl_weight=0.5, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.kl_weight = kl_weight
        
        # Métricas personalizadas
        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
        
    def call(self, inputs, training=None):
        """Forward pass del modelo."""
        x, y = inputs
        vae_input, initial_state_h, initial_state_c = x
        z_mean, z_log_var, z = self.encoder(vae_input, training=training)
        reconstruction, _, _ = self.decoder([z, vae_input, initial_state_h, initial_state_c], training=training)
        return reconstruction 
        
    def train_step(self, data):
        """
        Paso de entrenamiento para actualizar los pesos del modelo.
        """
        # Desempaquetar los datos
        x, y = data
        vae_input, initial_state_h, initial_state_c = x
        
        with tf.GradientTape() as tape:
            # Forward pass
            z_mean, z_log_var, z = self.encoder(vae_input)
            reconstruction, _, _ = self.decoder([z, vae_input, initial_state_h, initial_state_c])
            
            # Calcular pérdidas
            reconstruction_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(y, reconstruction)
            )
            
            kl_loss = -0.5 * tf.reduce_mean(
                z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
            )
            
            total_loss = reconstruction_loss + self.kl_weight * kl_loss
        
        # Backpropagation
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        # Actualizar métricas
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    def test_step(self, data):
        """
        Paso de prueba para evaluar el modelo.
        """
        # Desempaquetar los datos
        x, y = data
        vae_input, initial_state_h, initial_state_c = x
        
        # Forward pass
        z_mean, z_log_var, z = self.encoder(vae_input)
        reconstruction, _, _ = self.decoder([z, vae_input, initial_state_h, initial_state_c])
        
        # Calcular pérdidas
        reconstruction_loss = tf.reduce_mean(
            keras.losses.binary_crossentropy(y, reconstruction)
        )
        
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        )
        
        total_loss = reconstruction_loss + self.kl_weight * kl_loss
        
        # Actualizar métricas
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


class MusicVAE():
    """
    Una implementación de un Variational Autoencoder Jerárquico y Autorregresivo
    inspirado en MusicVAE, diseñado para generar progresiones de acordes.
    """
    def __init__(self, features_dim, sequence_length, latent_dim,
                 encoder_lstm_units=[128, 64, 256],
                 conductor_lstm_units=128,
                 decoder_lstm_units=256,
                 kl_weight=0.5):
        """
        Inicializa y construye la arquitectura completa del VAE.

        Args:
            features_dim (int): Dimensión de la entrada (ej. 84 para el piano roll).
            sequence_length (int): Longitud de las secuencias (ej. 16).
            latent_dim (int): Dimensión del espacio latente z.
            encoder_lstm_units (list): Lista de unidades para las capas LSTM del codificador.
            conductor_lstm_units (int): Unidades para la capa LSTM del Conductor.
            decoder_lstm_units (int): Unidades para la capa LSTM del Decodificador principal.
            kl_weight (float): Peso para la pérdida de divergencia KL en la pérdida total.
        """
        self.features_dim = features_dim
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        self.encoder_lstm_units = encoder_lstm_units
        self.conductor_lstm_units = conductor_lstm_units
        self.decoder_lstm_units = decoder_lstm_units
        self.kl_weight = kl_weight

        # Construir los componentes del modelo
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

        # Construir y compilar el VAE completo
        self.vae = self._build_vae()
        self._compile_vae()

    def _build_encoder(self):
        """Construye el modelo del codificador."""
        # Capa de Entrada
        inputs = layers.Input(shape=(self.sequence_length, self.features_dim), name="encoder_input")

        # Pila de LSTMs Bidireccionales para un mejor contexto
        x = layers.Bidirectional(layers.LSTM(self.encoder_lstm_units[0], return_sequences=True), name="bi_lstm_1")(inputs)
        x = layers.Bidirectional(layers.LSTM(self.encoder_lstm_units[1], return_sequences=True), name="bi_lstm_2")(x)

        # El resumen final de la secuencia
        summary_vector = layers.Bidirectional(layers.LSTM(self.encoder_lstm_units[2]), name="bi_lstm_summary")(x)

        # Capas densas para el espacio latente
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(summary_vector)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(summary_vector)

        # "Truco de la Reparametrización" para el muestreo
        def sampling(args):
            z_mean, z_log_var = args
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = K.random_normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

        z = layers.Lambda(sampling, name="z_sampling")([z_mean, z_log_var])

        return keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")

    def _build_decoder(self):
        """Construye el modelo del decodificador jerárquico y autorregresivo."""
        # --- Entradas del Decodificador ---
        latent_input = layers.Input(shape=(self.latent_dim,), name="latent_input")
        # Secuencia completa para el "Teacher Forcing"
        decoder_inputs = layers.Input(shape=(self.sequence_length, self.features_dim), name="decoder_teacher_forcing_inputs")
        # Estados ocultos iniciales para la generación continua
        initial_state_h = layers.Input(shape=(self.decoder_lstm_units,), name="initial_h")
        initial_state_c = layers.Input(shape=(self.decoder_lstm_units,), name="initial_c")

        # --- 1. Conductor (Crea el "Plan") ---
        # Repetimos z para que el LSTM pueda generar una secuencia de planes
        repeated_z = layers.RepeatVector(self.sequence_length)(latent_input)
        conductor_lstm_output = layers.LSTM(self.conductor_lstm_units, return_sequences=True, name="conductor_lstm")(repeated_z)
        # conductor_lstm ahora es una secuencia de N "embeddings de plan"

        # --- 2. Bloque Autorregresivo ---
        # Creamos una instancia de nuestra capa personalizada
        autoregressive_block = AutoregressiveDecoderLayer(
            self.features_dim, self.conductor_lstm_units,
            self.decoder_lstm_units, self.sequence_length
        )

        # La llamamos como si fuera una única función
        decoder_output_sequence, final_h, final_c = autoregressive_block(
            [conductor_lstm_output, decoder_inputs, initial_state_h, initial_state_c]
        )

        return keras.Model(
            [latent_input, decoder_inputs, initial_state_h, initial_state_c],
            [decoder_output_sequence, final_h, final_c],
            name="decoder"
        )

    def _build_vae(self):
        """Conecta el codificador y el decodificador para formar el VAE completo."""
        # Usar el modelo personalizado que sobrescribe train_step
        vae_model = MusicVAEModel(
            encoder=self.encoder,
            decoder=self.decoder,
            kl_weight=self.kl_weight,
            name="music_vae"
        )
        
        # Construir el modelo con datos dummy para evitar errores al guardar
        dummy_input = tf.zeros((1, self.sequence_length, self.features_dim))
        dummy_h = tf.zeros((1, self.decoder_lstm_units))
        dummy_c = tf.zeros((1, self.decoder_lstm_units))
        _ = vae_model([dummy_input, dummy_h, dummy_c], training=False)
        
        return vae_model

    def _compile_vae(self):
        """Compila el VAE."""
        self.vae.compile(optimizer='adam')
        
if __name__ == '__main__':
    # --- Parámetros de Ejemplo ---
    FEATURES_DIM = 84         # Dimensión del Piano Roll (ej. C1 a B7)
    SEQUENCE_LENGTH = 16      # N acordes por secuencia
    LATENT_DIM = 64          # Dimensión del espacio latente z

    # --- Instanciación del Modelo ---
    music_model = MusicVAE(
        features_dim=FEATURES_DIM,
        sequence_length=SEQUENCE_LENGTH,
        latent_dim=LATENT_DIM
    )
    
    # --- Resumen de la Arquitectura ---
    print("--- Resumen del Codificador ---")
    music_model.encoder.summary()
    
    print("\n--- Resumen del Decodificador ---")
    music_model.decoder.summary()

    print("\n--- Resumen del VAE Completo ---")
    music_model.vae.summary()