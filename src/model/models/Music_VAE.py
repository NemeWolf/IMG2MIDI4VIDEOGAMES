import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K

class MusicVAE:
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
        
        return tf.keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")

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
        conductor_lstm = layers.LSTM(self.conductor_lstm_units, return_sequences=True, name="conductor_lstm")(repeated_z)
        # conductor_lstm ahora es una secuencia de N "embeddings de plan"

        # --- 2. Decodificador Autorregresivo (El "Intérprete") ---
        # Definimos las capas que se reutilizarán en el bucle
        decoder_lstm = layers.LSTM(self.decoder_lstm_units, return_sequences=False, return_state=True, name="decoder_lstm")
        output_dense_layer = layers.Dense(self.features_dim, activation='sigmoid', name="output_dense")

        # Inicialización del bucle
        all_outputs = []
        # El "acorde previo" para el primer paso es un vector de ceros
        previous_chord = tf.zeros_like(decoder_inputs[:, 0, :])
        current_states = [initial_state_h, initial_state_c]

        for t in range(self.sequence_length):
            # Preparamos la entrada para este paso de tiempo
            plan_t = conductor_lstm[:, t, :]
            # En entrenamiento, el "acorde previo" es el real del paso anterior
            if t > 0:
                previous_chord = decoder_inputs[:, t-1, :]
            
            lstm_input = layers.concatenate([previous_chord, plan_t], axis=-1)
            # El LSTM necesita una dimensión de tiempo, así que la añadimos
            lstm_input = tf.expand_dims(lstm_input, 1)

            # Ejecutamos un paso del LSTM
            lstm_output, h, c = decoder_lstm(lstm_input, initial_state=current_states)
            current_states = [h, c] # Actualizamos el estado para el siguiente paso

            # Proyectamos la salida para predecir el siguiente acorde
            output_t = output_dense_layer(lstm_output)
            all_outputs.append(output_t)

        # Juntamos las salidas de cada paso en una única secuencia
        decoder_output_sequence = layers.Lambda(lambda x: tf.stack(x, axis=1), name="stack_outputs")(all_outputs)
        
        return tf.keras.Model(
            [latent_input, decoder_inputs, initial_state_h, initial_state_c],
            [decoder_output_sequence, current_states[0], current_states[1]],
            name="decoder"
        )

    def _build_vae(self):
        """Conecta el codificador y el decodificador para formar el VAE completo."""
        # --- Entradas del VAE ---
        vae_input = self.encoder.input
        # Estados iniciales (normalmente ceros durante el entrenamiento)
        initial_state_h = layers.Input(shape=(self.decoder_lstm_units,), name="vae_initial_h")
        initial_state_c = layers.Input(shape=(self.decoder_lstm_units,), name="vae_initial_c")

        # --- Conexión ---
        z_mean, z_log_var, z = self.encoder(vae_input)
        
        # Durante el entrenamiento, el VAE usa su propia entrada para el "teacher forcing"
        reconstruction, _, _ = self.decoder([z, vae_input, initial_state_h, initial_state_c])

        # Añadimos la pérdida KL como una capa de pérdida del modelo
        kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        self.vae_kl_loss = kl_loss * self.kl_weight
        
        return tf.keras.Model(
            [vae_input, initial_state_h, initial_state_c],
            reconstruction,
            name="music_vae"
        )

    def _compile_vae(self):
        """Compila el VAE con la pérdida combinada."""
        self.vae.add_loss(self.vae_kl_loss)
        self.vae.compile(
            optimizer='adam',
            loss='binary_crossentropy' # Pérdida de reconstrucción
        )

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

    # --- Cómo se Entrenaría el Modelo ---
    # Cargar el dataset 'train_set.pkl' y 'dev_set.pkl'
    # train_data = pd.read_pickle('final_datasets/train_set.pkl')['piano_roll'].values
    # train_data = np.stack(train_data, axis=0)
    
    # # Los estados iniciales son ceros
    # initial_h = np.zeros((train_data.shape[0], music_model.decoder_lstm_units))
    # initial_c = np.zeros((train_data.shape[0], music_model.decoder_lstm_units))
    
    # print(f"\nForma de los datos de entrenamiento: {train_data.shape}")
    
    # music_model.vae.fit(
    #     [train_data, initial_h, initial_c],
    #     train_data, # El modelo aprende a reconstruir su propia entrada
    #     epochs=50,
    #     batch_size=64
    # )