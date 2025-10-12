import os
import subprocess
import webbrowser
import time
import tensorflow as tf
from tensorflow import keras

class TensorBoardManager:
    """
    Clase para gestionar TensorBoard y visualizaciones del modelo.
    """
    
    def __init__(self, log_dir, config=None):
        """
        Inicializa el manager de TensorBoard.
        
        Args:
            log_dir (str): Directorio donde guardar los logs
            config (dict): Configuración del modelo (opcional)
        """
        self.log_dir = log_dir
        self.config = config
        os.makedirs(self.log_dir, exist_ok=True)
    
    def create_callback(self, histogram_freq=1, write_graph=True, 
                       write_images=False, profile_batch=0, embeddings_freq=0):
        """
        Crea el callback de TensorBoard optimizado.
        
        Args:
            histogram_freq (int): Frecuencia para histogramas
            write_graph (bool): Si escribir el grafo del modelo
            write_images (bool): Si escribir imágenes
            profile_batch (int): Batch para profiling (0 = desactivado)
            embeddings_freq (int): Frecuencia para embeddings
            
        Returns:
            TensorBoard callback configurado
        """
        return keras.callbacks.TensorBoard(
            log_dir=self.log_dir,
            histogram_freq=histogram_freq,
            write_graph=write_graph,
            write_images=write_images,
            update_freq='epoch',
            profile_batch=profile_batch,
            embeddings_freq=embeddings_freq,
            write_steps_per_second=True
        )
    
    def create_model_visualization(self, model_builder):
        """
        Crea visualizaciones adicionales del modelo.
        
        Args:
            model_builder: Instancia del modelo MusicVAE
        """
        try:
            # Crear datos dummy para trazar el modelo
            batch_size = 1
            dummy_input = tf.zeros((batch_size, 
                                  self.config['model_params']['sequence_length'], 
                                  self.config['model_params']['features_dim']),
                                  dtype=tf.float32)
            dummy_h = tf.zeros((batch_size, self.config['model_params']['decoder_lstm_units']),
                              dtype=tf.float32)
            dummy_c = tf.zeros((batch_size, self.config['model_params']['decoder_lstm_units']),
                              dtype=tf.float32)
            
            # Crear diagramas PNG
            self._create_architecture_diagrams(model_builder)
            
            # Crear grafo para TensorBoard
            self._create_tensorboard_graph(model_builder, dummy_input, dummy_h, dummy_c)
            
            # Guardar resumen del modelo
            self._save_model_summary(model_builder)
            
        except Exception as e:
            print(f"Error al crear visualizaciones del modelo: {e}")
            print("Continuando sin visualizaciones del modelo...")
    
    def _create_architecture_diagrams(self, model_builder):
        """
        Crea diagramas PNG de la arquitectura del modelo.
        """
        try:
            # Encoder
            keras.utils.plot_model(
                model_builder.encoder, 
                to_file=os.path.join(self.log_dir, 'encoder_architecture.png'),
                show_shapes=True, 
                show_layer_names=True,
                show_layer_activations=True,
                rankdir='TB',
                dpi=200,
                expand_nested=True
            )
            
            # Decoder
            keras.utils.plot_model(
                model_builder.decoder,
                to_file=os.path.join(self.log_dir, 'decoder_architecture.png'),
                show_shapes=True,
                show_layer_names=True,
                show_layer_activations=True,
                rankdir='TB',
                dpi=200,
                expand_nested=True
            )
            
            # VAE completo (vista simplificada)
            keras.utils.plot_model(
                model_builder.vae,
                to_file=os.path.join(self.log_dir, 'vae_architecture_simple.png'),
                show_shapes=True,
                show_layer_names=True,
                rankdir='TB',
                dpi=200,
                expand_nested=False
            )
            
            # VAE completo (vista expandida)
            keras.utils.plot_model(
                model_builder.vae,
                to_file=os.path.join(self.log_dir, 'vae_architecture_detailed.png'),
                show_shapes=True,
                show_layer_names=True,
                rankdir='TB',
                dpi=200,
                expand_nested=True
            )
            
            print(f"Diagramas de arquitectura guardados en: {self.log_dir}")
            
        except Exception as e:
            print(f"No se pudieron crear diagramas de arquitectura: {e}")
    
    def _create_tensorboard_graph(self, model_builder, dummy_input, dummy_h, dummy_c):
        """
        Crea el grafo del modelo para TensorBoard.
        """
        writer = tf.summary.create_file_writer(self.log_dir)
        
        try:
            with writer.as_default():
                @tf.function(input_signature=[
                    tf.TensorSpec(shape=[None, self.config['model_params']['sequence_length'], 
                                        self.config['model_params']['features_dim']], dtype=tf.float32),
                    tf.TensorSpec(shape=[None, self.config['model_params']['decoder_lstm_units']], dtype=tf.float32),
                    tf.TensorSpec(shape=[None, self.config['model_params']['decoder_lstm_units']], dtype=tf.float32)
                ])
                def clean_forward_pass(x, h, c):
                    """Forward pass limpio para visualización."""
                    return model_builder.vae([x, h, c], training=False)
                
                # Ejecutar y trazar
                _ = clean_forward_pass(dummy_input, dummy_h, dummy_c)
                
                # Escribir el grafo
                tf.summary.graph(clean_forward_pass.get_concrete_function().graph)
                
            print(f"Grafo guardado en TensorBoard: {self.log_dir}")
            
        except Exception as e:
            print(f"Error al crear grafo para TensorBoard: {e}")
            
        finally:
            writer.close()
    
    def _save_model_summary(self, model_builder):
        """
        Guarda un resumen textual detallado del modelo.
        """
        try:
            with open(os.path.join(self.log_dir, 'model_summary.txt'), 'w') as f:
                f.write("="*80 + "\n")
                f.write("RESUMEN DETALLADO DEL MODELO MUSICVAE\n")
                f.write("="*80 + "\n\n")
                
                if self.config:
                    f.write("CONFIGURACIÓN:\n")
                    f.write("-" * 40 + "\n")
                    for key, value in self.config['model_params'].items():
                        f.write(f"{key}: {value}\n")
                    f.write("\n")
                
                f.write("ENCODER:\n")
                f.write("-" * 40 + "\n")
                model_builder.encoder.summary(print_fn=lambda x: f.write(x + '\n'))
                f.write("\n")
                
                f.write("DECODER:\n")
                f.write("-" * 40 + "\n")
                model_builder.decoder.summary(print_fn=lambda x: f.write(x + '\n'))
                f.write("\n")
                
                f.write("VAE COMPLETO:\n")
                f.write("-" * 40 + "\n")
                model_builder.vae.summary(print_fn=lambda x: f.write(x + '\n'))
                f.write("\n")
                
                # Información adicional
                f.write("DETALLES ADICIONALES:\n")
                f.write("-" * 40 + "\n")
                total_params = model_builder.vae.count_params()
                trainable_params = sum([tf.keras.backend.count_params(w) for w in model_builder.vae.trainable_weights])
                f.write(f"Parámetros totales: {total_params:,}\n")
                f.write(f"Parámetros entrenables: {trainable_params:,}\n")
                f.write(f"Parámetros no entrenables: {total_params - trainable_params:,}\n")
            
            print(f"Resumen del modelo guardado en: {os.path.join(self.log_dir, 'model_summary.txt')}")
            
        except Exception as e:
            print(f"Error al guardar resumen del modelo: {e}")
    
    def launch_tensorboard(self, port=6006, auto_open=True):
        """
        Lanza TensorBoard en un subproceso.
        
        Args:
            port (int): Puerto para TensorBoard
            auto_open (bool): Si abrir automáticamente el navegador
        """
        if not os.path.exists(self.log_dir):
            print(f"Error: No se encontraron logs en {self.log_dir}")
            return None
        
        print(f"Lanzando TensorBoard para: {self.log_dir}")
        print("TensorBoard mostrará:")
        print("  - Scalars: Métricas de entrenamiento")
        print("  - Graphs: Arquitectura del modelo")
        print("  - Histograms: Distribución de pesos")
        print("\nPresiona Ctrl+C para detener TensorBoard")
        
        try:
            # Lanzar TensorBoard
            process = subprocess.Popen([
                "tensorboard", 
                "--logdir", self.log_dir, 
                "--port", str(port),
                "--host", "localhost",
                "--reload_interval", "1"
            ])
            
            # Esperar un momento
            time.sleep(3)
            
            print("\n" + "="*60)
            print("TENSORBOARD INICIADO")
            print("="*60)
            print(f"URL: http://localhost:{port}")
            print("\nOpciones para abrir:")
            print(f"1. Abre http://localhost:{port} en tu navegador")
            print("2. En VS Code: Ctrl+Shift+P > 'Simple Browser: Show'")
            print("3. En VS Code: Usar la extensión 'TensorBoard'")
            print("\nPresiona Ctrl+C para detener TensorBoard")
            print("="*60)
            
            if auto_open:
                try:
                    webbrowser.open(f"http://localhost:{port}")
                except:
                    pass  # Si no puede abrir el navegador, continuar
            
            return process
            
        except FileNotFoundError:
            print("Error: TensorBoard no está instalado.")
            print("Instálalo con: pip install tensorboard")
            return None
        except Exception as e:
            print(f"Error al lanzar TensorBoard: {e}")
            return None
    
    def stop_tensorboard(self, process):
        """
        Detiene un proceso de TensorBoard.
        
        Args:
            process: Proceso de TensorBoard a detener
        """
        if process:
            try:
                process.terminate()
                print("TensorBoard detenido.")
            except:
                pass

def create_tensorboard_manager(run_dir, config):
    """
    Función helper para crear un TensorBoardManager.
    
    Args:
        run_dir (str): Directorio del experimento
        config (dict): Configuración del modelo
        
    Returns:
        TensorBoardManager instance
    """
    logs_dir = os.path.join(run_dir, "logs")
    return TensorBoardManager(logs_dir, config)