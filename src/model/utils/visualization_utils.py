import os
import pandas as pd
import matplotlib.pyplot as plt

class TrainingVisualizer:
    """
    Clase para visualizar el historial de entrenamiento.
    """
    
    def __init__(self, run_dir):
        """
        Inicializa el visualizador.
        
        Args:
            run_dir (str): Directorio del experimento
        """
        self.run_dir = run_dir
    
    def plot_training_history(self, save_plots=True, show_plots=True):
        """
        Visualiza el historial de entrenamiento del modelo.
        
        Args:
            save_plots (bool): Si True, guarda las gráficas
            show_plots (bool): Si True, muestra las gráficas
        """
        history_path = os.path.join(self.run_dir, 'training_history.csv')
        
        if not os.path.exists(history_path):
            print(f"Error: No se encontró el archivo de historial en {history_path}")
            return
        
        history_df = pd.read_csv(history_path)
        print(f"Cargando historial de entrenamiento desde: {history_path}")
        
        # Crear figura con subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History', fontsize=16)
        
        # Gráfico 1: Pérdida total
        self._plot_total_loss(axes[0, 0], history_df)
        
        # Gráfico 2: KL Loss y Reconstruction Loss
        self._plot_component_losses(axes[0, 1], history_df)
        
        # Gráfico 3: Métricas de validación
        self._plot_validation_metrics(axes[1, 0], history_df)
        
        # Gráfico 4: Resumen general
        self._plot_all_metrics_overview(axes[1, 1], history_df)
        
        plt.tight_layout()
        
        # Guardar y mostrar
        if save_plots:
            plot_path = os.path.join(self.run_dir, 'training_plots.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Gráficas guardadas en: {plot_path}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        # Imprimir estadísticas
        self._print_training_summary(history_df)
    
    def _plot_total_loss(self, ax, history_df):
        """Grafica la pérdida total."""
        ax.plot(history_df['loss'], label='Training Loss', color='blue')
        if 'val_loss' in history_df.columns:
            ax.plot(history_df['val_loss'], label='Validation Loss', color='red')
        ax.set_title('Total Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)
    
    def _plot_component_losses(self, ax, history_df):
        """Grafica KL Loss y Reconstruction Loss."""
        if 'kl_loss' in history_df.columns and 'reconstruction_loss' in history_df.columns:
            ax.plot(history_df['kl_loss'], label='KL Loss', color='green')
            ax.plot(history_df['reconstruction_loss'], label='Reconstruction Loss', color='orange')
            ax.set_title('KL Loss vs Reconstruction Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True)
        else:
            ax.text(0.5, 0.5, 'KL/Reconstruction Loss\ndata not available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('KL Loss vs Reconstruction Loss')
    
    def _plot_validation_metrics(self, ax, history_df):
        """Grafica métricas de validación."""
        val_metrics = [col for col in history_df.columns if col.startswith('val_') and col != 'val_loss']
        if val_metrics:
            for metric in val_metrics:
                ax.plot(history_df[metric], label=metric)
            ax.set_title('Validation Metrics')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True)
        else:
            ax.text(0.5, 0.5, 'No validation metrics\navailable', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Validation Metrics')
    
    def _plot_all_metrics_overview(self, ax, history_df):
        """Grafica resumen de todas las métricas."""
        metrics_to_plot = ['loss']
        if 'val_loss' in history_df.columns:
            metrics_to_plot.append('val_loss')
        if 'kl_loss' in history_df.columns:
            metrics_to_plot.append('kl_loss')
        if 'reconstruction_loss' in history_df.columns:
            metrics_to_plot.append('reconstruction_loss')
        
        for metric in metrics_to_plot:
            ax.plot(history_df[metric], label=metric, alpha=0.7)
        
        ax.set_title('All Metrics Overview')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)
        ax.set_yscale('log')
    
    def _print_training_summary(self, history_df):
        """Imprime un resumen estadístico del entrenamiento."""
        print("\n" + "="*60)
        print("RESUMEN DEL ENTRENAMIENTO")
        print("="*60)
        
        total_epochs = len(history_df)
        print(f"Total de épocas completadas: {total_epochs}")
        
        # Mejor época
        if 'val_loss' in history_df.columns:
            best_epoch = history_df['val_loss'].idxmin() + 1
            best_val_loss = history_df['val_loss'].min()
            print(f"Mejor época (val_loss): {best_epoch}")
            print(f"Mejor validation loss: {best_val_loss:.6f}")
        else:
            best_epoch = history_df['loss'].idxmin() + 1
            best_loss = history_df['loss'].min()
            print(f"Mejor época (loss): {best_epoch}")
            print(f"Mejor training loss: {best_loss:.6f}")
        
        # Pérdidas finales
        final_loss = history_df['loss'].iloc[-1]
        print(f"Pérdida final de entrenamiento: {final_loss:.6f}")
        
        if 'val_loss' in history_df.columns:
            final_val_loss = history_df['val_loss'].iloc[-1]
            print(f"Pérdida final de validación: {final_val_loss:.6f}")
        
        if 'kl_loss' in history_df.columns:
            final_kl_loss = history_df['kl_loss'].iloc[-1]
            print(f"KL Loss final: {final_kl_loss:.6f}")
        
        if 'reconstruction_loss' in history_df.columns:
            final_recon_loss = history_df['reconstruction_loss'].iloc[-1]
            print(f"Reconstruction Loss final: {final_recon_loss:.6f}")
        
        print("="*60)

def create_training_visualizer(run_dir):
    """
    Función helper para crear un TrainingVisualizer.
    
    Args:
        run_dir (str): Directorio del experimento
        
    Returns:
        TrainingVisualizer instance
    """
    return TrainingVisualizer(run_dir)