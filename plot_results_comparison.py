# plot_results_comparison.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from typing import Dict, List, Tuple

# Estilos consistentes para cada algoritmo
ALG_STYLES = {
    "ddqn": {"color": "blue", "linestyle": "-"},
    "a3c": {"color": "green", "linestyle": "--"},
    "ppo": {"color": "red", "linestyle": "-."},
    "sac": {"color": "purple", "linestyle": ":"},
    "unknown": {"color": "gray", "linestyle": (0, (3, 5, 1, 5))} # Estilo para no encontrados
}

# Métricas a plotear y sus nombres de columna esperados
# Usamos tuplas (nombre_grafico, nombre_columna_csv)
TRAIN_METRICS_TO_PLOT: List[Tuple[str, str]] = [
    ("Recompensa Promedio Móvil (Entrenamiento)", "Recompensa_Promedio_Movil"),
    ("Pasos Promedio Móvil (Entrenamiento)", "Pasos_Promedio_Movil"),
    ("Metas Alcanzadas Promedio Móvil (Entrenamiento)", "Metas_Alcanzadas_Promedio_Movil"),
]

EVAL_METRICS_TO_PLOT: List[Tuple[str, str]] = [
    ("Recompensa por Episodio (Evaluación)", "Recompensa_Eval"),
    ("Pasos por Episodio (Evaluación)", "Pasos_Eval"),
    ("Tasa de Éxito por Episodio (Evaluación)", "Tasa_Exito"),
    ("Metas Alcanzadas por Episodio (Evaluación)", "Metas_Alcanzadas_Eval"),
]


def find_csv_files(results_dir: str, algorithms: List[str], mode: str = "train") -> Dict[str, str]:
    """
    Encuentra los archivos CSV para los algoritmos y modo especificados.
    Args:
        results_dir: Directorio donde buscar los CSVs.
        algorithms: Lista de nombres de algoritmos (ej. ["ddqn", "ppo"]).
        mode: "train" o "eval".
    Returns:
        Un diccionario {alg_name: path_to_csv}.
    """
    found_files = {}
    for alg in algorithms:
        expected_filename = f"{mode}_{alg}_metrics.csv"
        file_path = os.path.join(results_dir, expected_filename)
        if os.path.exists(file_path):
            found_files[alg] = file_path
        else:
            print(f"Advertencia: No se encontró el archivo {expected_filename} en {results_dir}")
    return found_files


def plot_comparison_graphs(
    data_files: Dict[str, str],
    metrics_to_plot: List[Tuple[str, str]],
    alg_styles: Dict[str, Dict[str, str]],
    output_dir: str,
    mode: str = "train"
):
    """
    Genera y guarda gráficos comparativos para las métricas especificadas.
    Args:
        data_files: Diccionario {alg_name: path_to_csv}.
        metrics_to_plot: Lista de tuplas (nombre_grafico, nombre_columna_csv).
        alg_styles: Diccionario de estilos por algoritmo.
        output_dir: Directorio donde guardar los gráficos.
        mode: "train" o "eval", para el eje x y el nombre del archivo.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="darkgrid")

    x_axis_label = 'Episodio' if mode == "train" else 'Episodio_Eval'

    for plot_title, metric_col_name in metrics_to_plot:
        plt.figure(figsize=(14, 7))
        any_data_plotted = False

        for alg_name, csv_path in data_files.items():
            try:
                df = pd.read_csv(csv_path)
                style = alg_styles.get(alg_name, alg_styles["unknown"])

                if metric_col_name in df.columns and x_axis_label in df.columns:
                    sns.lineplot(
                        x=x_axis_label,
                        y=metric_col_name,
                        data=df,
                        label=alg_name.upper(),
                        color=style["color"],
                        linestyle=style["linestyle"],
                        linewidth=2
                    )
                    any_data_plotted = True
                    print(f"Ploteando {metric_col_name} para {alg_name.upper()} desde {csv_path}")
                elif metric_col_name not in df.columns:
                    print(f"Advertencia: Columna '{metric_col_name}' no encontrada en {csv_path} para {alg_name.upper()}")
                elif x_axis_label not in df.columns:
                     print(f"Advertencia: Columna X '{x_axis_label}' no encontrada en {csv_path} para {alg_name.upper()}")

            except Exception as e:
                print(f"Error cargando o ploteando datos de {csv_path} para {alg_name.upper()}: {e}")

        if any_data_plotted:
            plt.title(plot_title, fontsize=16)
            plt.xlabel("Episodio", fontsize=12)
            plt.ylabel(metric_col_name.replace("_", " ").title(), fontsize=12)
            plt.legend(title="Algoritmo", fontsize=10)
            plt.tight_layout()
            plot_filename_safe = plot_title.lower().replace(" ", "_").replace("(", "").replace(")", "")
            save_path = os.path.join(output_dir, f"comparison_{mode}_{plot_filename_safe}.png")
            try:
                plt.savefig(save_path)
                print(f"Gráfico comparativo guardado en: {save_path}")
            except Exception as e:
                print(f"Error guardando gráfico {save_path}: {e}")
        else:
            print(f"No se encontraron datos para plotear el gráfico: {plot_title}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generar gráficos comparativos de resultados de RL.")
    parser.add_argument(
        "--results_dir",
        type=str,
        default=os.path.join("models", "results"), # Asume que main.py guarda en models/results/
        help="Directorio que contiene los archivos CSV de métricas (ej. 'models/results')."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join("models", "results", "comparison_plots"),
        help="Directorio donde guardar los gráficos generados."
    )
    parser.add_argument(
        "--algorithms",
        type=str,
        nargs='+',
        default=["ddqn", "a3c", "ppo", "sac"],
        help="Lista de algoritmos a incluir en la comparación."
    )
    args = parser.parse_args()

    print(f"Buscando archivos CSV en: {args.results_dir}")
    print(f"Algoritmos a comparar: {', '.join(args.algorithms)}")
    print(f"Los gráficos se guardarán en: {args.output_dir}")

    # Encontrar archivos de entrenamiento y evaluación
    train_csv_files = find_csv_files(args.results_dir, args.algorithms, mode="train")
    eval_csv_files = find_csv_files(args.results_dir, args.algorithms, mode="eval")

    if not train_csv_files and not eval_csv_files:
        print("No se encontraron archivos CSV para procesar. Saliendo.")
        return

    # Generar gráficos de entrenamiento
    if train_csv_files:
        print("\n--- Generando gráficos de ENTRENAMIENTO ---")
        plot_comparison_graphs(
            data_files=train_csv_files,
            metrics_to_plot=TRAIN_METRICS_TO_PLOT,
            alg_styles=ALG_STYLES,
            output_dir=args.output_dir,
            mode="train"
        )
    else:
        print("\nNo se encontraron archivos de entrenamiento para generar gráficos.")

    # Generar gráficos de evaluación
    if eval_csv_files:
        print("\n--- Generando gráficos de EVALUACIÓN ---")
        plot_comparison_graphs(
            data_files=eval_csv_files,
            metrics_to_plot=EVAL_METRICS_TO_PLOT,
            alg_styles=ALG_STYLES,
            output_dir=args.output_dir,
            mode="eval"
        )
    else:
        print("\nNo se encontraron archivos de evaluación para generar gráficos.")

    print("\nProceso de generación de gráficos completado.")


if __name__ == "__main__":
    main()