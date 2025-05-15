import pandas as pd
import os

def unify_codecarbon_reports(input_files: list, output_file: str):
    """
    Unifica múltiples archivos CSV de CodeCarbon en uno solo y añade una fila total.

    Args:
        input_files (list): Lista de rutas a los archivos CSV de entrada.
        output_file (str): Ruta para el archivo CSV unificado de salida.
    """
    all_data = []
    total_duration = 0.0
    total_emissions = 0.0
    total_energy_consumed = 0.0

    # Leer y concatenar todos los archivos
    for file_path in input_files:
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                all_data.append(df)
                # Sumar duraciones y emisiones para la fila total
                # Asegurarse de que las columnas existen antes de sumar
                if 'duration' in df.columns:
                    total_duration += df['duration'].sum()
                if 'emissions' in df.columns:
                    total_emissions += df['emissions'].sum()
                if 'energy_consumed' in df.columns:
                    total_energy_consumed += df['energy_consumed'].sum()
                print(f"Archivo '{file_path}' procesado.")
            except pd.errors.EmptyDataError:
                print(f"Advertencia: El archivo '{file_path}' está vacío y será ignorado.")
            except Exception as e:
                print(f"Error procesando el archivo '{file_path}': {e}")
        else:
            print(f"Advertencia: El archivo '{file_path}' no fue encontrado y será ignorado.")

    if not all_data:
        print("No se encontraron datos en los archivos de entrada. No se generará el archivo unificado.")
        return

    # Concatenar todos los DataFrames
    unified_df = pd.concat(all_data, ignore_index=True)

    # Crear la fila "Total"
    # Tomamos los valores de la primera fila para las columnas que no se suman (país, os, etc.)
    # Es una simplificación; si estos valores varían, podrías querer manejarlos diferente.
    if not unified_df.empty:
        total_row_data = unified_df.iloc[0].copy() # Copiar la primera fila como base
        # Sobrescribir los valores que queremos sumar o cambiar
        total_row_data['timestamp'] = "TOTAL"
        total_row_data['project_name'] = "Total_All_Trainings"
        total_row_data['run_id'] = "N/A"
        total_row_data['experiment_id'] = "N/A"
        total_row_data['duration'] = total_duration
        total_row_data['emissions'] = total_emissions
        total_row_data['energy_consumed'] = total_energy_consumed
        
        # Calcular emisiones_rate promedio ponderado por duración si es necesario,
        # o simplemente dejarlo como N/A o el de la primera fila.
        # Aquí, lo dejaremos como N/A o puedes calcularlo si lo necesitas.
        if total_duration > 0:
            total_row_data['emissions_rate'] = total_emissions / total_duration
        else:
            total_row_data['emissions_rate'] = 0.0 # o pd.NA

        # Para otras columnas numéricas que no son sumas directas, podrías poner NaN o un promedio
        numeric_cols_to_na = ['cpu_power', 'gpu_power', 'ram_power', 'cpu_energy', 'gpu_energy', 'ram_energy']
        for col in numeric_cols_to_na:
            if col in total_row_data:
                 total_row_data[col] = pd.NA # o 0.0, o podrías calcular un promedio ponderado

        # Convertir la serie a un DataFrame de una sola fila
        total_row_df = pd.DataFrame([total_row_data])

        # Añadir la fila total al DataFrame unificado
        unified_df = pd.concat([unified_df, total_row_df], ignore_index=True)
    else:
        print("DataFrame unificado está vacío, no se añadirá fila total.")


    # Guardar el DataFrame unificado en un nuevo archivo CSV
    try:
        unified_df.to_csv(output_file, index=False)
        print(f"Archivo unificado guardado en: '{output_file}'")
    except Exception as e:
        print(f"Error al guardar el archivo unificado: {e}")

if __name__ == "__main__":
    # Lista de tus archivos CSV de entrada
    path = "models/results/codecarbon_reports"
    input_csv_files = [
        f"{path}/emissions_ddqn_train.csv",
        f"{path}/emissions_ppo_train.csv",
        f"{path}/emissions_sac_train.csv"
    ]
    
    # Nombre del archivo CSV de salida
    output_csv_file = f"{path}/unified_emissions_agents_train.csv"
    
    unify_codecarbon_reports(input_csv_files, output_csv_file)
