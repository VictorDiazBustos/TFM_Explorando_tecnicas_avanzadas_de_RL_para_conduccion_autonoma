# TFM_Explorando_tecnicas_avanzadas_de_RL_para_conduccion_autonoma

Este repositorio contiene el código fuente y los materiales asociados al Trabajo Final de Máster (TFM) para el Máster Universitario en Ciencia de Datos de la Universitat Oberta de Catalunya (UOC).

## Índice
- [Descripción](#descripción)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Requisitos](#requisitos)
- [Instrucciones de instalación](#instrucciones-de-instalación)
- [Cómo ejecutar el proyecto](#cómo-ejecutar-el-proyecto)
- [Ejemplos de uso (visualización)](#ejemplos-de-uso-visualización)
- [Resultados obtenidos](#resultados-obtenidos)

## Descripción

El presente proyecto investiga la aplicación de diversas técnicas avanzadas de Aprendizaje por Refuerzo Profundo (DRL) para entrenar un agente capaz de realizar tareas de conducción autónoma en un entorno de simulación de tráfico urbano. El objetivo principal es desarrollar un sistema donde un vehículo aprenda a navegar de manera segura y eficiente, tomando decisiones como cambios de carril, gestión de velocidad y maniobras en intersecciones.

Se utiliza el simulador de tráfico SUMO (Simulation of Urban MObility) como entorno de entrenamiento y evaluación, interactuando con él a través de su API TraCI. Se implementan y comparan varios algoritmos de DRL, incluyendo DDQN, A3C, PPO y SAC, analizando su capacidad de aprendizaje, estabilidad y rendimiento final en tareas de conducción simulada.

El proyecto aborda los desafíos inherentes al diseño de funciones de recompensa adecuadas, la representación del estado del entorno y la estabilidad de los algoritmos de DRL en un dominio complejo y dinámico como la conducción autónoma.

## Estructura del repositorio

A continuación, se describe la estructura principal de carpetas y archivos del repositorio:

```
.
├── agents/                 # Implementaciones de los algoritmos de RL
│   ├── base_agent.py       # Clase base abstracta para los agentes
│   ├── ddqn_agent.py       # Agente Dueling Double Deep Q-Network
│   ├── a3c_agent.py        # Agente Asynchronous Advantage Actor-Critic
│   ├── ppo_agent.py        # Agente Proximal Policy Optimization
│   └── sac_agent.py        # Agente Soft Actor-Critic (versión discreta)
├── escenario/              # Archivos de configuración del escenario SUMO
│   ├── osm.net.xml.gz      # Red vial del escenario
│   ├── osm.poly.xml.gz     # Elementos visuales del escenario
│   └── osm.sumocfg         # Archivo de configuración principal de SUMO
├── frames/                 # (Generado durante la ejecución con GUI) Capturas de pantalla para video
├── models/                 # (Generado durante la ejecución) Modelos entrenados y puntos de control
│   └── results/            # (Generado durante la ejecución) Archivos CSV y gráficos de métricas
├── .gitignore              # Archivos y carpetas a ignorar por Git
├── config.py               # Archivo de configuración centralizado para hiperparámetros
├── main.py                 # Script principal para entrenamiento y evaluación de agentes
├── requirements.txt        # Dependencias del proyecto Python
├── sumo_environment.py     # Clase de Python para interactuar con el entorno SUMO
├── episode.mp4             # (Generado durante la ejecución con GUI) Video generado de un episodio 
├── train_a3c.sh            # Script de shell para entrenar A3C
├── train_all.sh            # Script de shell para entrenar todos los algoritmos secuencialmente
├── train_ddqn.sh           # Script de shell para entrenar DDQN
├── train_ppo.sh            # Script de shell para entrenar PPO
├── train_sac.sh            # Script de shell para entrenar SAC
├── eval_a3c.sh             # Script de shell para evaluar A3C
├── eval_all.sh             # Script de shell para evaluar todos los algoritmos secuencialmente
├── eval_ddqn.sh            # Script de shell para evaluar DDQN
├── eval_ppo.sh             # Script de shell para evaluar PPO
├── eval_sac.sh             # Script de shell para evaluar SAC
└── README.md               # Este archivo
```

## Requisitos

### Software:

*   **Python:** Versión 3.10 o superior (desarrollado y probado con Python 3.12).
*   **SUMO:** Versión 1.20.0 (Simulation of Urban MObility). Es crucial usar esta versión o una compatible con la API TraCI utilizada.
    *   Descarga desde [SUMO Download Page](https://sumo.dlr.de/docs/Downloads.php).

### Librerías Python:

Las dependencias principales se listan en el archivo `requirements.txt`. Las más importantes son:

*   `torch`: PyTorch para las redes neuronales.
*   `numpy`: Para operaciones numéricas.
*   `pandas`: Para el manejo y guardado de métricas en CSV.
*   `matplotlib` y `seaborn`: Para la generación de gráficos.
*   `traci`: Cliente Python para la API de SUMO (generalmente viene con la instalación de SUMO).

## Instrucciones de instalación

Sigue estos pasos para configurar el entorno y ejecutar el proyecto:

1.  **Clonar el repositorio:**
    ```bash
    git clone git@github.com:VictorDiazBustos/TFM_Explorando_tecnicas_avanzadas_de_RL_para_conduccion_autonoma.git
    ```

2.  **Instalar SUMO:**
    Descarga e instala SUMO v1.20.0 desde el [sitio oficial de SUMO](https://sumo.dlr.de/docs/Downloads.php) siguiendo las instrucciones para tu sistema operativo.
    Asegúrate de que los ejecutables de SUMO (especialmente `sumo` y `sumo-gui`) y las herramientas de TraCI (directorio `tools`) estén en el PATH de tu sistema, o que la variable de entorno `SUMO_HOME` esté configurada apuntando al directorio de instalación de SUMO. El script `sumo_environment.py` intentará encontrar TraCI, pero tener `SUMO_HOME` configurado suele ser lo más robusto.

3.  **Crear un entorno virtual (recomendado):**
    Es altamente recomendable usar un entorno virtual para aislar las dependencias del proyecto.
    ```bash
    conda env create -f environment.yml
    conda activate sumo_env
    ```

## Cómo ejecutar el proyecto

El script principal para interactuar con el proyecto es `main.py`.

### Entrenamiento de agentes

El script principal para interactuar con el proyecto es `main.py`. Los hiperparámetros y configuraciones del entorno se pueden ajustar en `config.py`.

Se proporcionan scripts de shell (`.sh`) para facilitar la ejecución de los entrenamientos y evaluaciones. Antes de ejecutar los scripts de shell, asegúrate de que tienen permisos de ejecución:
```bash
chmod +x *.sh
```

Puedes entrenar un algoritmo específico o todos secuencialmente:

```bash
./train_ddqn.sh
./train_ppo.sh
./train_sac.sh
# ./train_a3c.sh  (Nota: A3C tiene problemas de concurrencia pendientes)
./train_all.sh    # Entrena DDQN, PPO, y SAC
```

Estos scripts invocan a main.py con argumentos predefinidos. Puedes editarlos para cambiar los parámetros.

Los modelos se guardarán en el directorio especificado por `--save_dir` (o models/ por defecto), con puntos de control exponenciales (ep10, ep100, ep1000, etc.) y un modelo final. Las métricas de entrenamiento se guardarán en `<save_dir>/results/train_<alg_name>_metrics.csv` y se generarán gráficos correspondientes.

Scripts para evaluar los agentes:

```bash
./eval_ddqn.sh
./eval_ppo.sh
./eval_sac.sh
# ./eval_a3c.sh
./eval_all.sh # Evalua los modelos finales de DDQN, PPO, SAC
```

Las métricas de evaluación se guardarán en `<save_dir>/results/eval_<alg_name>_metrics.csv` y se generarán gráficos.

## Ejemplos de uso (visualización)

Durante la ejecución con el argumento `--gui` (o usando los scripts .sh que lo habilitan), se abrirá la interfaz gráfica de SUMO, permitiendo observar el comportamiento del vehículo controlado por el agente. Si se generan frames en la carpeta `frames/`, se creará un vídeo `episode.mp4` al finalizar una ejecución con GUI.

## Resultados obtenidos

Los resultados detallados, incluyendo curvas de aprendizaje, métricas de evaluación y un análisis comparativo del rendimiento de los diferentes algoritmos, se encuentran documentados en la memoria del TFM.
