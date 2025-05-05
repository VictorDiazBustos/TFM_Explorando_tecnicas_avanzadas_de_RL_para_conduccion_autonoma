import numpy as np
import torch
import argparse
import os
import time
from collections import deque
from typing import Dict, Optional

from sumo_environment import SUMOEnvironment
from agents.ddqn_agent import DDQNAgent
from agents.a3c_agent import A3CAgent
from agents.ppo_agent import PPOAgent
from agents.sac_agent import SACAgent
from agents.base_agent import BaseAgent
from config import CONFIG
import pandas as pd
import matplotlib.pyplot as plt


def create_env(config, gui=False, is_eval=False, STATE_DIM=52, port: Optional[int] = None):
    """Crea una instancia del entorno SUMO."""
    sumo_cfg = config['env']['sumo_config_eval'] if is_eval else config['env']['sumo_config_train']
    return SUMOEnvironment(
        sumo_config=sumo_cfg,
        max_steps=config['env']['max_steps'],
        gui=gui,
        STATE_DIM=STATE_DIM,
        port=port
    )

def create_agent(alg_name, state_dim, action_dim, config, clipping=True, env_creator=None):
    """Crea una instancia de agente basada en el nombre del algoritmo."""
    if alg_name == "ddqn":
        return DDQNAgent(state_dim, action_dim, config['ddqn'], clipping)
    elif alg_name == "a3c":
        if env_creator is None:
            raise ValueError("A3CAgent requiere una función env_creator.")
        # Pasar la configuración específica de A3C y el env_creator
        return A3CAgent(state_dim, action_dim, config['a3c'], clipping, env_creator)
    elif alg_name == "ppo":
        return PPOAgent(state_dim, action_dim, config['ppo'], clipping)
    elif alg_name == "sac":
        # Asegurarse de que SACAgent sea la versión Discreta corregida
        return SACAgent(state_dim, action_dim, config['sac'])
    else:
        raise ValueError(f"Algoritmo desconocido: {alg_name}")

def plot_metrics(df: pd.DataFrame, metrics: list[str], save_dir: str, prefix: str):
    """Genera y guarda gráficos para las métricas especificadas."""
    print(f"Generando gráficos para {prefix}...")
    sns.set_theme(style="darkgrid")
    for metric in metrics:
        if metric in df.columns:
            plt.figure(figsize=(12, 6))
            # Usar 'Episodio' como eje x si está disponible, sino el índice
            x_axis = 'Episodio' if 'Episodio' in df.columns else df.index
            sns.lineplot(x=x_axis, y=metric, data=df)
            plt.title(f'{prefix.replace("_", " ").title()} - {metric.replace("_", " ").title()}')
            plt.xlabel('Episodio' if 'Episodio' in df.columns else 'Entrada')
            plt.ylabel(metric.replace("_", " ").title())
            plot_filename = os.path.join(save_dir, f"{prefix}_{metric}.png")
            try:
                plt.savefig(plot_filename)
                print(f"Gráfico guardado: {plot_filename}")
            except Exception as e:
                 print(f"Error guardando gráfico {plot_filename}: {e}")
            plt.close() # Cerrar la figura para liberar memoria
        else:
            print(f"Advertencia: Métrica '{metric}' no encontrada en el DataFrame para {prefix}.")

def train_std_agent(agent: BaseAgent, env: SUMOEnvironment, config: Dict, alg_name: str, save_dir: str):
    """Bucle de entrenamiento estándar para DDQN, PPO, SAC con guardado exponencial."""
    print(f"--- Iniciando Entrenamiento: {alg_name.upper()} ---")
    num_episodes = config['training']['num_episodes']
    print_interval = config['training']['print_interval']

    results_dir = os.path.join(save_dir, "results") # Directorio para CSVs y gráficos
    os.makedirs(results_dir, exist_ok=True)
    csv_filename = os.path.join(results_dir, f"train_{alg_name}_metrics.csv")

    # Métricas a registrar
    metrics_log: Dict[str, list] = {
        'Episodio': [],
        'Recompensa_Episodio': [],
        'Pasos_Episodio': [],
        'Pasos_Totales': [],
        'Recompensa_Promedio_Movil': [],
        'Metas_Alcanzadas_Episodio': [],
        'Metas_Alcanzadas_Promedio_Movil': [],
    }
    
    # Inicializar listas para pérdidas comunes (si existen en loss_dict)
    possible_losses = ['loss', 'policy_loss', 'value_loss', 'actor_loss', 'critic_loss', 'entropy_loss', 'alpha_loss', 'alpha', 'epsilon']
    for loss_name in possible_losses:
        metrics_log[loss_name] = []

    episode_rewards = deque(maxlen=print_interval) # Almacenar recompensas recientes para promedio
    episode_goals = deque(maxlen=print_interval)   # Almacenar las metas alcanzadas recientes para promedio
    total_steps = 0
    next_save_episode = 1 # Inicializar el próximo número de episodio para guardar

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        episode_reward = 0
        step = 0
        goals_reached = 0
        done = False
        if hasattr(agent, 'memory_ready'): # Reiniciar flag de memoria de PPO si el agente lo tiene
            agent.memory_ready = False
        
        episode_losses = {loss_name: [] for loss_name in possible_losses} # Para promediar pérdidas del episodio

        while not done:
            if alg_name == "ppo" and agent.memory_ready:
                try:
                    loss_dict = agent.train(last_state=state, last_done=done)
                    for key, value in loss_dict.items():
                        if key in episode_losses:
                            episode_losses[key].append(value)
                except Exception as e_train:
                    print(f"Error durante agent.train() [PPO Pre-Action] en episodio {episode}: {e_train}")

            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            if info.get("segment_goal_reached", False):
                goals_reached += 1

            episode_reward += reward
            total_steps += 1
            step += 1

            if alg_name == "ppo":
                # PPO almacena el resultado después del paso
                agent.store_outcome(reward, done)
            elif alg_name in ["ddqn", "sac"]:
                # DDQN/SAC almacenan la transición completa
                agent.store_transition(state, action, reward, next_state, done)
                # Verificar si la memoria del agente (buffer) tiene suficientes muestras
                memory_attr = getattr(agent, 'memory', None)
                batch_size_key = config[alg_name].get('batch_size')
                if memory_attr is not None and batch_size_key is not None and len(memory_attr) >= batch_size_key:
                    # TODO:
                    loss_dict = agent.train()
                    # try:
                    #     loss_dict = agent.train()
                    # except Exception as e:
                    #     print(f"Error durante agent.train() para {alg_name}: {e}")

            state = next_state

            if done:
                # Si PPO terminó el episodio antes de n_steps, entrenar con datos parciales
                if alg_name == "ppo" and not agent.memory_ready:
                    # Verificar si hay alg con qué entrenar
                    if hasattr(agent, 'memory') and len(agent.memory.get('states', [])) > 0:
                        print(f"PPO Episodio {episode} terminó temprano, entrenando con datos parciales ({len(agent.memory['states'])} pasos)...")
                        try: 
                            loss_dict = agent.train(last_state=state, last_done=done)
                            for key, value in loss_dict.items():
                                if key in episode_losses:
                                    episode_losses[key].append(value)
                        except Exception as e:
                            print(f"Error durante agent.train() [PPO Post-Done] en episodio {episode}: {e}")
                    if hasattr(agent, 'memory_ready'):
                        agent.memory_ready = False
                break

        # Registro de Métricas del Episodio
        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards)
        episode_goals.append(goals_reached)
        avg_goals = np.mean(episode_goals)
        metrics_log['Episodio'].append(episode)
        metrics_log['Recompensa_Episodio'].append(episode_reward)
        metrics_log['Pasos_Episodio'].append(step)
        metrics_log['Pasos_Totales'].append(total_steps)
        metrics_log['Recompensa_Promedio_Movil'].append(avg_reward)
        metrics_log['Metas_Alcanzadas_Episodio'].append(goals_reached)
        metrics_log['Metas_Alcanzadas_Promedio_Movil'].append(avg_goals)

        # Registrar promedio de pérdidas del episodio (o último valor si solo hay uno)
        for loss_name in possible_losses:
            values = episode_losses.get(loss_name, [])
            if values:
                metrics_log[loss_name].append(np.mean(values))
            else:
                # Si no hubo entrenamiento o no se devolvió esa pérdida, añadir NaN o un marcador
                # Asegurarse que todas las listas tengan la misma longitud
                metrics_log[loss_name].append(np.nan)

        if episode % print_interval == 0:
            loss_str = ", ".join([f"{k}: {v[-1]:.3f}" for k, v in metrics_log.items() if k not in ['Episodio', 'Recompensa_Episodio', 'Pasos_Episodio', 'Pasos_Totales', 'Recompensa_Promedio_Movil'] and not np.isnan(v[-1])])
            print(f"Alg: {alg_name.upper()} | Episodio: {episode}/{num_episodes} | Pasos: {step} | Pasos Totales: {total_steps} | Recompensa Ep: {episode_reward:.2f} | Recompensa Prom (Últimos {print_interval}): {avg_reward:.2f} | Metas Ep: {goals_reached} | Pérdida {loss_str}")
            if 'loss_dict' in locals():
                print(f"\tLosses: {loss_dict}")

        # Lógica de Guardado Exponencial
        if episode == next_save_episode:
            checkpoint_filename = f"{alg_name}_ep{episode}_model.pth"
            checkpoint_path = os.path.join(save_dir, checkpoint_filename)
            try:
                agent.save(checkpoint_path)
                print(f"Punto de control guardado en {checkpoint_path}")
            except Exception as e:
                print(f"Error al guardar el punto de control en {checkpoint_path}: {e}")
            if next_save_episode >= 1:
                next_save_episode *= 10
            else:
                next_save_episode = 10

            # Guardar CSV parcial en cada checkpoint
            try:
                df_partial = pd.DataFrame(metrics_log)
                df_partial.to_csv(csv_filename, index=False)
                print(f"CSV parcial guardado en {csv_filename}")
            except Exception as e_csv:
                print(f"Error guardando CSV parcial: {e_csv}")


    final_filename = f"{alg_name}_final_model.pth"
    final_save_path = os.path.join(save_dir, final_filename)
    try:
        agent.save(final_save_path)
        print(f"Modelo final guardado en {final_save_path}")
    except Exception as e:
        print(f"Error al guardar el modelo final en {final_save_path}: {e}")

    # Guardar CSV final
    try:
        df_final = pd.DataFrame(metrics_log)
        df_final.to_csv(csv_filename, index=False)
        print(f"CSV final guardado en {csv_filename}")
        # Generar gráficos
        plot_metrics_list = ['Recompensa_Episodio', 'Recompensa_Promedio_Movil', 'Pasos_Episodio', 'Metas_Alcanzadas_Episodio', 'Metas_Alcanzadas_Promedio_Movil']
        # Añadir pérdidas que realmente se registraron
        plot_metrics_list.extend([k for k, v in metrics_log.items() if k in possible_losses and any(not np.isnan(val) for val in v)])
        plot_metrics(df_final, plot_metrics_list, results_dir, f"train_{alg_name}")
    except Exception as e_final:
        print(f"Error guardando CSV final o generando gráficos: {e_final}")

    print(f"--- Entrenamiento Finalizado: {alg_name.upper()} ---")
    env.close()


def train_a3c_agent(agent: A3CAgent, config: Dict, save_dir: str):
    """Bucle de entrenamiento específico para A3C."""
    print("--- Iniciando Entrenamiento: A3C ---")
    total_timesteps = config['training']['total_timesteps_a3c']
    save_path = os.path.join(save_dir, "a3c_final_model.pth")
    start_time = time.time()

    try:
        agent.start_training()
        print(f"Workers A3C iniciados. Entrenando por aprox. {total_timesteps} pasos entre todos los workers...")
        # Duración simplificada basada en tiempo para el control del entrenamiento A3C
        training_duration_seconds = config['training'].get('a3c_duration_sec', 3600) # Por defecto 1 hora
        print(f"Ejecutando durante {training_duration_seconds} segundos...")
        time.sleep(training_duration_seconds)

    except KeyboardInterrupt:
        print("Entrenamiento interrumpido por el usuario.")
    finally:
        print("Deteniendo workers A3C...")
        agent.stop_training()
        end_time = time.time()
        print(f"Workers A3C detenidos. Duración del entrenamiento: {end_time - start_time:.2f} segundos.")
        try:
            agent.save(save_path)
            print(f"Modelo final A3C guardado en {save_path}")
        except Exception as e:
            print(f"Error al guardar el modelo final A3C en {save_path}: {e}")

    print("--- Entrenamiento Finalizado: A3C ---")

# Definir éxito (ejemplo: no colisión, no teleport, y no llegó a max_steps)
def check_success(info, done, steps, max_steps):
    terminal_reason = info.get('terminal', '')
    if done and terminal_reason not in ['collision', 'teleported', 'forced_collision', 'unknown_failure'] and steps < max_steps:
            return 1
    # Considerar 'goal_reached' explícitamente si está en info
    elif done and terminal_reason == 'goal_reached':
            return 1
    elif done and info.get('new_route_assigned') and steps < max_steps: # Si sigue corriendo tras asignar ruta
            # Podría considerarse éxito parcial o necesitar otra lógica
            return 1 # Considerémoslo éxito por ahora si no hubo fallo grave
    return 0

def evaluate_agent(agent: BaseAgent, env: SUMOEnvironment, config: dict, alg_name: str, save_dir: str):
    """Evalúa un agente entrenado."""
    print(f"\n--- Iniciando Evaluación: {alg_name.upper()} ---")
    num_episodes = config['evaluation']['num_episodes']
    total_rewards = []
    results_dir = os.path.join(save_dir, "results") # Directorio para CSVs y gráficos
    os.makedirs(results_dir, exist_ok=True)
    csv_filename = os.path.join(results_dir, f"eval_{alg_name}_metrics.csv")
    
    # Métricas de evaluación
    eval_metrics_log: Dict[str, list] = {
        'Episodio_Eval': [],
        'Recompensa_Eval': [],
        'Pasos_Eval': [],
        'Colisiones_Eval': [],
        'Teleportaciones_Eval': [],
        'Paradas_Emergencia_Eval': [],
        'Tasa_Exito': [], # 1 si éxito, 0 si fallo
        'Metas_Alcanzadas_Eval': [],
    }

    agent_module = None
    if hasattr(agent, 'network') and isinstance(agent.network, torch.nn.Module):
        agent_module = agent.network
    elif hasattr(agent, 'policy_net') and isinstance(agent.policy_net, torch.nn.Module):
        agent_module = agent.policy_net
    elif hasattr(agent, 'actor') and isinstance(agent.actor, torch.nn.Module):
        agent_module = agent.actor

    if agent_module:
        try:
            agent_module.eval()
            print("Agente puesto en modo evaluación.")
        except Exception as e:
            print(f"Advertencia: No se pudo poner el agente en modo eval: {e}")

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        episode_reward = 0
        step = 0
        done = False
        episode_collisions = 0
        episode_teleports = 0
        episode_emergency_stops = 0
        episode_info = {}
        goals_reached = 0
        
        while not done:
            action = 0 # Acción por defecto
            try:
                if hasattr(agent, 'select_action') and 'evaluate' in agent.select_action.__code__.co_varnames:
                    action = agent.select_action(state, evaluate=True)
                else:
                    action = agent.select_action(state)
            except Exception as e:
                print(f"Error seleccionando acción en evaluación: {e}. Usando acción por defecto 0.")

            next_state, reward, done, info = env.step(action)

            if info.get("segment_goal_reached", False):
                goals_reached += 1

            episode_reward += reward
            state = next_state
            step += 1
            episode_info = info

            if info.get('terminal') == 'collision' or info.get('terminal') == 'forced_collision':
                episode_collisions += 1
            elif info.get('terminal') == 'teleported':
                episode_teleports += 1
            if info.get('emergency_stop'):
                episode_emergency_stops += 1
            if done:
                break

        # Registrar métricas del episodio de evaluación
        success = check_success(episode_info, done, step, config['env']['max_steps'])
        eval_metrics_log['Episodio_Eval'].append(episode)
        eval_metrics_log['Recompensa_Eval'].append(episode_reward)
        eval_metrics_log['Pasos_Eval'].append(step)
        eval_metrics_log['Colisiones_Eval'].append(episode_collisions)
        eval_metrics_log['Teleportaciones_Eval'].append(episode_teleports)
        eval_metrics_log['Paradas_Emergencia_Eval'].append(episode_emergency_stops)
        eval_metrics_log['Tasa_Exito'].append(success)
        eval_metrics_log['Metas_Alcanzadas_Eval'].append(goals_reached)

        print(f"Alg: {alg_name.upper()} | Eval Ep: {episode}/{num_episodes} | Pasos: {step} | Recompensa: {episode_reward:.2f} | Metas: {goals_reached} | Éxito: {'Sí' if success else 'No'} | Col: {episode_collisions} | Tel: {episode_teleports}")

    # Poner el agente de nuevo en modo entrenamiento si es necesario
    if agent_module:
        try:
            agent_module.train()
            print("Agente puesto de nuevo en modo entrenamiento.")
        except Exception as e:
             print(f"Advertencia: No se pudo poner el agente en modo train: {e}")

    # Calcular y mostrar promedios finales
    avg_reward = np.mean(eval_metrics_log['Recompensa_Eval'])
    std_reward = np.std(eval_metrics_log['Recompensa_Eval'])
    avg_steps = np.mean(eval_metrics_log['Pasos_Eval'])
    avg_success_rate = np.mean(eval_metrics_log['Tasa_Exito']) * 100 # En porcentaje
    avg_collisions = np.mean(eval_metrics_log['Colisiones_Eval'])
    avg_teleports = np.mean(eval_metrics_log['Teleportaciones_Eval'])
    avg_goals_reached = np.mean(eval_metrics_log['Metas_Alcanzadas_Eval'])

    print(
        f"--- Evaluación Finalizada: {alg_name.upper()} ---\n"
        f"Recompensa Promedio: {avg_reward:.2f} +/- {std_reward:.2f}\n"
        f"Pasos Promedio: {avg_steps:.1f}\n"
        f"Tasa de Éxito Promedio: {avg_success_rate:.1f}%\n"
        f"Colisiones Promedio: {avg_collisions:.2f}\n"
        f"Teleportaciones Promedio: {avg_teleports:.2f}"
        f"Metas por Episodio Promedio: {avg_goals_reached:.2f}\n"
    )

    # Guardar CSV de evaluación
    try:
        df_eval = pd.DataFrame(eval_metrics_log)
        df_eval.to_csv(csv_filename, index=False)
        print(f"CSV de evaluación guardado en {csv_filename}")
        plot_metrics(df_eval, ['Recompensa_Eval', 'Pasos_Eval', 'Tasa_Exito', 'Colisiones_Eval', 'Metas_Alcanzadas_Eval'], results_dir, f"eval_{alg_name}")
    except Exception as e_final:
        print(f"Error guardando CSV de evaluación o generando gráficos: {e_final}")

    env.close()
    return avg_reward

def main():
    parser = argparse.ArgumentParser(description="Entrenar o evaluar agentes RL para conducción autónoma en SUMO.")
    parser.add_argument("--alg", type=str, required=True, choices=["ddqn", "a3c", "ppo", "sac"], help="Algoritmo a usar.")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"], help="Modo: train (entrenar) o eval (evaluar).")
    parser.add_argument("--episodes", type=int, help="Número de episodios para entrenamiento/evaluación (sobrescribe config para agentes estándar).")
    parser.add_argument("--load_path", type=str, default=None, help="Ruta para cargar un modelo pre-entrenado (ej., models/ddqn_ep100_model.pth). Requerido para eval a menos que se use el guardado final por defecto.")
    parser.add_argument("--save_dir", type=str, default="models", help="Directorio para guardar modelos entrenados y puntos de control.")
    parser.add_argument("--gui", action="store_true", help="Habilitar GUI de SUMO durante entrenamiento/evaluación.")
    parser.add_argument("--clipping", action="store_true", default=True, help="Habilitar recorte de gradientes.")

    args = parser.parse_args()

    # Configuración Inicial
    os.makedirs(args.save_dir, exist_ok=True)
    results_subdir = os.path.join(args.save_dir, "results")
    os.makedirs(results_subdir, exist_ok=True)


    # Sobrescribir episodios de config si se proporcionan
    if args.episodes is not None:
        if args.mode == "train" and args.alg != "a3c":
            CONFIG['training']['num_episodes'] = args.episodes
        elif args.mode == "eval":
            CONFIG['evaluation']['num_episodes'] = args.episodes

    # Obtener dimensiones del entorno
    print("Inicializando entorno para obtener dimensiones...")

    # Usar un entorno sin GUI para la configuración para evitar ventanas emergentes innecesarias
    
    BASE_STATE_COMPONENTS = 12
    MAX_NEARBY_VEHICLES = 8
    FEATURES_PER_VEHICLE = 5
    VEHICLE_STATE_COMPONENTS = MAX_NEARBY_VEHICLES * FEATURES_PER_VEHICLE
    DEFAULT_STATE_DIM = BASE_STATE_COMPONENTS + VEHICLE_STATE_COMPONENTS
    DEFAULT_ACTION_DIM = 8
    temp_env = None
    try:
        temp_env = create_env(CONFIG, gui=False, is_eval=False, STATE_DIM=DEFAULT_STATE_DIM)
        initial_state = temp_env.reset()

        # Asegurar que state_dim se infiera correctamente (debería ser un array plano)
        if isinstance(initial_state, np.ndarray):
            state_dim = initial_state.shape[0]
        else:
            print("Advertencia: Formato de estado inicial inesperado. Intentando obtener dimensión...")
            # Intentar obtener la longitud si es una lista o tupla
            try:
                state_dim = len(initial_state)
            except TypeError:
                print("Error: No se pudo determinar state_dim. Estableciendo a {DEFAULT_STATE_DIM}.")
                state_dim = DEFAULT_STATE_DIM
                
        action_dim = len(temp_env.ACTION_NAMES)
        print(f"Dimensión de estado: {state_dim}, Dimensión de acción: {action_dim}")
    except Exception as e:
        state_dim = DEFAULT_STATE_DIM
        action_dim = DEFAULT_ACTION_DIM
        print(f"Error al inicializar el entorno temporal para obtener dimensiones: {e}")
        print(f"Estableciendo dimensiones por defecto: state_dim={state_dim}, action_dim={action_dim}")
    finally:
        if temp_env:
            try:
                temp_env.close() # Cerrar entorno temporal
            except Exception as e_close:
                print(f"Error cerrando entorno temporal: {e_close}")

    # Construir ruta relativa al escenario desde el script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_filename = CONFIG['env']['sumo_config_eval'] if args.mode == 'eval' else CONFIG['env']['sumo_config_train']
    absolute_config_path = os.path.join(script_dir, config_filename)

    # Verificar que el archivo existe
    if not os.path.exists(absolute_config_path):
        print(f"¡ERROR CRÍTICO! No se encuentra el archivo de configuración SUMO en la ruta absoluta:")
        print(absolute_config_path)
        print("Verifica la ruta relativa 'scene' y el nombre de archivo en config.py")
        exit(1)
    else:
        print(f"Ruta absoluta al config SUMO: {absolute_config_path}")

    # env_creator = lambda gui_flag=False, port=None: create_env(CONFIG, gui=gui_flag, is_eval=(args.mode == 'eval'), port=port)
    env_creator = lambda gui_flag=False, port=None: SUMOEnvironment(
        sumo_config=absolute_config_path,
        max_steps=CONFIG['env']['max_steps'],
        gui=gui_flag,
        port=port,
        STATE_DIM=state_dim
    )

    print(f"Creando agente: {args.alg.upper()}")
    try:
        # Crear agente
        # agent = create_agent(args.alg, state_dim, action_dim, CONFIG, env_creator=env_creator)
        agent = create_agent(
            args.alg,
            state_dim,
            action_dim,
            CONFIG,
            clipping=args.clipping,
            env_creator=env_creator if args.alg == 'a3c' else None # Pasar creator solo a A3C
        )
    except Exception as e:
        print(f"Error creando el agente '{args.alg}': {e}")
        exit(1)

    # Ejecución
    if args.mode == "train":
        print(f"Iniciando modo entrenamiento para {args.alg.upper()}...")
        # Cargar modelo si se proporciona ruta para continuar entrenamiento
        if args.load_path and os.path.exists(args.load_path):
            try:
                agent.load(args.load_path)
                print(f"Modelo pre-entrenado cargado desde {args.load_path} para continuar entrenamiento.")
                # Opcional: Intentar inferir next_save_episode basado en el nombre de archivo cargado si es posible
                # O reiniciar el calendario de guardado exponencial
            except Exception as e:
                print(f"No se pudo cargar el modelo desde {args.load_path}. Iniciando entrenamiento desde cero. Error: {e}")

        # Crear entorno de entrenamiento
        train_env = None
        if args.alg != "a3c":
            # A3C crea sus propios entornos vía env_creator
            try:
                train_env = env_creator(gui_flag=args.gui)
            except Exception as e:
                print(f"Error creando el entorno de entrenamiento: {e}")
                exit(1)


        if args.alg == "a3c":
            train_a3c_agent(agent, CONFIG, args.save_dir)
        elif args.alg in ["ddqn", "ppo", "sac"] and train_env is not None:
            train_std_agent(agent, train_env, CONFIG, args.alg, args.save_dir)
        else:
            print(f"Lógica de entrenamiento para {args.alg} no implementada o entorno no creado.")
            if train_env:
                train_env.close()


    elif args.mode == "eval":
        print(f"Iniciando modo evaluación para {args.alg.upper()}...")
        if args.load_path:
            load_model_path = args.load_path
        else:
            # Por defecto, cargar el modelo final si no se da una ruta específica
            load_model_path = os.path.join(args.save_dir, f"{args.alg}_final_model.pth")
            print(f"No se especificó --load_path, intentando cargar modelo final por defecto: {load_model_path}")

        if not os.path.exists(load_model_path):
            print(f"Error: Archivo de modelo no encontrado en {load_model_path}. Usa --load_path para especificar el punto de control correcto (ej., models/{args.alg}_ep100_model.pth) o el modelo final.")
            exit(1)

        try:
            agent.load(load_model_path)
            print(f"Modelo cargado desde {load_model_path} para evaluación.")
        except Exception as e:
            print(f"Error cargando modelo desde {load_model_path}: {e}")
            exit(1)

        # Crear entorno de evaluación
        eval_env = None
        try:
            eval_env = env_creator(gui_flag=args.gui)
        except Exception as e:
            print(f"Error creando el entorno de evaluación: {e}")
            exit(1)

        if eval_env:
            evaluate_agent(agent, eval_env, CONFIG, args.alg, args.save_dir)

if __name__ == "__main__":
    main()
