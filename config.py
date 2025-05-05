
CONFIG = {
    "env": {
        "sumo_config_train": "scene/osm.sumocfg",
        "sumo_config_eval": "scene/osm.sumocfg",
        "max_steps": 2000, # Pasos máximos por episodio
    },
    "training": {
        "num_episodes": 1000, # Para entrenamiento basado en episodios (DDQN, PPO, SAC)
        "total_timesteps_a3c": 500000, # Para entrenamiento basado en tiempo/pasos (A3C)
        "print_interval": 100, # Imprimir estadísticas cada N episodios
        "a3c_duration_sec": 3600 # Duración del entrenamiento A3C en segundos (ej: 1 hora)
    },
    "evaluation": {
        "num_episodes": 10,
    },
    # --- Configuraciones de algoritmos (ddqn, a3c, ppo, sac) ---
    "ddqn": {
        "gamma": 0.99,
        "epsilon": 1.0,
        "epsilon_min": 0.05,
        "epsilon_decay": 0.999,
        "lr": 0.0005,
        "target_update": 100, # Frecuencia de actualización de la red objetivo (pasos)
        "batch_size": 64,
        "memory_size": 50000
    },
    "a3c": {
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "lr": 0.0001,
        "value_loss_coef": 0.5,
        "entropy_coef": 0.01, # Coeficiente de bonificación de entropía
        "n_steps": 20, # Pasos por rollout antes de la actualización
        "n_workers": 4, # Número de workers paralelos
        "max_grad_norm": 40.0
    },
    "ppo": {
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "lr": 0.0001,
        "clip_epsilon": 0.2,
        "vf_coef": 0.5, # Coeficiente de pérdida de la función de valor
        "entropy_coef": 0.05, # Coeficiente de bonificación de entropía
        "n_steps": 64, # Pasos por rollout (recolectar datos durante estos pasos)
        "n_epochs": 3 , # Épocas de optimización por rollout
        "batch_size": 32, # Tamaño del mini-batch para las épocas de optimización
    },
    "sac": {
        "gamma": 0.99,
        "tau": 0.005, # Coeficiente de actualización suave
        "alpha": 0.2, # Temperatura de entropía (puede ser aprendida)
        "learn_alpha": True,
        "actor_lr": 0.0003,
        "critic_lr": 0.0003,
        "alpha_lr": 0.0003, # Si learn_alpha es True
        "target_update_interval": 1, # Con qué frecuencia ejecutar actualizaciones suaves (usualmente 1 para SAC)
        "batch_size": 256,
        "buffer_size": 1000000
    }
}