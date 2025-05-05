#!/bin/bash
echo "Ejecutando entrenamiento PPO (5000 episodios)..."
python main.py --alg ppo --mode train --episodes 5000 --save_dir models
echo "Entrenamiento PPO finalizado."