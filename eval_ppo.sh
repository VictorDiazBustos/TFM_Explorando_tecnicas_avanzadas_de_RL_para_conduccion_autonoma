#!/bin/bash
echo "Ejecutando evaluación PPO (modelo final, 20 episodios, con GUI)..."
python main.py --alg ppo --mode eval --episodes 20 --gui --save_dir models
echo "Evaluación PPO finalizada."