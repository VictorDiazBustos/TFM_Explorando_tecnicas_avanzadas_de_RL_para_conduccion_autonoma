#!/bin/bash
echo "Ejecutando entrenamiento SAC (1000 episodios)..."
python main.py --alg sac --mode train --episodes 1000 --save_dir models
echo "Entrenamiento SAC finalizado."