#!/bin/bash
echo "Ejecutando entrenamiento SAC (3000 episodios)..."
python main.py --alg sac --mode train --episodes 3000 --save_dir models
echo "Entrenamiento SAC finalizado."