#!/bin/bash
export SUMO_HOME="usr/share/sumo/"
echo "Ejecutando entrenamiento DDQN (5000 episodios)..."
python main.py --alg ddqn --mode train --episodes 5000 --save_dir models
echo "Entrenamiento DDQN finalizado."