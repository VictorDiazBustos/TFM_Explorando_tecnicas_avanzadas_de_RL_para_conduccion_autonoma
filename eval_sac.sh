#!/bin/bash
echo "Ejecutando evaluación SAC (modelo final, 20 episodios, con GUI)..."
python main.py --alg sac --mode eval --episodes 20 --gui --save_dir models
echo "Evaluación SAC finalizada."