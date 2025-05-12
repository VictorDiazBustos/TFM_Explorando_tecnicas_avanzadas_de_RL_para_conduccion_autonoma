#!/bin/bash
echo "Ejecutando evaluación DDQN (modelo final, 20 episodios, con GUI)..."
python main.py --alg ddqn --mode eval --episodes 20 --gui --save_dir models --gen_eval_routes
echo "Evaluación DDQN finalizada."