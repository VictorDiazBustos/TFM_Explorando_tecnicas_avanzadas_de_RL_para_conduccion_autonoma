#!/bin/bash
echo "Ejecutando evaluación A3C (modelo final, 20 episodios, con GUI)..."
python main.py --alg a3c --mode eval --episodes 20 --gui --save_dir models
echo "Evaluación A3C finalizada."