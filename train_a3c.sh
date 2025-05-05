#!/bin/bash
echo "Ejecutando entrenamiento A3C (por duración definida en main.py)..."
python main.py --alg a3c --mode train --save_dir models
echo "Entrenamiento A3C finalizado (o interrumpido)."