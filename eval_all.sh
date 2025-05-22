#!/bin/bash
./eval_ddqn.sh
./eval_ppo.sh
./eval_sac.sh
python plot_results_comparison.py
