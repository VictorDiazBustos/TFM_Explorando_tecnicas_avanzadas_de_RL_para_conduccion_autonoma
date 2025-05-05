#!/bin/bash

start_total=$(date +%s)

start_ddqn=$(date +%s)
./train_ddqn.sh
end_ddqn=$(date +%s)

start_ppo=$(date +%s)
./train_ppo.sh
end_ppo=$(date +%s)

start_sac=$(date +%s)
./train_sac.sh
end_sac=$(date +%s)

end_total=$(date +%s)

echo "Tiempo de train_ddqn.sh: $((end_ddqn - start_ddqn)) segundos"
echo "Tiempo de train_ppo.sh: $((end_ppo - start_ppo)) segundos"
echo "Tiempo de train_sac.sh: $((end_sac - start_sac)) segundos"
echo "Tiempo total: $((end_total - start_total)) segundos"