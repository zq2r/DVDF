#!/bin/bash

newTmuxSession(){ #new tmux session
    session=$1
    tmux has-session -t $session 2>/dev/null
    if [ $? == 0 ]; then
        echo "Session $session already exists"
        tmux kill-session -t $session
        tmux new-session -d -s $session
        echo "Session $session created done"
    else
        tmux new-session -d -s $session
        echo "Session $session created done"
    fi
}

envs=("halfcheetah-medium-v2")

srctypes=("random" "medium" "expert")

seeds=("100")
device="cuda:0"
algo="SQL"
save_model="True"

for env in "${envs[@]}"
do  
    for srctype in "${srctypes[@]}"
    do
        for seed in "${seeds[@]}"
        do  
            tmux_name="offline_${algo}_${env}_${srctype}_${seed}"
            newTmuxSession ${tmux_name}
            tmux send -t ${tmux_name} "cd path/to/project" C-m
            tmux send -t ${tmux_name} "conda activate dvdf" C-m
            tmux send -t ${tmux_name} "python train_offline.py --policy=${algo} --env=${env} --srctype=${srctype} --seed=${seed} --device=${device} --save_model=${save_model}" C-m
        done
    done
done
