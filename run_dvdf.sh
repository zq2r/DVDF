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


envs=("ant-kinematic")
srctypes=("expert")

seeds=("100" "200" "300")
device="cuda:1"
algo="DV_IGDF"
save_model="False"

tradeoff="0.7"
xi="0.5"
target_ratio="0.1"

dir="./logs/DVDF"

for env in "${envs[@]}"
do  
    for srctype in "${srctypes[@]}"
    do
        for seed in "${seeds[@]}"
        do  
            tmux_name="${algo}_${env}_${srctype}_${seed}"
            newTmuxSession ${tmux_name}
            tmux send -t ${tmux_name} "cd path/to/project" C-m
            tmux send -t ${tmux_name} "conda activate odrl" C-m
            tmux send -t ${tmux_name} "python train_dvdf.py --algo=${algo} --env=${env} --srctype=${srctype} --seed=${seed} --device=${device} --save-model=${save_model} --tradeoff=${tradeoff} --xi=${xi} --target_ratio=${target_ratio} --dir=${dir}" C-m
        done
    done
done
