import numpy as np
import torch
import gym
import argparse
import os
import random
import math
import time
import copy
import yaml
import json # in case the user want to modify the hyperparameters
import d4rl # used to make offline environments for source domains
import d4rl
import algo.utils as utils
import h5py
from tqdm import tqdm
from pathlib                              import Path
from algo.call_algo                       import call_algo
from dataset.call_dataset                 import call_tar_dataset
from envs.mujoco.call_mujoco_env          import call_mujoco_env
from envs.infos                           import get_normalized_score

from gym.envs.mujoco.half_cheetah_v3    import  HalfCheetahEnv
from gym.envs.mujoco.ant_v3             import  AntEnv
from gym.envs.mujoco.walker2d_v3        import  Walker2dEnv
from gym.envs.mujoco.hopper_v3          import  HopperEnv

from gym.wrappers.time_limit            import  TimeLimit


def eval_policy(policy, env, eval_episodes=10, eval_cnt=None):
    eval_env = env

    avg_reward = 0.
    for episode_idx in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            next_state, reward, done, _ = eval_env.step(action)

            avg_reward += reward
            state = next_state
    avg_reward /= eval_episodes

    print("[{}] Evaluation over {} episodes: {}".format(eval_cnt, eval_episodes, avg_reward))

    return avg_reward


def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="./logs/Offline")
    parser.add_argument("--policy", default="IQL", help='policy to use') # support IQL, SQL
    parser.add_argument("--env", default="ant-kinematic")
    parser.add_argument('--srctype', default="random", help='dataset type used in the source domain (and the target domain)') # only useful when source domain is offline
    parser.add_argument('--mode', default=3, type=int, help='the training mode, there are four types, 0: online-online, 1: offline-online, 2: online-offline, 3: offline-offline')
    parser.add_argument("--seed", default=100, type=int)
    parser.add_argument("--save_model", default=True, type=bool)        # Save model and optimizer parameters
    parser.add_argument('--tar_env_interact_interval', help='interval of interacting with target env', default=10, type=int)
    parser.add_argument('--max_step', default=int(1e6), type=int)  # the maximum gradient step for off-dynamics rl learning
    parser.add_argument('--params', default=None, help='Hyperparameters for the adopted algorithm, ought to be in JSON format')
    parser.add_argument('--device', default='cuda:0', type=str)
    args = parser.parse_args()  
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    if '_' in args.env:
        args.env = args.env.replace('_', '-')
    src_env_name = args.env.split('-')[0]
    
    if "halfcheetah" in args.env:
        src_env = HalfCheetahEnv
    elif "hopper" in args.env:
        src_env = HopperEnv
    elif "walker2d" in args.env:
        src_env = Walker2dEnv
    elif "ant" in args.env:
        src_env = AntEnv
    else:
        raise NotImplementedError
    
    #load env and dataset
    if args.env in ["halfcheetah", "hopper", "walker2d", "ant"]:
        src_eval_env = gym.make(args.env + "-" + args.srctype + "-v2")
        src_eval_env.seed(args.seed)
        src_dataset = d4rl.qlearning_dataset(src_eval_env)
        
        size = int(src_dataset["observations"].shape[0] * 0.1)

        ind = np.random.randint(0, src_dataset["observations"].shape[0], size=size)
    
        src_dataset = {
            "observations": src_dataset['observations'][ind],
            "actions": src_dataset['actions'][ind],
            "next_observations": src_dataset['next_observations'][ind],
            "rewards": src_dataset['rewards'][ind],
            "terminals": src_dataset['terminals'][ind],
        }
        
    else:
        src_eval_env = TimeLimit(
                    src_env(xml_file=f"{str(Path(__file__).parent.absolute())}/envs/mujoco/assets/{args.env.replace('-', '_')}.xml",),
                    max_episode_steps=1000          
                )
        src_eval_env.seed(args.seed)
        
        src_dataset_path = f"{str(Path(__file__).parent.absolute())}/dataset/source/{args.env}-{args.srctype}.hdf5"
        data_dict = {}
        with h5py.File(src_dataset_path, 'r') as dataset_file:
            for k in tqdm(get_keys(dataset_file), desc="load datafile"):
                try:  # first try loading as an array
                    data_dict[k] = dataset_file[k][:]
                except ValueError as e:  # try loading as a scalar
                    data_dict[k] = dataset_file[k][()]
        src_dataset = data_dict
    
    
    policy_config_name = 'sql' # or iql

    # load pre-defined hyperparameter config for training
    with open(f"{str(Path(__file__).parent.absolute())}/config/mujoco/{policy_config_name}/{src_env_name}.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if args.params is not None:
        override_params = json.loads(args.params)
        config.update(override_params)
        print('The following parameters are updated to:', args.params)

    
    print("------------------------------------------------------------")
    print("Policy: {}, Env: {}, Seed: {}".format(args.policy, args.env, args.seed))
    print("------------------------------------------------------------")
    
    #outdir = args.dir + '/' + args.policy + '/' + args.env + '-' + args.srctype + '-' + str(args.seed)
    outdir = args.dir + '/' + args.env + '/' + args.srctype + '/' + str(args.seed)
    
    if args.save_model and not os.path.exists("{}/models".format(outdir)):
        os.makedirs("{}/models".format(outdir))

    # seed all
    src_eval_env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    # get necessary information from both domains
    state_dim = src_eval_env.observation_space.shape[0]
    action_dim = src_eval_env.action_space.shape[0] 
    max_action = float(src_eval_env.action_space.high[0])
    min_action = -max_action
    

    config.update({
        'env_name': args.env,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'max_action': max_action,
        'tar_env_interact_interval': int(args.tar_env_interact_interval),
        'max_step': int(args.max_step),
    })

    from algo.offline.iql import IQL
    from algo.offline.sql import SQL
    
    algo = IQL # or SQL
    policy = algo(config, device)
    
    ## write logs to record training parameters
    with open(outdir + '/log.txt','w') as f:
        f.write('\n Policy: {}; Dataset: {}, seed: {}'.format(args.policy, args.env + '-' + args.srctype, args.seed))
        for item in config.items():
            f.write('\n {}'.format(item))


    src_replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)
    src_replay_buffer.convert_D4RL(src_dataset)

    eval_cnt = 0
    
    eval_src_return = eval_policy(policy, src_eval_env, eval_cnt=eval_cnt)
    eval_cnt += 1

    # offline training
    for t in range(int(config['max_step'])):
        policy.train(src_replay_buffer, config['batch_size'], writer=None)

        if (t + 1) % config['eval_freq'] == 0:
            
            if args.env in ["halfcheetah", "hopper", "walker2d", "ant"]:
                src_eval_return = eval_policy(policy, src_eval_env, eval_cnt=eval_cnt)
                src_eval_return = src_eval_env.get_normalized_score(src_eval_return)
            else:
                src_eval_return = eval_policy(policy, src_eval_env, eval_cnt=eval_cnt)
            print(f"Step: {t}  Return: {src_eval_return}")
            
            with open(outdir + '/return.txt', 'a') as f:
                f.write(f"{t}  {src_eval_return} \n")
            eval_cnt += 1
            if args.save_model:
                policy.save('{}/models/model'.format(outdir))
                