import numpy as np
import torch
import gym
import argparse
import os
import random
import yaml
import json # in case the user want to modify the hyperparameters
import d4rl # used to make offline environments for source domains
import algo.utils as utils
from tqdm import tqdm
import h5py
from pathlib                              import Path
from algo.call_algo                       import call_algo
import wandb

def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],  
        tags=config["tag"],    
        name=config["name"],
        resume="allow",
        reinit=True,
    )
    wandb.run.save()

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
    parser.add_argument("--dir", default="./logs")
    parser.add_argument("--algo", default="DV_IGDF", help='policy to use')
    parser.add_argument("--env", default="halfcheetah-kinematic") # support 
    parser.add_argument('--srctype', default="medium", help='dataset type used in the source domain') # only useful when source domain is offline
    parser.add_argument("--seed", default=100, type=int)
    parser.add_argument("--save-model", default=True, type=bool)        # Save model and optimizer parameters
    parser.add_argument('--tar_env_interact_interval', help='interval of interacting with target env', default=10, type=int)
    parser.add_argument('--max_step', default=int(1e6), type=int)  # the maximum gradient step for off-dynamics rl learning
    parser.add_argument('--limited_size', default=False, type=bool)
    parser.add_argument('--params', default=None, help='Hyperparameters for the adopted algorithm, ought to be in JSON format')
    parser.add_argument('--device', default="cuda:0", type=str)
    parser.add_argument('--tradeoff', default=0.8, type=float)
    parser.add_argument("--xi", default=0.75, type=float)
    parser.add_argument("--target_ratio", default=0.1, type=float)
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    project = "DVDF"
    group = f"xi_{args.xi}"
    name = f"{args.env}-{args.srctype}-{args.seed}"
    tag = f"{args.env}-{args.srctype}"
    
    api = wandb.Api()
    filters = {"display_name": name,
               "group": group,}
    matching_runs = list(api.runs(project, filters=filters))
    if matching_runs:
        print(f"found {len(matching_runs)} matching runs, deleting...")
        for run in matching_runs:
            run.delete()
    else:
        print("no need to delete")
    
    wandb_init(
        config={
            "project": project,
            "group": group,
            "tag": tag,
            "name": name,
        }
    )
    
    task = args.env.split('-')[0]
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    
    #load src dataset
    src_dataset_path = f"{str(Path(__file__).parent.absolute())}/dataset/source/{args.env}-{args.srctype}.hdf5"
    data_dict = {}
    with h5py.File(src_dataset_path, 'r') as dataset_file:
        for k in tqdm(get_keys(dataset_file), desc="load datafile"):
            try:  # first try loading as an array
                data_dict[k] = dataset_file[k][:]
            except ValueError as e:  # try loading as a scalar
                data_dict[k] = dataset_file[k][()]
    src_dataset = data_dict
    
    #load tar dataset
    tar_env = gym.make(task + "-" + args.srctype + "-v2")
    tar_dataset = d4rl.qlearning_dataset(tar_env)
    
    if args.limited_size:
        size = 5000
    else:
        size = int(tar_dataset["observations"].shape[0] * args.target_ratio)

    ind = np.random.randint(0, tar_dataset["observations"].shape[0], size=size)
    
    tar_dataset = {
        "observations": tar_dataset['observations'][ind],
        "actions": tar_dataset['actions'][ind],
        "next_observations": tar_dataset['next_observations'][ind],
        "rewards": tar_dataset['rewards'][ind],
        "terminals": tar_dataset['terminals'][ind],
    }
    
    # load pre-defined hyperparameter config for training
    with open(f"{str(Path(__file__).parent.absolute())}/config/mujoco/{args.algo.lower()}/{task}.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    if args.params is not None:
        override_params = json.loads(args.params)
        config.update(override_params)
        print('The following parameters are updated to:', args.params)
        
    config["tradeoff"] = args.tradeoff
    config["src_Q_path"] = f"{str(Path(__file__).parent.absolute())}/logs/Offline/{args.env}/{args.srctype}/{args.seed}/models/model_critic"
    config["src_V_path"] = f"{str(Path(__file__).parent.absolute())}/logs/Offline/{args.env}/{args.srctype}/{args.seed}/models/model_value"
    #self.q_funcs.load_state_dict(torch.load(filename + "_critic"))
    

    print("------------------------------------------------------------")
    print("Policy: {}, Env: {}, Seed: {}".format(args.algo, args.env + "-" + args.srctype, args.seed))
    print("------------------------------------------------------------")   
    
    outdir = args.dir + '/' + args.env + '/' + args.srctype  + '/' + str(args.seed)
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if args.save_model and not os.path.exists("{}/models".format(outdir)):
        os.makedirs("{}/models".format(outdir))

    # seed all
    tar_env.action_space.seed(args.seed) if tar_env is not None else None


    # get necessary information from both domains
    state_dim = tar_env.observation_space.shape[0]
    action_dim = tar_env.action_space.shape[0] 
    max_action = float(tar_env.action_space.high[0])
    min_action = -max_action

    config.update({
        'env_name': args.env + args.srctype,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'max_action': max_action,
        'tar_env_interact_interval': int(args.tar_env_interact_interval),
        'max_step': int(args.max_step),
        "xi": args.xi,
    })

    policy = call_algo(args.algo, config, 3, device)
    
    with open(outdir + '/log.txt','w') as f:
        f.write('\n Policy: {}; Env: {}, seed: {}'.format(args.algo, args.env, args.seed))
        for item in config.items():
            f.write('\n {}'.format(item))

    src_replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)
    tar_replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)

    src_replay_buffer.convert_D4RL(src_dataset)
    tar_replay_buffer.convert_D4RL(tar_dataset)

    eval_cnt = 0
    
    eval_tar_return = eval_policy(policy, tar_env, eval_cnt=eval_cnt)
    eval_cnt += 1

    for t in range(int(config['max_step'])):
        policy.train(src_replay_buffer, tar_replay_buffer, config['batch_size'], writer=None)

        if (t + 1) % config['eval_freq'] == 0:
            tar_eval_return = eval_policy(policy, tar_env, eval_cnt=eval_cnt)
            eval_normalized_score = tar_env.get_normalized_score(tar_eval_return)
            print(f"Step: {t}  Score: {eval_normalized_score}")
            
            wandb.log({"Score": eval_normalized_score}, step=t)
            
            with open(outdir + '/return.txt', 'a') as f:
                f.write(f"{t}  {eval_normalized_score} \n")
            eval_cnt += 1

            if args.save_model:
                policy.save('{}/models/model'.format(outdir))
                
    wandb.finish()