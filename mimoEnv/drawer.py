import gym
import time
import mimoEnv
import argparse
import numpy as np

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_for', default=0, type=int,
                        help='Total timesteps of training')            
    parser.add_argument('--save_every', default=None, type=int,
                        help='Number of timesteps between model saves')
    parser.add_argument('--algorithm', default=None, type=str, 
                        choices=['PPO', 'SAC', 'TD3', 'DDPG', 'A2C', 'HER'],
                        help='RL algorithm from Stable Baselines3')
    parser.add_argument('--load_model', default=False, type=str,
                        help='Name of model to load')
    parser.add_argument('--save_model', default='', type=str,
                        help='Name of model to save')
    parser.add_argument('--exp', default=0, type=int,
                        help='Id of experiment, determines values of mu and sigma for drawer force')
                        
    
    args = parser.parse_args()
    algorithm = args.algorithm
    train_for = args.train_for
    load_model = args.load_model
    save_model = args.save_model
    exp_id = args.exp
    
    if algorithm == 'PPO':
        from stable_baselines3 import PPO as RL
    elif algorithm == 'SAC':
        from stable_baselines3 import SAC as RL
    elif algorithm == 'TD3':
        from stable_baselines3 import TD3 as RL
    elif algorithm == 'DDPG':
        from stable_baselines3 import DDPG as RL
    elif algorithm == 'A2C':
        from stable_baselines3 import A2C as RL

    experiments = [
        {'name':'mu1_sigma0', 'mu':0, 'sigma': 0,},
        {'name':'mu12_sigma0', 'mu':12, 'sigma':0,},
        {'name':'mu6_sigma0', 'mu':6, 'sigma':0},
        {'name':'mu6_sigma3', 'mu':6, 'sigma':3},
    ]

    experiment = experiments[exp_id]
    name = experiment['name']
    mu = experiment['mu']
    sigma = experiment['sigma']
    
    for iter in range(10):

        env = gym.make('MIMoDrawer-v0', drawer_force_mu=mu, drawer_force_sigma=sigma)
        env.reset()

        model = RL("MultiInputPolicy",
                env,
                tensorboard_log="models/tensorboard_logs/drawer/" + name + "_iter"+str(iter),
                verbose=1)
        model.learn(total_timesteps=train_for)
        model.save("models/drawer_" + name + "_iter"+str(iter))
        del env, model
        time.sleep(60)
        

if __name__ == '__main__':
    main()
