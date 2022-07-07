import gym
import time
import mimoEnv
import argparse
import numpy as np
import pandas as pd

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_for', default=0, type=int,
                        help='Total timesteps of training')
    parser.add_argument('--test_for', default=1000, type=int,
                        help='Total timesteps of testing of trained policy')               
    parser.add_argument('--save_every', default=None, type=int,
                        help='Number of timesteps between model saves')
    parser.add_argument('--algorithm', default=None, type=str, 
                        choices=['PPO', 'SAC', 'TD3', 'DDPG', 'A2C', 'HER'],
                        help='RL algorithm from Stable Baselines3')
    parser.add_argument('--load_model', default=False, type=str,
                        help='Name of model to load')
    parser.add_argument('--save_model', default='', type=str,
                        help='Name of model to save')
    parser.add_argument('--test_mu', default=None, type=int,
                        help='Mean of distribution of forces of drawer')
    parser.add_argument('--exp', default=0, type=int,
                        help='Id of experiment, determines values of mu and sigma for drawer force')
    parser.add_argument('--iter', default=0, type=int,
                        help='Iteration of experiment')

    
    args = parser.parse_args()
    algorithm = args.algorithm
    train_for = args.train_for
    test_for = args.test_for
    load_model = args.load_model
    save_model = args.save_model
    exp_id = args.exp
    iter = args.iter
    test_mu = args.test_mu

    experiments = [
        {'name':'mu1_sigma0', 'mu':1, 'sigma': 0,},
        {'name':'mu12_sigma0', 'mu':12, 'sigma':0,},
        {'name':'mu6_sigma0', 'mu':6, 'sigma':0},
        {'name':'mu6_sigma6', 'mu':6, 'sigma':6},
    ]

    experiment = experiments[exp_id]
    name = experiment['name']
    mu = experiment['mu']
    sigma = experiment['sigma']
    if test_mu is None:
        test_mu = mu
    
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

    env = gym.make('MIMoDrawer-v0', drawer_force_mu=test_mu, drawer_force_sigma=0)
    model = RL.load("models/drawer_" + name + "_iter"+str(iter), env)
    obs = env.reset()
    n_drawer_open = 0
    timestep = 0

    df_drawer = pd.DataFrame(columns=["timestep","drawer_pos","n_drawer_open"])
    df_proprio = pd.DataFrame()

    for idx in range(test_for):
        timestep += 1
        action, _ = model.predict(obs)
        obs, rew, done, info = env.step(action)
        
        drawer_force = info['drawer_force']
        drawer_pos = info['drawer_pos']
        n_drawer_open += (1 if drawer_pos<0.2 else 0)
        df_drawer = df_drawer.append({
            'timestep': timestep,
            'drawer_pos': drawer_pos,
            'n_drawer_open': n_drawer_open,
            'drawer_force': drawer_force,
            }, ignore_index=True)

        proprio = obs['observation']
        df_proprio = df_proprio.append(pd.Series(np.append(np.append(timestep,drawer_force),proprio)), ignore_index=True)

        if done:
            obs = env.reset()
            timestep = 0

    print('mu:',mu,'\tsigma:',sigma,'\titer:',iter,'\t','score:',n_drawer_open/test_for)
    df_drawer.to_csv('results/drawer_testmu0'+str(test_mu)+'_mu'+str(mu)+'_sigma'+str(sigma)+'_iter'+str(iter)+'.csv')
    df_proprio.to_csv('results/proprio_testmu0'+str(test_mu)+'_mu'+str(mu)+'_sigma'+str(sigma)+'_iter'+str(iter)+'.csv')

if __name__ == '__main__':
    main()
