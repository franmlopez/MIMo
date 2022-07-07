import gym
import time
import mimoEnv
import argparse
import numpy as np


def test(env, test_for=1000, model=None):
    env.seed(42)
    obs = env.reset()
    for idx in range(test_for):
        if model is None:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs)
        obs, rew, done, _ = env.step(action)
        env.render()
        if done:
            time.sleep(1)
            obs = env.reset()
    env.reset()


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
    parser.add_argument('--test_mu', default=0, type=int,
                        help='Mean of distribution of forces of drawer')
    parser.add_argument('--test_sigma', default=0, type=int,
                        help='Std of distribution of forces of drawer')
    parser.add_argument('--mu', default=0, type=int,
                        help='mu of model to load')
    parser.add_argument('--sigma', default=0, type=int,
                        help='sigma of model to load')                        
    parser.add_argument('--iter', default=0, type=int,
                        help='iter of model to load')                        
    
    args = parser.parse_args()
    algorithm = args.algorithm
    train_for = args.train_for
    test_for = args.test_for
    load_model = args.load_model
    save_model = args.save_model
    save_every = args.save_every
    save_every = train_for if save_every==None else save_every
    test_mu = args.test_mu
    test_sigma = args.test_sigma
    mu = args.mu
    sigma = args.sigma
    iter = args.iter
    
    env = gym.make('MIMoDrawer-v0', drawer_force_mu=test_mu, drawer_force_sigma=test_sigma)

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

    # load pretrained model or create new one
    if algorithm is None:
        model = None
    elif load_model:
        model = RL.load("models/drawer_mu" + str(mu) + '_sigma' + str(sigma) + '_iter' + str(iter), env)


    test(env, model=model, test_for=test_for)


if __name__ == '__main__':
    main()
