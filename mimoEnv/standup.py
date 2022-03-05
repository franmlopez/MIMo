import gym
import time
import mimoEnv
import argparse

def test(env, test_for=1000, model=None):
    env.seed(42)
    obs = env.reset()

    for _ in range(test_for):
        if model == None:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        env.render()
        if done:
            obs = env.reset()
    env.reset()
    env.close()

def main():

    env = gym.make('MIMoStandup-v0')
    obs = env.reset()

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_for', default=0, type=int,
                        help='Total timesteps of training')
    parser.add_argument('--test_for', default=1000, type=int,
                        help='Total timesteps of testing of trained policy')               
    parser.add_argument('--save_every', default=100000, type=int,
                        help='Number of timesteps between model saves')
    parser.add_argument('--new_model', default='PPO', type=str, 
                        choices=['PPO','SAC','TD3','DDPG','A2C','HER'],                   
                        help='RL algorithm from Stable Baselines3')
    parser.add_argument('--load_model', default=False, type=str,
                        help='Name of model to load')
    parser.add_argument('--save_model', default='', type=str,
                        help='Name of model to save')
    
    args = parser.parse_args()
    new_model = args.new_model
    load_model = args.load_model
    save_model = args.save_model
    save_every = args.save_every
    train_for = args.train_for
    test_for = args.test_for

    if new_model=='PPO':
        from stable_baselines3 import PPO as RL
    elif new_model=='SAC':
        from stable_baselines3 import SAC as RL
    elif new_model=='TD3':
        from stable_baselines3 import TD3 as RL
    elif new_model=='DDPG':
        from stable_baselines3 import DDPG as RL
    elif new_model=='A2C':
        from stable_baselines3 import A2C as RL

    # load pretrained model or create new one
    if load_model:
        model = RL.load("models/standup" + load_model, env, tensorboard_log="models/standup"+save_model+"/")
    else:
        model = RL("MultiInputPolicy", env, tensorboard_log="models/standup"+save_model+"/", verbose=1)

    # train model
    counter=0
    while train_for>0:
        counter+=1
        train_for_iter = min(train_for, save_every)
        train_for = train_for - train_for_iter
        model.learn(total_timesteps=train_for_iter)
        model.save("models/standup" + save_model + "_" + str(counter))
    
    test(env, model=model, test_for=test_for)


if __name__=='__main__':
    main()
