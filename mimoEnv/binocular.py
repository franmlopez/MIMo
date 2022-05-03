import gym
import time
import mimoEnv
import argparse
import numpy as np
import matplotlib.pyplot as plt


def test(env, test_for=1000, episode_length=10, model=None):
    env.seed(42)
    obs = env.reset()
    
    plt.ion()
    fig, ax = plt.subplots()
    grayscale_weights = np.array([0.299, 0.587, 0.114])
    img_left = np.dot(obs['eye_left'], grayscale_weights)
    img_right = np.dot(obs['eye_right'], grayscale_weights)
    img_left_3d = np.reshape(img_left, img_left.shape + (1,))
    img_right_3d = np.reshape(img_right, img_right.shape + (1,))
    img_center_3d = (img_left_3d + img_right_3d) / 2.0
    img_stereo = np.concatenate((img_left_3d,img_center_3d,img_right_3d), axis=2)
    img_stereo = img_stereo.astype(int)
    axim = ax.imshow(img_stereo)
    time.sleep(0.5)

    for idx in range(test_for):
        if model is None:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs)
        obs, _, done, _ = env.step(action)
        #env.render()
        
        img_left = np.dot(obs['eye_left'], grayscale_weights)
        img_right = np.dot(obs['eye_right'], grayscale_weights)
        img_left_3d = np.reshape(img_left, img_left.shape + (1,))
        img_right_3d = np.reshape(img_right, img_right.shape + (1,))
        img_center_3d = (img_left_3d + img_right_3d) / 2.0
        img_stereo = np.concatenate((img_left_3d,img_center_3d,img_right_3d), axis=2)
        img_stereo = img_stereo.astype(int)
        axim.set_data(img_stereo)
        fig.canvas.flush_events()
        time.sleep(0.2)

        if done or (idx-1)%episode_length==0:
            time.sleep(1)
            obs = env.reset()
            
    env.reset()


def main():

    env = gym.make('MIMoBinocular-v0')
    obs = env.reset()
    print(env.sim.data.qpos)

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_for', default=0, type=int,
                        help='Total timesteps of training')
    parser.add_argument('--test_for', default=1000, type=int,
                        help='Total timesteps of testing of trained policy')               
    parser.add_argument('--save_every', default=100000, type=int,
                        help='Number of timesteps between model saves')
    parser.add_argument('--algorithm', default=None, type=str, 
                        choices=['PPO', 'SAC', 'TD3', 'DDPG', 'A2C', 'HER'],
                        help='RL algorithm from Stable Baselines3')
    parser.add_argument('--load_model', default=False, type=str,
                        help='Name of model to load')
    parser.add_argument('--save_model', default='', type=str,
                        help='Name of model to save')
    
    args = parser.parse_args()
    algorithm = args.algorithm
    load_model = args.load_model
    save_model = args.save_model
    save_every = args.save_every
    train_for = args.train_for
    test_for = args.test_for

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
        model = RL.load("models/binocular" + load_model, env)
    else:
        model = RL("MultiInputPolicy", env, tensorboard_log="models/tensorboard_logs/", verbose=1)

    # train model
    counter = 0
    while train_for > 0:
        counter += 1
        train_for_iter = min(train_for, save_every)
        train_for = train_for - train_for_iter
        model.learn(total_timesteps=train_for_iter)
        model.save("models/reach" + save_model + "_" + str(counter))
    
    test(env, model=model, test_for=test_for)


if __name__ == '__main__':
    main()
