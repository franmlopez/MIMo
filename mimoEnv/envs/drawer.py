import os
import numpy as np
import copy
import mujoco_py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from mimoEnv.envs.mimo_env import MIMoEnv, DEFAULT_PROPRIOCEPTION_PARAMS

DRAWER_XML = os.path.abspath(os.path.join(__file__, "..", "..", "assets", "drawer_scene.xml"))


class MIMoDrawerEnv(MIMoEnv):

    def __init__(self,
                 model_path=DRAWER_XML,
                 initial_qpos={},
                 n_substeps=1,
                 proprio_params=DEFAULT_PROPRIOCEPTION_PARAMS,
                 touch_params=None,
                 vision_params=None,
                 vestibular_params=None,
                 goals_in_observation=False,
                 done_active=False,
                 prediction=False):

        self.steps = 0
        self.toy_init_y = -0.15
        self.agent = Agent()
        self.prediction = prediction

        super().__init__(model_path=model_path,
                         initial_qpos=initial_qpos,
                         n_substeps=n_substeps,
                         proprio_params=proprio_params,
                         touch_params=touch_params,
                         vision_params=vision_params,
                         vestibular_params=vestibular_params,
                         goals_in_observation=goals_in_observation,
                         done_active=done_active)

    def _is_done(self, achieved_goal=None, desired_goal=None, info=None):
        toy_y = self.sim.data.get_body_xpos('toy')[1]
        contact_threshold = 0.005
        done = (np.linalg.norm(toy_y - self.toy_init_y) > contact_threshold)
        return done
        
    def compute_reward(self, achieved_goal, desired_goal, info):
        """Dummy function"""
        return 0

    def _is_success(self, achieved_goal, desired_goal):
        """Indicates whether or not the achieved goal successfully achieved the desired goal."""
        return False

    def _is_failure(self, achieved_goal, desired_goal):
        """ Dummy function """
        return False

    def _sample_goal(self):
        """ Dummy function """
        return np.zeros(self._get_proprio_obs().shape)

    def _get_achieved_goal(self):
        """ Dummy function """
        return np.zeros(self._get_proprio_obs().shape)

    def _reset_sim(self):
        """Resets a simulation and indicates whether or not it was successful.
        If a reset was unsuccessful (e.g. if a randomized state caused an error in the
        simulation), this method should indicate such a failure by returning False.
        In such a case, this method will be called again to attempt a the reset again.
        """

        # train prediction network
        if self.prediction:
            for i in range(5):
                self.agent.train_predict_net(self.agent.predict_nets[i])

        # reset target in random initial position and velocities as zero
        qpos = np.array([-1.09241e-05, 4.03116e-06, 0.349443, 1, -2.16682e-06, 8.70054e-07, 5.45378e-06, -0.00492904, 0.288979, 0.00376704, 0.0026713, 0.287948, -0.00657526, -0.576921, 0.00130918, -9.38079e-05, -7.15269e-09, -4.2171e-09, -5.75793e-09, 7.15269e-09, -4.2171e-09, 5.75793e-09, 0.522459, 0.734782, -0.484279, -0.262299, -0.259645, 0.863403, 0.221394, 0.139992, -0.0377091, -0.142301, 0.00894348, -0.0741003, -0.35951, 0.00726376, -0.0193619, -0.374183, -0.107132, 1.94503e-05, -1.83078e-05, -0.196001, 0.0888279, -0.00018612, -2.22405e-05, -0.000995124, -0.107133, 3.80335e-05, -1.40993e-05, -0.195996, 0.0888214, -0.000213338, 1.77433e-06, -0.00099508, 2.4489e-05, 0.660031, -0.15, 0.484858, 1, -1.06491e-07, 1.37723e-05, 3.49732e-09, 0.451229, -0.15, 0.489973, 1, -5.02742e-16, 3.35704e-09, -1.50003e-07])
        qvel = np.zeros(self.sim.data.qvel.shape)

        new_state = mujoco_py.MjSimState(
            self.initial_state.time, qpos, qvel, self.initial_state.act, self.initial_state.udd_state
        )

        self.sim.set_state(new_state)
        self.sim.forward()
        self.toy_init_y = self.sim.data.get_body_xpos('toy')[1]
        return True

    def step(self, action):

        proprio_before = self._get_obs()['observation']
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()
        done = self._is_done()

        # Intrinsic reward
        if self.prediction:
            proprio_after = obs['observation']
            predictions = [self.agent.predict(self.agent.predict_nets[i], proprio_before, action, proprio_after) for i in range(5)]
            disagreement = np.var(predictions)
            intrinsic_reward = 1/10 * disagreement
        else:
            intrinsic_reward = 0

        # Extrinsic reward
        right_fingers_x = self.sim.data.get_body_xpos('right_fingers')[0]
        drawer_open_x = 0.05   
        drawer_open = (right_fingers_x < drawer_open_x)
        left_fingers_pos = self.sim.data.get_body_xpos('left_fingers')
        toy_pos = self.sim.data.get_body_xpos('toy')
        fingers_toy_dist = np.linalg.norm(left_fingers_pos - toy_pos)
        extrinsic_reward = -fingers_toy_dist + 1*drawer_open

        # Total reward
        reward =  1000*done + intrinsic_reward # + extrinsic_reward

        # Info
        info={'intrinsic_reward':intrinsic_reward, 'extrinsic_reward':extrinsic_reward}

        return obs, reward, done, info


##################
### PREDICTION ###
##################

class Buffer():

    def __init__(self, max_buffer_size=10000):
        self.buffer_counter = 0
        self.buffer_size = 0
        self.max_buffer_size = max_buffer_size
        self.proprio_before = np.array([])
        self.proprio_after = np.array([])
        self.action = np.array([])
        
    def reset(self):
        self.buffer_counter = 0
        self.buffer_size = 0
        self.proprio_before = np.array([])
        self.proprio_after = np.array([])
        self.action = np.array([])

    def store(self, proprio_before, proprio_after, action):
        '''
        Store experiences in buffer (up to max_buffer_size)
        '''
        if self.buffer_counter == 0:
            self.proprio_before = np.array([proprio_before.detach().numpy()])
            self.proprio_after = np.array([proprio_after.detach().numpy()])
            self.action = np.array([action.detach().numpy()])
        else: 
            self.proprio_before = np.concatenate((self.proprio_before, [proprio_before.detach().numpy()]), axis=0)
            self.proprio_after = np.concatenate((self.proprio_after, [proprio_after.detach().numpy()]), axis=0)
            self.action = np.concatenate((self.action, [action.detach().numpy()]), axis=0)
            self.proprio_before = self.proprio_before[-self.max_buffer_size:]
            self.proprio_after = self.proprio_after[-self.max_buffer_size:]
            self.action = self.action[-self.max_buffer_size:]
        self.buffer_counter += 1
        self.buffer_size = np.min([self.buffer_counter, self.max_buffer_size])

class Predict(nn.Module):
    def __init__(self, learning_rate=1e-3, n_proprio=141, n_actions=16):
        super().__init__()

        self.fc_1 = nn.Linear(n_proprio + n_actions, 512)
        nn.init.kaiming_uniform_(self.fc_1.weight.data)
        self.fc_2 = nn.Linear(512, 512)
        nn.init.kaiming_uniform_(self.fc_2.weight.data)
        self.fc_3 = nn.Linear(512, n_proprio)
        nn.init.kaiming_uniform_(self.fc_3.weight.data)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, proprio_before, action):
        proprio_action = torch.cat([proprio_before,action], dim=1)
        proprio_prediction = F.relu(self.fc_1(proprio_action))
        proprio_prediction = F.relu(self.fc_2(proprio_prediction))
        proprio_prediction = F.relu(self.fc_3(proprio_prediction))
        return proprio_prediction

class Agent():
    def __init__(self, batch_size=100):
        self.predict_nets = [Predict(),Predict(),Predict(),Predict(),Predict()]
        self.buffer = Buffer()
        self.batch_size = batch_size

    def predict(self, predict_net, proprio_before, action, proprio_after):
        # Compute prediction reward:
        proprio_before = torch.reshape(torch.from_numpy(proprio_before), (1,len(proprio_before))).float()
        proprio_after = torch.reshape(torch.from_numpy(proprio_after), (1,len(proprio_after))).float()
        action = torch.reshape(torch.from_numpy(action), (1,len(action))).float()
        proprio_prediction = predict_net(proprio_before, action)
        prediction_reward = F.mse_loss(proprio_prediction, proprio_after).item()
        # Store in buffer:
        proprio_before = proprio_before.view(-1)
        proprio_after = proprio_after.view(-1)
        action = action.view(-1)
        self.buffer.store(proprio_before, proprio_after, action)
        return prediction_reward
    
    def train_predict_net(self, predict_net):
        '''
        Train prediction network on a batch sampled from buffer.
        '''
        if self.buffer.buffer_size < self.batch_size:
            return None
           
        predict_net.train()
        running_loss = 0

        batch_idx = np.random.choice(self.buffer.buffer_size, self.batch_size, replace=False)
        proprio_before = torch.from_numpy(self.buffer.proprio_before[batch_idx]).float()
        proprio_after = torch.from_numpy(self.buffer.proprio_after[batch_idx]).float()
        action = torch.from_numpy(self.buffer.action[batch_idx]).float()

        predict_net.optimizer.zero_grad()
        proprio_prediction = predict_net(proprio_before, action)
        predict_loss = F.mse_loss(proprio_after, proprio_prediction)
        predict_loss.backward()
        predict_net.optimizer.step()

        running_loss += predict_loss.item()
    
        return running_loss
