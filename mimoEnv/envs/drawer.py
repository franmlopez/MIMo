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
                 drawer_force_mu=0,
                 drawer_force_sigma=0,
                 ):

        self.steps = 0
        self.drawer_force_mu = drawer_force_mu
        self.drawer_force_sigma = drawer_force_sigma

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
        #toy_y = self.sim.data.get_body_xpos('toy')[1]
        #contact_threshold = 0.005
        #done = (np.linalg.norm(toy_y - self.toy_init_y) > contact_threshold)
        return False
        
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

        # reset target in random initial position and velocities as zero
        qpos = np.array([0.000709667, -4.6158e-06, 0.350098, 0.998933, -7.16743e-06, 0.0461819, 0.000250664, -0.00509185, 0.103724, -0.0579377, -0.0161323, 0.101646, -0.0659375, -0.206492, 0.0315221, 0.00158587, -1.23292e-08, -3.94402e-09, -2.35528e-09, 1.23292e-08, -3.94402e-09, 2.35528e-09, 0.62636, 0.524773, -0.498628, -0.607125, 0.0848819, 0.841533, 0.0559336, 0.139842, -0.0661014, -0.143389, 0.00439174, -0.0522003, -0.361253, 0.00350864, -0.0109386, -0.36993, -0.0923748, 3.72571e-05, 0.000489106, -0.00180591, 0.00180126, 1.87318e-05, -5.91088e-05, -0.000916676, -0.0924224, 4.98519e-05, -0.000507101, -0.00158078, 0.00152869, 2.29407e-05, -5.21039e-05, -0.000916652, -3.8779e-05])
        qvel = np.zeros(self.sim.data.qvel.shape)

        new_state = mujoco_py.MjSimState(
            self.initial_state.time, qpos, qvel, self.initial_state.act, self.initial_state.udd_state
        )

        self.sim.set_state(new_state)
        self.sim.forward()

        # Add randomly sampled drawer force
        force = np.random.normal(loc=self.drawer_force_mu, scale=self.drawer_force_sigma)
        self.drawer_force = max(force, 0)
        self.sim.data.xfrc_applied[25,:] = np.array([self.drawer_force, 0, 0, 0, 0, 0])
        return True

    def step(self, action):

        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()
        
        # Reward: drawer opening - metabolic cost
        drawer_pos = self.sim.data.get_body_xpos('handle')[0]
        drawer_opening = 0.2 - drawer_pos
        cost = 0.005 * np.square(self.sim.data.ctrl).sum()
        reward =  drawer_opening - cost

        # Info
        done = False
        info={
            'force' : self.drawer_force,
            'drawer_opening' : drawer_opening,
        }

        return obs, reward, done, info