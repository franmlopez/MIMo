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

        # reset target in random initial position and velocities as zero
        qpos = np.array([0.000500764, -0.000544089, 0.350125, 0.922961, 0.0183931, 0.0439376, -0.381935, -0.00378106, 0.146461, -0.0586624, -0.0284711, 0.136437, -0.0688624, -0.284341, 0.033117, 0.00373593, -4.61789e-08, -1.31973e-08, -4.50582e-09, 4.61789e-08, -1.31973e-08, 4.50582e-09, 0.572368, 1.14456, -0.436011, -0.852487, -0.283398, 0.122159, 0.0600522, -2.65132, -0.0662976, -0.143193, 0.00132513, -0.0578722, -0.362256, 0.00139129, -0.0121369, -0.366302, -0.0952619, -0.000319591, 0.00064712, -0.00186952, 0.00188041, -1.4498e-06, -6.1099e-05, -0.000716553, -0.0953267, 0.000402864, -0.000667143, -0.00161504, 0.00156153, 4.1315e-05, -5.31858e-05, -0.000716543, -2.55131e-07])
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
        drawer_pos = self.sim.data.get_body_xpos('drawer')[0]
        drawer_opening = 0.2 - drawer_pos
        cost = 0.001 * np.square(self.sim.data.ctrl).sum()
        reward =  drawer_opening # - cost

        # Info
        done = False
        info={
            'force' : self.drawer_force,
            'drawer_opening' : drawer_opening,
        }

        return obs, reward, done, info