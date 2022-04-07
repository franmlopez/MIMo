import os
import numpy as np
import copy
import mujoco_py

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
                 done_active=False):

        self.steps = 0

        super().__init__(model_path=model_path,
                         initial_qpos=initial_qpos,
                         n_substeps=n_substeps,
                         proprio_params=proprio_params,
                         touch_params=touch_params,
                         vision_params=vision_params,
                         vestibular_params=vestibular_params,
                         goals_in_observation=goals_in_observation,
                         done_active=done_active)

    def _is_done(self, achieved_goal, desired_goal, info):
        left_fingers_pos = self.sim.data.get_body_xpos('left_fingers')
        toy_pos = self.sim.data.get_body_xpos('toy')
        contact_with_toy_threshold = 0.075
        contact_with_toy = (np.linalg.norm(left_fingers_pos - toy_pos) < contact_with_toy_threshold)
        return contact_with_toy
        
    def compute_reward(self, achieved_goal, desired_goal, info):
        """Reward computed based on drawer being opened and contact with toy"""
        right_fingers_x = self.sim.data.get_body_xpos('right_fingers')[0]
        drawer_open_x = 0.05
        drawer_open = (right_fingers_x < drawer_open_x)
        left_fingers_pos = self.sim.data.get_body_xpos('left_fingers')
        toy_pos = self.sim.data.get_body_xpos('toy')
        fingers_toy_dist = np.linalg.norm(left_fingers_pos - toy_pos)
        contact_with_toy_threshold = 0.075
        contact_with_toy = (fingers_toy_dist < contact_with_toy_threshold)
        reward = -fingers_toy_dist + 1*drawer_open + 1000*contact_with_toy
        return reward

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
        qpos = np.array([-1.09241e-05, 4.03116e-06, 0.349443, 1, -2.16682e-06, 8.70054e-07, 5.45378e-06, -0.00492904, 0.288979, 0.00376704, 0.0026713, 0.287948, -0.00657526, -0.576921, 0.00130918, -9.38079e-05, -7.15269e-09, -4.2171e-09, -5.75793e-09, 7.15269e-09, -4.2171e-09, 5.75793e-09, 0.522459, 0.734782, -0.484279, -0.262299, -0.259645, 0.863403, 0.221394, 0.139992, -0.0377091, -0.142301, 0.00894348, -0.0741003, -0.35951, 0.00726376, -0.0193619, -0.374183, -0.107132, 1.94503e-05, -1.83078e-05, -0.196001, 0.0888279, -0.00018612, -2.22405e-05, -0.000995124, -0.107133, 3.80335e-05, -1.40993e-05, -0.195996, 0.0888214, -0.000213338, 1.77433e-06, -0.00099508, 2.4489e-05, 0.660031, -0.15, 0.484858, 1, -1.06491e-07, 1.37723e-05, 3.49732e-09, 0.451229, -0.15, 0.489973, 1, -5.02742e-16, 3.35704e-09, -1.50003e-07])
        qvel = np.zeros(self.sim.data.qvel.shape)

        new_state = mujoco_py.MjSimState(
            self.initial_state.time, qpos, qvel, self.initial_state.act, self.initial_state.udd_state
        )

        self.sim.set_state(new_state)
        self.sim.forward()
        return True