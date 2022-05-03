import os
import numpy as np
import copy
import mujoco_py

from mimoEnv.envs.mimo_env import MIMoEnv, DEFAULT_PROPRIOCEPTION_PARAMS

EYEHAND_XML = os.path.abspath(os.path.join(__file__, "..", "..", "assets", "eyehand_scene.xml"))

VISION_PARAMS = {
    "eye_left": {"width": 256, "height": 256},
}

class MIMoEyeHandEnv(MIMoEnv):

    def __init__(self,
                 model_path=EYEHAND_XML,
                 initial_qpos={},
                 n_substeps=2,
                 proprio_params=DEFAULT_PROPRIOCEPTION_PARAMS,
                 touch_params=None,
                 vision_params=VISION_PARAMS,
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

    def compute_reward(self, achieved_goal, desired_goal, info):
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

        self.sim.set_state(self.initial_state)
        self.sim.forward()

        # perform 10 random actions
        for _ in range(10):
            action = self.action_space.sample()
            self._set_action(action)
            self.sim.step()
            self._step_callback()

        # reset target in random initial position and velocities as zero
        qpos = self.sim.data.qpos
        qpos[[-6, -5]] = np.array([
            self.initial_state.qpos[-6] + self.np_random.uniform(low=-0.3, high=0.3, size=1)[0],
            self.initial_state.qpos[-5] + self.np_random.uniform(low=-0.3, high=0.3, size=1)[0]
        ])
        qvel = np.zeros(self.sim.data.qvel.shape)

        new_state = mujoco_py.MjSimState(
            self.initial_state.time, qpos, qvel, self.initial_state.act, self.initial_state.udd_state
        )

        self.sim.set_state(new_state)
        self.sim.forward()
        self.target_init_pos = copy.deepcopy(self.sim.data.get_body_xpos('target'))
        return True

    def _step_callback(self):
        # manually set right eye to follow left eye
        self.sim.data.qpos[19] = -self.sim.data.qpos[16] #horizontal
        self.sim.data.qpos[20] = self.sim.data.qpos[17] # vertical
