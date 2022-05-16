import os
import numpy as np
import copy
import mujoco_py

from mimoEnv.envs.mimo_env import MIMoEnv, DEFAULT_PROPRIOCEPTION_PARAMS
import mimoEnv.utils as mimo_utils

BINOCULAR_XML = os.path.abspath(os.path.join(__file__, "..", "..", "assets", "binocular_scene.xml"))

VISION_PARAMS = {
    "eye_left": {"width": 64, "height": 64},
    "eye_right": {"width": 64, "height": 64},
}

PROPRIOCEPTION_PARAMS = {
    #"components": ["velocity", "torque", "limits"],
    #"threshold": .035,
}

TEXTURES = ["texture"+str(idx) for idx in range(1,51)]


class MIMoBinocularEnv(MIMoEnv):

    def __init__(self,
                 model_path=BINOCULAR_XML,
                 initial_qpos={},
                 n_substeps=1,
                 proprio_params=PROPRIOCEPTION_PARAMS,
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

        # Preload target textures:
        self.target_textures = {}
        for texture in TEXTURES:
            tex_id = mimo_utils.texture_name2id(self.sim.model, texture)
            self.target_textures[texture] = tex_id
        target_material_name = "target-texture"
        self._target_material_id = mimo_utils.material_name2id(self.sim.model, target_material_name)
        
    def return_obs(self):
        obs = self._get_obs()
        return obs

    def compute_reward(self, achieved_goal, desired_goal, info):
        #obs = self._get_obs()
        #grayscale_weights = np.array([0.299, 0.587, 0.114])
        #img_left = np.dot(obs['eye_left'], grayscale_weights)
        #img_right = np.dot(obs['eye_right'], grayscale_weights)
        #reward = np.sum(img_left==img_right)
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

        # reset target in random depth and velocities as zero
        qpos = self.sim.data.qpos
        qpos[[-7]] = np.array([
            self.initial_state.qpos[-7] + self.np_random.uniform(low=-1, high=0, size=1)[0],
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
        # manually set torsional eye positions to 0
        self.sim.data.qpos[18] = 0  # left - torsional
        self.sim.data.qpos[21] = 0  # right - torsional

    def swap_target_texture(self, texture):
        """ Changes target texture. Valid emotion names are in self.target_textures, which links readable
        texture names to their associated texture ids """
        assert texture in self.target_textures, "{} is not a valid texture!".format(texture)
        new_tex_id = self.target_textures[texture]
        self.sim.model.mat_texid[self._target_material_id] = new_tex_id