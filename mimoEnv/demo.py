import gym
import time
import numpy as np
import mimoEnv

env = gym.make("MIMoDemo-v0")

max_steps = 100 
poses = [
    {'name':'wave1', 'face':'neutral', 'duration':2, 'qpos':np.array([0, 0, 0.33, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  -0.5, 0, 0, 0, 0, 0, 0.313826, 0.9572, -1.68454, -1.13813, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])},
    {'name':'wave2', 'face':'neutral', 'duration':30, 'qpos':np.array([0, 0, 0.33, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.5, 0, 0, 0, 0, 0, 0.313826, 0.9572, -1.68454, -2.41624, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])},
    {'name':'wave1', 'face':'neutral', 'duration':30, 'qpos':np.array([0, 0, 0.33, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.5, 0, 0, 0, 0, 0, 0.313826, 0.9572, -1.68454, -1.13813, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])},
    {'name':'wave2', 'face':'neutral', 'duration':30, 'qpos':np.array([0, 0, 0.33, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.5, 0, 0, 0, 0, 0, 0.313826, 0.9572, -1.68454, -2.41624, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])},
    {'name':'wave1', 'face':'neutral', 'duration':30, 'qpos':np.array([0, 0, 0.33, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.5, 0, 0, 0, 0, 0, 0.313826, 0.9572, -1.68454, -1.13813, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])},
    {'name':'righthand', 'face':'neutral', 'duration':50,'qpos':np.array([0, 0, 0.33, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.54236, 0.47822, 0, 0, 0, 0, 0, 0.695981, 0.9106, -0.45332, -0.795545, 0.28278, 0.708715, 0.229634, -1.37069, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])},
    {'name':'righthand', 'face':'neutral', 'duration':20,'qpos':np.array([0, 0, 0.33, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.54236, 0.47822, 0, 0, 0, 0, 0, 0.695981, 0.9106, -0.45332, -0.795545, 0.28278, 0.708715, 0.229634, -1.37069, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])},
    {'name':'lefthand', 'face':'neutral', 'duration':50, 'qpos':np.array([0, 0, 0.33, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.92976, 0.26734, 0, -0.180642, 0, -0.180642, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.352041, 0.9572, -0.409865, -1.28307, 1.24109, 0.78639, -0.678208, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])},
    {'name':'lefthand', 'face':'neutral', 'duration':20, 'qpos':np.array([0, 0, 0.33, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.92976, 0.26734, 0, -0.180642, 0, -0.180642, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.352041, 0.9572, -0.409865, -1.28307, 1.24109, 0.78639, -0.678208, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])},
    {'name':'backstretch1', 'face':'neutral', 'duration':50, 'qpos':np.array([0, 0, 0.33, 1, 0, 0, 0, 0, 0.3142, 0, 0, 0.3142, 0, 1.937, 0.00374, 0, 0, 0, 0, 0, 2.059, 1.0038, 0, -0.821898, 0, 0, 0, -0.666865, -0.4887, 0.9572, -0.409865, -1.28307, 0, 0.00964, 0.00047, -0.784169, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])},
    {'name':'backstretch1', 'face':'neutral', 'duration':20, 'qpos':np.array([0, 0, 0.33, 1, 0, 0, 0, 0, 0.3142, 0, 0, 0.3142, 0, 1.937, 0.00374, 0, 0, 0, 0, 0, 2.059, 1.0038, 0, -0.821898, 0, 0, 0, -0.666865, -0.4887, 0.9572, -0.409865, -1.28307, 0, 0.00964, 0.00047, -0.784169, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])},
    {'name':'backstretch2', 'face':'neutral', 'duration':75, 'qpos':np.array([0, 0, 0.33, 1, 0, 0, 0, 0, -0.3142, 0, 0, -0.3142, 0, -1.937, 0.00374, 0, 0, 0, 0, 0, -0.4887, 1.0038, 0, -0.821898, 0, 0, 0, -0.666865, 2.059, 0.9572, -0.409865, -1.28307, 0, 0.00964, 0.00047, -0.784169, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])},
    {'name':'backstretch2', 'face':'neutral', 'duration':20, 'qpos':np.array([0, 0, 0.33, 1, 0, 0, 0, 0, -0.3142, 0, 0, -0.3142, 0, -1.937, 0.00374, 0, 0, 0, 0, 0, -0.4887, 1.0038, 0, -0.821898, 0, 0, 0, -0.666865, 2.059, 0.9572, -0.409865, -1.28307, 0, 0.00964, 0.00047, -0.784169, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])},
    {'name':'balance1', 'face':'neutral', 'duration':100, 'qpos':np.array([0, 0, 0.33, 1, 0, 0, 0, 0.107492, 0.122538, 0, 0.150784, 0, 0.00174, 0.03874, 0.71546, 0, 0, 0, 0, 0, -0.004637, 1.1669, 0, -0.821898, 0, 0, 0, -0.666865, 0.0081015, 0.7009, -0.004285, -0.439784, 0, 0.00964, 0.00047, -0.784169, 0, 0, 0, 0, 0, 0, 0, 0, -2.2409, -0.575598, 0, -1.06154, 0.143875, 0, 0, 0])},
    {'name':'balance1', 'face':'neutral', 'duration':10, 'qpos':np.array([0, 0, 0.33, 1, 0, 0, 0, 0.107492, 0.122538, 0, 0.150784, 0, 0.00174, 0.03874, 0.71546, 0, 0, 0, 0, 0, -0.004637, 1.1669, 0, -0.821898, 0, 0, 0, -0.666865, 0.0081015, 0.7009, -0.004285, -0.439784, 0, 0.00964, 0.00047, -0.784169, 0, 0, 0, 0, 0, 0, 0, 0, -2.2409, -0.575598, 0, -1.06154, 0.143875, 0, 0, 0])},
    {'name':'balance2', 'face':'neutral', 'duration':30, 'qpos':np.array([0, 0, 0.33, 1, 0, 0, 0, 0.167492, 0.102538, 0, 0.150784, 0, 0.00174, 0.03874, 0.71546, 0, 0, 0, 0, 0, -0.004637, 1.1669, 0, -0.821898, 0, 0, 0, -0.666865, 0.0081015, 0.7009, -0.004285, -0.439784, 0, 0.00964, 0.00047, -0.784169, 0, 0, 0, 0, 0, 0, 0, 0, -2.2409, -0.575598, 0, -2.25791, -0.73513, 0, 0, 0])},
    {'name':'balance1', 'face':'neutral', 'duration':30, 'qpos':np.array([0, 0, 0.33, 1, 0, 0, 0, 0.107492, 0.122538, 0, 0.150784, 0, 0.00174, 0.03874, 0.71546, 0, 0, 0, 0, 0, -0.004637, 1.1669, 0, -0.821898, 0, 0, 0, -0.666865, 0.0081015, 0.7009, -0.004285, -0.439784, 0, 0.00964, 0.00047, -0.784169, 0, 0, 0, 0, 0, 0, 0, 0, -2.2409, -0.575598, 0, -1.06154, 0.143875, 0, 0, 0])},
    #{'name':'backstretch', 'face':'neutral','duration':100,'qpos':np.array([0, 0, 0.33, 1, 0, 0, 0, 0, 0, -0.259395, 0, 0, 0, 0, -1.07702, 0, 0, 0, 0, 0, -0.208453, 2.495, -1.65557, 0.08727, 0, 0, 0, 0, -0.208453, 2.4717, -1.65557, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])},
    #{'name':'backstretch', 'face':'neutral','duration':20,'qpos':np.array([0, 0, 0.33, 1, 0, 0, 0, 0, 0, -0.259395, 0, 0, 0, 0, -1.07702, 0, 0, 0, 0, 0, -0.208453, 2.495, -1.65557, 0.08727, 0, 0, 0, 0, -0.208453, 2.4717, -1.65557, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])},
    #{'name':'cross', 'face':'neutral','duration':100, 'qpos':np.array([0, 0, 0.33, 1, 0, 0, 0, -0.2443, 0.3142, 0.5323, -0.4712, -0.3142, 0.5323, -0.17433, -0.57618, 0.21996, 0, 0, 0, 0, -0.4887, 1.7494, -1.728, 0, 0, 0, 0, -0.388268, 0.861581, 1.1902, -0.757505, 0, 0, 0, 0, 0, -1.50662, -0.8901, -0.201752, 0, -1.1, -0.481046, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])},
    {'name':'airplane', 'face':'neutral','duration':100, 'qpos':np.array([0, 0, 0.33, 1, 0, 0, 0, 0.2443, 0, 0.5323, 0.089528, 0, 0.5323, 0, -0.93204, -0.19552, 0, 0, 0, 0, -0.195715, 1.6096, -0.380895, 0, 0, 0, 0, 0, 0.27561, 0.7708, -0.004285, 0, -0.64411, 0, 0, 0, 0, 0, 0, 0.00478975, 0, 0, 0, 0, 0.3491, -0.142416, 0.0849205, -1.45166, 0, 0, 0, 0])},
    {'name':'airplane', 'face':'neutral','duration':30, 'qpos':np.array([0, 0, 0.33, 1, 0, 0, 0, 0.2443, 0, 0.5323, 0.089528, 0, 0.5323, 0, -0.93204, -0.19552, 0, 0, 0, 0, -0.195715, 1.6096, -0.380895, 0, 0, 0, 0, 0, 0.27561, 0.7708, -0.004285, 0, -0.64411, 0, 0, 0, 0, 0, 0, 0.00478975, 0, 0, 0, 0, 0.3491, -0.142416, 0.0849205, -1.45166, 0, 0, 0, 0])},
    {'name':'T', 'face':'neutral', 'duration':100, 'qpos':np.array([0, 0, 0.33, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.3766, 0, 0, 0, 0, 0, 0, 0, 1.3766, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])},
    {'name':'wave2', 'face':'neutral', 'duration':50, 'qpos':np.array([0, 0, 0.33, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.5, 0, 0, 0, 0, 0, 0.313826, 0.9572, -1.68454, -2.41624, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])},
    {'name':'wave1', 'face':'neutral', 'duration':30, 'qpos':np.array([0, 0, 0.33, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.5, 0, 0, 0, 0, 0, 0.313826, 0.9572, -1.68454, -1.13813, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])},
    {'name':'wave2', 'face':'neutral', 'duration':30, 'qpos':np.array([0, 0, 0.33, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.5, 0, 0, 0, 0, 0, 0.313826, 0.9572, -1.68454, -2.41624, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])},
    {'name':'wave1', 'face':'neutral', 'duration':30, 'qpos':np.array([0, 0, 0.33, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.5, 0, 0, 0, 0, 0, 0.313826, 0.9572, -1.68454, -1.13813, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])},
    {'name':'wave1', 'face':'neutral', 'duration':200, 'qpos':np.array([0, 0, 0.33, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.5, 0, 0, 0, 0, 0, 0.313826, 0.9572, -1.68454, -1.13813, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])},
]

obs = env.reset()
old_pose = poses[0]['qpos']
for pose in poses[1:]:
    
    new_pose = pose['qpos']
    face = pose['face']
    timesteps = pose['duration']
    for step in range(timesteps):
        _,_,_,_ = env.step(np.zeros(env.action_space.shape))
        env.sim.data.qpos[:] = old_pose + (new_pose-old_pose) * step/(timesteps-1)
        env.swap_facial_expressions(face)
        env.render()
    old_pose = new_pose

env.close()