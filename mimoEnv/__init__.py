from gym.envs.registration import register

register(id='MIMo-v0',
         entry_point='mimoEnv.envs:MIMoTestEnv',
         max_episode_steps=1000,
         )

register(id='MIMoReach-v0',
         entry_point='mimoEnv.envs:MIMoReachEnv',
         max_episode_steps=1000,
         )

register(id='MIMoStandup-v0',
         entry_point='mimoEnv.envs:MIMoStandupEnv',
         max_episode_steps=500, 
         )

register(id='MIMoDrawer-v0',
         entry_point='mimoEnv.envs:MIMoDrawerEnv',
         max_episode_steps=250,
         )