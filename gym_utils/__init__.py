
import gym

from .continuous_cartpole import ContinuousCartPoleEnv

gym.register(id='ContinuousCartpole-v0',
             entry_point='gym_utils.continuous_cartpole:ContinuousCartPoleEnv',
             max_episode_steps=200)

gym.register(id='SparsePendulum-v0',
             entry_point='gym_utils.sparse_pendulum:SparsePendulumEnv',
             max_episode_steps=200)

gym.register(id='ToyCircle-v0',
             entry_point='gym_utils.toy_circle:ToyCircleEnv',
             max_episode_steps=200)