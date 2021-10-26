
import numpy as np
import torch

from gym import Wrapper, spaces
from torch.distributions import MultivariateNormal
from skimage.transform import resize


class RGBArrayAsObservationWrapper(Wrapper):
    def __init__(self, env, height=84, width=84):
        super(RGBArrayAsObservationWrapper, self).__init__(env)
        self.height = height
        self.width = width
        self.observation_space = spaces.Box(low=0, high=255, shape=[3, height, width], dtype=np.uint8)
        self._max_episode_steps = env.spec.max_episode_steps

    def reset(self, **kwargs):
        obs = self.env.reset()
        if len(obs.shape) != 3:
            obs = self.env.render(mode="rgb_array")
            obs = self._preprocess_obs(obs)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if len(obs.shape) != 3:
            obs = self.env.render(mode="rgb_array")
            obs = self._preprocess_obs(obs)
        return obs, reward, done, info

    def _preprocess_obs(self, obs):
        obs = resize(obs, (self.height, self.width)) * 255
        obs = np.transpose(obs, [2, 0, 1])
        return obs


class NoisyObservationWrapper(Wrapper):
    def __init__(self, env, noisy_dims, noise_std=1.):
        """
        Given an environment that produces vector observations, this environment wraps it and adds
        noise dimensions that are sampled from an isotropic Gaussian.
        @param env: env to be wrapped.
        @param noisy_dims: wrapped observation will have original * noisy_dims dimensions.
        @param noise_std: std of the Gaussian noise distribution used for extra dimensions.
        """
        super(NoisyObservationWrapper, self).__init__(env)
        self._max_episode_steps = env.spec.max_episode_steps
        self.true_obs_dim = env.observation_space.shape[0]
        self.noisy_obs_dim = self.true_obs_dim * noisy_dims
        self.observation_space = spaces.Box(
            low=-np.inf * np.ones(self.noisy_obs_dim),
            high=np.inf * np.ones(self.noisy_obs_dim))
        self.noise_dist = MultivariateNormal(
                loc=torch.zeros(self.noisy_obs_dim - self.true_obs_dim),
                covariance_matrix=(noise_std**2)*torch.eye(self.noisy_obs_dim - self.true_obs_dim))

    def reset(self, **kwargs):
        obs = self.env.reset()
        obs = self.add_noise(obs)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.add_noise(obs)
        return obs, reward, done, info

    def add_noise(self, obs):
        obs = torch.from_numpy(obs).float()
        noise = self.noise_dist.sample().float()
        obs = torch.cat([obs, noise]).numpy()

        return obs
