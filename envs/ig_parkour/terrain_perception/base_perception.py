import enum
import abc
import torch
class Terrain_Obs_Type(enum.Enum):
    RAY = 1
    HF = 2
    VISION = 3
    PC = 4

class Base_Terrain_Perception:
    def __init__(self, config):
        self.config = config

    @abc.abstractmethod
    def get_obs(self, char_state, terrain):
        return


    def get_dim(self):
        obs_shape = self.get_shape()
        return torch.prod(obs_shape)

    def get_shape(self):
        obs = self.get_obs(num_env=2)
        return obs.shape[1:]



