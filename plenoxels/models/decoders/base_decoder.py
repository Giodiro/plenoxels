import abc

import torch.nn as nn

class BaseDecoder(nn.Module, abc.ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def compute_density(self, features, rays_d):
        pass

    @abc.abstractmethod
    def compute_color(self, features, rays_d):
        pass
