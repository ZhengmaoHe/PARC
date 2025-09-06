import numpy as np
import torch
import torch.nn as nn
import learning.nets.net_builder as net_builder
import learning.nets.cnn_tokenizer as cnn_tokenizer
from collections import OrderedDict
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from diffusion.diffusion_util import PositionalEncoding

class MLP(nn.Module):
    def __init__(self, dims, activation, device="cuda:0"):
        super().__init__()

        layers = []

        for i in range(len(dims) - 1):
            curr_layer = nn.Linear(dims[i], dims[i+1])
            torch.nn.init.zeros_(curr_layer.bias)
            layers.append(curr_layer)
            layers.append(activation())

        self.model = nn.Sequential(*layers)
        self.to(device)

        return
    
    def forward(self, x):

        return self.model(x)

class DM_CNN_MLP(nn.Module):
    def __init__(self, input_dict, activation):
        super().__init__()

        self._obs_shapes = input_dict["obs_shapes"]

        self._grid_dim_x = self._obs_shapes["hf"]['shape'][0]#num_x_neg + num_x_pos + 1
        self._grid_dim_y = self._obs_shapes["hf"]['shape'][1]#num_y_neg + num_y_pos + 1

        self._cnn, info = net_builder.build_net(input_dict["cnn_name"], input_dict, nn.LeakyReLU())
        #assert isinstance(self._cnn, cnn_tokenizer.CNN_Tokenizer) # doesnt work with subclasses

        non_hf_dim = 0
        for key in self._obs_shapes:
            if key == "hf":
                continue
            shape = self._obs_shapes[key]["shape"]
            curr_dim = 1
            for i in range(len(shape)):
                curr_dim *= shape[i]
            non_hf_dim += curr_dim

        self._non_hf_dim = non_hf_dim

        flat_cnn_out_dim = self._cnn._cnn_out_shape[0] * self._cnn._cnn_out_shape[1] * self._cnn._cnn_out_shape[2]
        
        out_dim = 512
        dims = [
            non_hf_dim + flat_cnn_out_dim,
            1024, 1024, out_dim
        ]
        self._actor_net = MLP(dims=dims, activation=activation)
        self._critic_net = MLP(dims=dims, activation=activation)

        self._actor_out_dim = out_dim
        self._critic_out_dim = out_dim


        return
    
    def forward(self, x):
        batch_size = x.shape[0]

        hf = x[..., self._non_hf_dim:].view(batch_size, 1, self._grid_dim_x, self._grid_dim_y)
        non_hf = x[..., :self._non_hf_dim]

        conved_hf = self._cnn(hf)
        conved_hf = conved_hf.reshape(batch_size, -1)
        x = torch.cat([non_hf, conved_hf], dim=1)
        return x

    def forward_actor(self, obs):
        x = self.forward(obs)
        return self._actor_net(x)
        
    def forward_critic(self, obs):
        x = self.forward(obs)
        return self._critic_net(x)


# A two layer MLP with a convnet to preprocess the heightfield
def build_net(input_dict, activation):

    net = DM_CNN_MLP(input_dict, activation)
    info = dict()  

    return net, info
