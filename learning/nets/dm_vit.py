import numpy as np
import torch
import torch.nn as nn
import learning.nets.net_builder as net_builder
import learning.nets.cnn_tokenizer as cnn_tokenizer
from collections import OrderedDict
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from diffusion.diffusion_util import PositionalEncoding

class MLP(nn.Module):
    def __init__(self, dims, hidden_act = nn.SiLU(), device="cuda:0", dropout=0.0):
        super().__init__()

        layers = []

        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.Dropout(dropout))
            layers.append(hidden_act)
        layers.append(nn.Linear(dims[-2], dims[-1]))

        self.model = nn.Sequential(*layers)
        self.to(device)

        return
    def forward(self, x):

        return self.model(x)

class DM_ViT(nn.Module):
    def __init__(self, input_dict, activation):
        super().__init__()

        self._obs_shapes = input_dict["obs_shapes"]
        self._d_model = input_dict["d_model"]
        device = input_dict["device"] # hack

        # self._flat_obs_slices = []
        # curr_dim = 0
        self._num_tokens = 0
        self._embed = OrderedDict()
        for key in self._obs_shapes:
            
            curr_obs_shape = self._obs_shapes[key]['shape']

            if key == "hf":

                continue

            if len(curr_obs_shape) > 1:
                assert len(curr_obs_shape) == 2
                obs_dim = curr_obs_shape[1]
                curr_num_tokens = curr_obs_shape[0]
            else:
                obs_dim = curr_obs_shape[0]
                curr_num_tokens = 1

            self._embed[key] = nn.Linear(obs_dim, self._d_model, device=device)
            torch.nn.init.zeros_(self._embed[key].bias)
            self._num_tokens += curr_num_tokens
        #     self._flat_obs_slices.append(slice(curr_dim, curr_dim+flat_obs_dim))
        #     curr_dim += flat_obs_dim
        
        # num_x_neg = input_dict["local_grid"]["num_x_neg"]
        # num_x_pos = input_dict["local_grid"]["num_x_pos"]
        # num_y_neg = input_dict["local_grid"]["num_y_neg"]
        # num_y_pos = input_dict["local_grid"]["num_y_pos"]

        self._grid_dim_x = self._obs_shapes["hf"]['shape'][0]#num_x_neg + num_x_pos + 1
        self._grid_dim_y = self._obs_shapes["hf"]['shape'][1]#num_y_neg + num_y_pos + 1

        self._cnn, info = net_builder.build_net(input_dict["cnn_name"], input_dict, nn.LeakyReLU())
        #assert isinstance(self._cnn, cnn_tokenizer.CNN_Tokenizer) # doesnt work with subclasses

        self._cnn_token_linear = nn.Linear(self._cnn._out_token_dim, self._d_model)
        torch.nn.init.zeros_(self._cnn_token_linear.bias)

        self._num_tokens += self._cnn._num_out_tokens

        dropout = input_dict["dropout"]
        num_heads = input_dict["num_heads"]
        d_hid = input_dict["d_hid"]
        num_layers = input_dict["num_layers"]

        self._in_pos_encoding = PositionalEncoding(self._d_model, dropout, seq_len=self._num_tokens) 
        encoder_layers = TransformerEncoderLayer(self._d_model, num_heads, d_hid, dropout, activation="gelu", batch_first=True)
        self._transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

        #critic_encoder_layers = TransformerEncoderLayer(self._d_model, num_heads, d_hid, dropout, activation="gelu", batch_first=True)
        #self._critic_transformer_encoder = TransformerEncoder(critic_encoder_layers, num_layers)


        final_layers = [
            nn.Linear(self._d_model, self._d_model),
            nn.ReLU()
        ]
        torch.nn.init.zeros_(final_layers[0].bias)
        self._final_act_actor = nn.Sequential(*final_layers)
        
        final_layers_critic = [
            nn.Linear(self._d_model, self._d_model),
            nn.ReLU()
        ]
        torch.nn.init.zeros_(final_layers[0].bias)
        self._final_act_critic = nn.Sequential(*final_layers_critic)

        return
    
    def _tokenize_obs(self, obs):
        ret_tokens = []

        curr_dim = 0
        for key in self._obs_shapes:
            curr_obs_shape = self._obs_shapes[key]['shape']
            if len(curr_obs_shape) > 1:
                assert len(curr_obs_shape) == 2
                flat_obs_dim = curr_obs_shape[0] * curr_obs_shape[1]

                unflat_obs = obs[:, curr_dim:curr_dim + flat_obs_dim].view(-1, curr_obs_shape[0], curr_obs_shape[1])
            else:
                flat_obs_dim = curr_obs_shape[0]
                unflat_obs = obs[:, curr_dim:curr_dim + flat_obs_dim]

            curr_dim += flat_obs_dim
            
            if key == "hf":
                token = self._cnn(unflat_obs.unsqueeze(1)) # need to unsqueeze channel dim?
                token = self._cnn_token_linear(token)
            else:
                if len(unflat_obs.shape) == 2:
                    unflat_obs = unflat_obs.unsqueeze(1) # unsqueeze sequence dimension
                token = self._embed[key](unflat_obs)

            assert len(token.shape) == 3

            ret_tokens.append(token)


        ret_tokens = torch.cat(ret_tokens, dim=1) # (B, S, d_model)
        return ret_tokens
    
    def get_out_token_dim(self):
        return self._d_model
    
    def forward(self, x):
        x = self._tokenize_obs(x)
        x = self._in_pos_encoding(x)
        x = self._transformer_encoder(x) # (B, S, d_model)
        #x = self._final_act(x[:, 0])# only return the first token
        return x

    def forward_actor(self, obs):
        return self._final_act_actor(self.forward(obs)[:, 0]) * 0.25
        
    def forward_critic(self, obs):
        return self._final_act_critic(self.forward(obs)[:, 1])

    # def forward_actor(self, obs):
    #     tokens = self._tokenize_obs(obs)
    #     tokens = self._in_pos_encoding(tokens)
    #     out_tokens = self._actor_transformer_encoder(tokens) 
    #     return out_tokens[:, 0]
    

    # def forward_critic(self, obs):
    #     tokens = self._tokenize_obs(obs)
    #     tokens = self._in_pos_encoding(tokens)
    #     out_tokens = self._critic_transformer_encoder(tokens) 
    #     return out_tokens[:, 0]


# A two layer MLP with a convnet to preprocess the heightfield
def build_net(input_dict, activation):

    net = DM_ViT(input_dict, activation)
    info = dict()  

    return net, info
