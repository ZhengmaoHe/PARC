import envs.ig_parkour.terrain_perception.base_perception as base_perception
import util.geom_util as geom_util
import util.terrain_util as terrain_util
import torch

class Grid_HeightFieldC2F_Perception(base_perception.Base_Terrain_Perception):
    name = 'HF_GRID_C2F'
    def __init__(self, config, num_envs, device):
        self.num_envs = num_envs
        self.device = device

        self.dx = config["dx"]
        self.dy = config["dy"]
        self.num_x = config["num_x"]
        self.num_y = config["num_y"]
        self.root_y_offset = config["root_y_offset"]
        self.dx_incr_rate = config["dx_incr_rate"]
        self.dy_incr_rate = config["dy_incr_rate"]
        center = config["center"]

        center = torch.tensor(center, dtype=torch.float32, device=self.device)
        self._xy_points = geom_util.get_xy_grid_points_coarse2fine(center, self.dx, self.dy, 
                                                                   self.num_x, self.num_y, 
                                                                   self.root_y_offset, 
                                                                   self.dx_incr_rate, self.dy_incr_rate)
    
    def get_obs(self):
        return self._xy_points

    def get_num_points(self):
        return self._xy_points.shape[0]
    