import envs.ig_parkour.terrain_perception.base_perception as base_perception
import util.geom_util as geom_util
import util.terrain_util as terrain_util
import torch

class Grid_HeightField_Perception(base_perception.Base_Terrain_Perception):
    name = 'HF_GRID'
    def __init__(self, config, num_envs, device):
        self.num_envs = num_envs
        self.device = device

        self.dx = config["dx"]
        self.dy = config["dy"]
        self.num_x_neg = config["num_x_neg"]
        self.num_x_pos = config["num_x_pos"]
        self.num_y_neg = config["num_y_neg"]
        self.num_y_pos = config["num_y_pos"]
        center = config["center"]

        center = torch.tensor(center, dtype=torch.float32, device=self.device)
        self._xy_points = geom_util.get_xy_grid_points(center, self.dx, self.dy, 
                                                          self.num_x_neg, self.num_x_pos, self.num_y_neg, self.num_y_pos)
        self._xy_points = self._xy_points.view(-1,2)
    def get_obs(self):
        return self._xy_points

    def get_num_points(self):
        return self._xy_points.shape[0]