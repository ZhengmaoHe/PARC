import envs.ig_parkour.terrain_perception.base_perception as base_perception
import util.geom_util as geom_util
import torch

class Ray_HeightField_Perception(base_perception.Base_Terrain_Perception):
    name = 'HF_RAY'
    def __init__(self, config, num_envs, device):
        self.num_envs = num_envs
        self.device = device
        self.ray_points_behind = config["ray_points_behind"]
        self.ray_points_ahead = config["ray_points_ahead"]
        self.ray_num_left = config["ray_num_left"]
        self.ray_num_right = config["ray_num_right"]
        self.ray_dx = config["ray_dx"]
        self.ray_angle = config["ray_angle"]
        self._xy_points = geom_util.get_xy_points_cone(
                                    center=torch.zeros(size=(2,), dtype=torch.float32, device=self.device),
                                    dx=self.ray_dx,
                                    num_neg=self.ray_points_behind,
                                    num_pos=self.ray_points_ahead,
                                    num_rays_neg=self.ray_num_left,
                                    num_rays_pos=self.ray_num_right,
                                    angle_between_rays=self.ray_angle)
        
    #temporalily hack
    def get_obs(self):
        return self._xy_points

    def get_num_points(self):
        return torch.prod(self._xy_points.shape[:-1])