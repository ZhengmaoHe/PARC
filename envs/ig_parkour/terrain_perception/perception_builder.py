import envs.ig_parkour.terrain_perception.base_perception as base_perception
import envs.ig_parkour.terrain_perception.hf_grid_coarse2fine as hf_grid_c2f
import envs.ig_parkour.terrain_perception.hf_grid as hf_grid
import envs.ig_parkour.terrain_perception.hf_ray as hf_ray

from util.logger import Logger

def build_perception(percetion_config, num_envs, device):
    tp_name = percetion_config["name"]
    Logger.print("Using terrain obs : {}".format(tp_name))
    if (tp_name == "HF_GRID"):
        percetion = hf_grid.Grid_HeightField_Perception(config=percetion_config, num_envs=num_envs, device=device)

    elif (tp_name == "HF_RAY"):
        percetion = hf_ray.Ray_HeightField_Perception(config=percetion_config, num_envs=num_envs, device=device)
    
    elif (tp_name == "HF_GRID_C2F"):
        percetion = hf_grid_c2f.Grid_HeightFieldC2F_Perception(config=percetion_config, num_envs=num_envs, device=device)
    
    return percetion