import torch
import numpy as np
import pickle
import yaml
import os

import envs.base_env as base_env
import envs.ig_parkour.mgdm_dm_util as mgdm_dm_util
import util.torch_util as torch_util
import util.terrain_util as terrain_util
import util.geom_util as geom_util
import anim.motion_lib as motion_lib

import tools.procgen.astar as astar
import tools.procgen.mdm_path as mdm_path
import util.motion_util as motion_util
import diffusion.gen_util as gen_util
import zmotion_editing_tools.motion_edit_lib as medit_lib

SIM_CHAR_IDX = 0
REF_CHAR_IDX = 1

class DeepMimicEnv(mgdm_dm_util.RefCharEnv):
    def __init__(self, config, num_envs, device, visualize, char_model):
        env_config = config["env"]
        dm_config = env_config["dm"]
        self._random_reset_pos = dm_config.get("random_reset_pos", False)

        super().__init__(config, num_envs, device, visualize, char_model)

        self._min_motion_weight = dm_config.get("min_motion_weight", 0.01)
        self._demo_mode = env_config["demo_mode"]
        self._rand_reset = env_config["rand_reset"]
        self._target_xy_future_time_max = dm_config["target_xy_future_time_max"]
        self._target_xy_future_time_min = dm_config["target_xy_future_time_min"]
        self._ignore_fail_rates = dm_config.get("ignore_fail_rates", False)
        
        self._target_pos_xy_user_movement = torch.zeros((num_envs,3),device=device)
        self._terrains_per_motion = dm_config["terrains_per_motion"]
        q = dm_config["fail_rate_quantiles"]
        self._fail_rate_quantiles = torch.tensor(q, dtype=torch.float32, device=self._device)
        self._one_motion_mode = False
        self._selected_motion_id = 0
        self._terrain_build_mode = dm_config.get("terrain_build_mode", "square")

        self._record_motion_frame_hist = dm_config.get("record_motion_frame_hist", False)
        if self._record_motion_frame_hist:
            self._hist_length = 5
            self._motion_frame_hist = []

        mdm_model_path = dm_config.get("mdm_model", None)
        if mdm_model_path is not None:
            with open(mdm_model_path, "rb") as f:
                self._mdm_model = pickle.load(f)

        motion_file = dm_config["motion_file"]
        self._motion_classes = dm_config["motion_classes"]
        # self._motion_class_to_id_map = dict()
        # id = 0
        # for motion_class in self._motion_classes:
        #     self._motion_class_to_id_map[motion_class] = id
        #     id += 1
        #print(self._motion_class_to_id_map)
        #exit()
        self._motion_id_class = []
        self._has_motion_classes = len(self._motion_classes) > 0 and os.path.splitext(motion_file)[1] == ".yaml"
        self._has_motion_classes = self._has_motion_classes and dm_config["has_motion_classes"]
        if self._has_motion_classes:
            with open(motion_file, "rb") as f:
                motion_yaml = yaml.safe_load(f)
            
            for elem in motion_yaml["motions"]:
                motion_path = elem['file']
                class_found = False
                for i in range(len(self._motion_classes)):
                    motion_class = self._motion_classes[i]
                    if "/" + motion_class + "/" in motion_path:
                        self._motion_id_class.append(i)
                        class_found = True
                        break

                assert class_found, "motion class not found in: " + motion_path
            self._motion_id_class = torch.tensor(self._motion_id_class, dtype=torch.int64, device=device)
            self._motion_class_to_ids_map = dict()
            for i in range(len(self._motion_classes)):
                self._motion_class_to_ids_map[self._motion_classes[i]] = torch.nonzero(self._motion_id_class == i).squeeze(-1)
            #print(self._motion_class_to_ids_map[self._motion_classes[i]])
        #exit()
            
        
        self._motion_lib = motion_lib.MotionLib(motion_input=motion_file, 
                                                kin_char_model=char_model,
                                                device=device,
                                                contact_info=True)
        if len(self._motion_id_class) > 0:
            assert self._motion_id_class.shape[0] == self._motion_lib.num_motions()
        
        self._motion_ids = torch.zeros(size=(num_envs,), device=self._device, dtype=torch.int64)
        self._motion_terrain_ids = torch.zeros(size=(num_envs,), device=self._device, dtype=torch.int64)
        self._motion_time_offsets = torch.zeros(size=(num_envs,), device=self._device, dtype=torch.float32)

        #self._motion_id_fail_counts = torch.ones(size=[self._motion_lib.num_motions()], dtype=torch.int64, device=self._device)
        #self._motion_id_done_counts = torch.ones(size=[self._motion_lib.num_motions()], dtype=torch.int64, device=self._device)
        if "fail_rates_path" in dm_config:
            print("LOADING SAVED FAIL RATES")
            self._motion_id_fail_rates = torch.load(dm_config["fail_rates_path"]).to(dtype=torch.float32, device=self._device)

        else:
            self._motion_id_fail_rates = torch.ones(size=[self._motion_lib.num_motions()], dtype=torch.float32, device=self._device)
        self._ema_weight = 0.01

        self._motion_start_time_fraction = torch.zeros(size=[num_envs], dtype=torch.float32, device=device)
        #self.set_reset_motion_time(0.0)
        #self.set_user_reset_root_offset(0.0, 0.0, 0.0)
        #self.set_user_reset_heading_offset(0.0)
        return
    
    def build_terrain(self, env_config, terrain_save_path, x_offset = 0.0, y_offset = 0.0):
        if self._terrain_build_mode == "square":
            return self.build_terrain_square(env_config, terrain_save_path, x_offset = x_offset, y_offset = y_offset)
        elif self._terrain_build_mode == "wide":
            return self.build_terrain_wide(env_config, terrain_save_path, x_offset = x_offset, y_offset = y_offset)
        elif self._terrain_build_mode == "file":
            return self.load_motion_terrain_file(env_config, terrain_save_path)
        else:
            assert False, "unsupported terrain build mode"

    def load_motion_terrain_file(self, env_config, terrain_save_path):

        motion_terrain_yaml_path = env_config["dm"]["motion_file"]
        with open(motion_terrain_yaml_path, "r") as f:
            motion_terrain_yaml = yaml.safe_load(f)
            terrain_path = motion_terrain_yaml["terrain"]

        with open(terrain_path, "rb") as f:
            terrain = pickle.load(f)["terrain"]

        

        self._terrain = terrain
        self._terrain.to_torch(device=self._device)

        verts, tris = terrain_util.convert_heightfield_to_voxelized_trimesh(
                        hf = self._terrain.hf,
                        min_x = self._terrain.min_point[0].item(),
                        min_y = self._terrain.min_point[1].item(),
                        dx = self._terrain.dxdy[0].item(),
                        padding=0)
        
        # weird but we depend on it
        verts = [[verts]]
        tris = [[tris]]

        self._terrains_per_motion = 1
        #self._dm_motion_offsets = torch.zeros(size=[self._motion_lib.num_motions(), self._terrains_per_motion, 2], dtype=torch.float32, device=self._device)

        self._dm_motion_offsets = []
        for elem in motion_terrain_yaml["motions"]:
            motion_filepath = elem["file"]
            with open(motion_filepath, "rb") as f:
                motion_data = pickle.load(f)
                if "min_point_offset" in motion_data:
                    min_point_offset = motion_data["min_point_offset"]
                else:
                    min_point_offset = torch.tensor([0.0, 0.0], dtype=torch.float32, device=self._device)
                self._dm_motion_offsets.append(min_point_offset)
        self._dm_motion_offsets = torch.stack(self._dm_motion_offsets)
        self._dm_motion_offsets = self._dm_motion_offsets.to(dtype=torch.float32, device=self._device)
        self._dm_motion_offsets = self._dm_motion_offsets.unsqueeze(1).expand(-1, self._terrains_per_motion, -1)

        # use data within the motions to get the motion offsets


        save_data = {
            "terrain": self._terrain,
            "terrains_per_motion": self._terrains_per_motion,
            "motion_offsets": self._dm_motion_offsets,
            "all_terrain_verts": verts,
            "all_terrain_tris": tris
        }

        print("saving terrain data to", terrain_save_path)
        with open(terrain_save_path, "wb") as f:
            pickle.dump(save_data, f)

        return verts, tris

    def build_terrain_square(self, env_config, terrain_save_path, x_offset = 0.0, y_offset = 0.0):
        hmap_config = env_config["dm"]["heightmap"]
        dx = float(hmap_config["horizontal_scale"])
        dy = dx
        padding = hmap_config["padding"]
        #padding = 0.0
        num_padding_cells = padding / dx
        assert (round(num_padding_cells) - num_padding_cells) < 1e-5
        num_padding_cells = int(num_padding_cells)

        assert self._terrains_per_motion == 1

        num_motions = self._motion_lib.num_motions()

        # To help reset agents into their respective terrain
        self._dm_motion_offsets = torch.zeros(size=[self._motion_lib.num_motions(), self._terrains_per_motion, 2], dtype=torch.float32, device=self._device)

        global_terrain_min_point = torch.zeros(size=[2], dtype=torch.float32, device=self._device)
        global_terrain_max_point = global_terrain_min_point.clone()

        all_terrain_verts = []
        all_terrain_tris = []

        og_y_offset = y_offset

        num_terrains_x = int(np.ceil(np.sqrt(num_motions)))
        num_terrains_y = int(num_terrains_x)


        # TODO: calculate the max width and length of all terrains, so we know how much each terrain occupies
        dim_x_per_terrain = 0
        dim_y_per_terrain = 0
        for i in range(num_motions):
            curr_terrain = self._motion_lib._terrains[i]
            dim_x_per_terrain = max(curr_terrain.dims[0].item(), dim_x_per_terrain)
            dim_y_per_terrain = max(curr_terrain.dims[1].item(), dim_y_per_terrain)

        dim_x_per_terrain += 2 * num_padding_cells
        dim_y_per_terrain += 2 * num_padding_cells

        first_x_offset = -dim_x_per_terrain * num_terrains_x * dx / 2.0
        first_y_offset = -dim_y_per_terrain * num_terrains_y * dy / 2.0

        og_x_offset = x_offset

        #print(dim_x_per_terrain, dim_y_per_terrain)
        x_offset += first_x_offset

        curr_motion_id = 0
        for i in range(num_terrains_x):
            y_offset = og_y_offset + first_y_offset
            for j in range(num_terrains_y):
                if curr_motion_id > num_motions - 1:
                    break
                curr_motion_all_terrain_verts = []
                curr_motion_all_terrain_tris = []
                curr_terrain = self._motion_lib._terrains[curr_motion_id]
                terrain_min_h = torch.min(curr_terrain.hf).item()
                # copy_terrain = curr_terrain.torch_copy()
                # copy_terrain.pad(num_padding_cells, terrain_min_h)
                # np_terrain = copy_terrain.numpy_copy()
                curr_terrain.pad(num_padding_cells, terrain_min_h)
                np_terrain = curr_terrain.numpy_copy()


                self._dm_motion_offsets[curr_motion_id, 0, 0] = x_offset - np_terrain.min_point[0]
                self._dm_motion_offsets[curr_motion_id, 0, 1] = y_offset - np_terrain.min_point[1]
                np_terrain.min_point[0] = x_offset
                np_terrain.min_point[1] = y_offset

                # todo: keep track of corner vert ids so we can make cheap padding meshes
                terrain_is_flat = abs(np_terrain.hf.max() - np_terrain.hf.min()) < 1e-5
                if terrain_is_flat:
                    # p2 ------- p3
                    # |           |
                    # |           |
                    # |           |
                    # p0 ------- p1
                    min_x = np_terrain.min_point[0]
                    min_y = np_terrain.min_point[1]
                    max_x = min_x + dx*(np_terrain.dims[0]-1)
                    max_y = min_y + dx*(np_terrain.dims[1]-1)
                    z = np_terrain.hf[0, 0]
                    p0 = np.array([min_x - dx/2, min_y - dx/2, z])
                    p1 = np.array([max_x + dx/2, min_y - dx/2, z])
                    p2 = np.array([min_x - dx/2, max_y + dx/2, z])
                    p3 = np.array([max_x + dx/2, max_y + dx/2, z])
                    verts = np.array([p0, p1, p2, p3], dtype=np.float32)
                    tris = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.uint32)
                else:
                    verts, tris = terrain_util.convert_heightfield_to_voxelized_trimesh(
                        hf = np_terrain.hf,
                        min_x = np_terrain.min_point[0],
                        min_y = np_terrain.min_point[1],
                        dx = np_terrain.dxdy[0],
                        padding=0)
                    
                curr_motion_all_terrain_verts.append(verts)
                curr_motion_all_terrain_tris.append(tris)
                all_terrain_verts.append(curr_motion_all_terrain_verts)
                all_terrain_tris.append(curr_motion_all_terrain_tris)  

                # global_terrain_min_point[0] = min(global_terrain_min_point[0].item(), float(np_terrain.min_point[0]))
                # global_terrain_min_point[1] = min(global_terrain_min_point[1].item(), float(np_terrain.min_point[1]))

                # #curr_terrain_max_point = np_terrain.min_point + np_terrain.dims * dx
                # global_terrain_max_point[0] = max(global_terrain_max_point[0].item(), float(np_terrain.min_point[0] + dim_x_per_terrain * dx + padding * 2.0))
                # global_terrain_max_point[1] = max(global_terrain_max_point[1].item(), float(np_terrain.min_point[1] + dim_y_per_terrain * dy + padding * 2.0))

                curr_motion_id += 1
                y_offset += dy * dim_y_per_terrain
            x_offset += dx * dim_x_per_terrain

            

        #new_terrain_dims = torch.round((global_terrain_max_point - global_terrain_min_point) / dx).to(dtype=torch.int)

        # new_terrain_dims = 

        # print("global terrain dims:", new_terrain_dims)
        # print("global terrain min point:", global_terrain_min_point.cpu().numpy())
        # print("global terrain max point:", global_terrain_max_point.cpu().numpy())
        new_terrain_dims = torch.tensor([dim_x_per_terrain * num_terrains_x, 
                                         dim_y_per_terrain * num_terrains_y], dtype=torch.int64, device=self._device)

        self._terrain = terrain_util.SubTerrain("heightmap", new_terrain_dims[0].item(), 
                                                new_terrain_dims[1].item(), dx, dx,
                                                og_x_offset + first_x_offset, 
                                                og_y_offset + first_y_offset,
                                                device=self._device)
        
        
        curr_motion_id = 0
        for i in range(num_terrains_x):
            for j in range(num_terrains_y):
                if curr_motion_id > num_motions - 1:
                    break
                curr_terrain = self._motion_lib._terrains[curr_motion_id]

                start_x_ind = dim_x_per_terrain * i
                start_y_ind = dim_y_per_terrain * j
                self._terrain.hf[start_x_ind:start_x_ind+curr_terrain.dims[0], start_y_ind:start_y_ind+curr_terrain.dims[1]] = curr_terrain.hf[...]
                self._terrain.hf_mask[start_x_ind:start_x_ind+curr_terrain.dims[0], start_y_ind:start_y_ind+curr_terrain.dims[1]] = curr_terrain.hf_mask[...]

                curr_motion_id += 1
        
        # start_x_ind = 0
        # for i in range(num_motions):
        #     curr_terrain = self._motion_lib._terrains[i]
        #     start_y_ind = 0
        #     for j in range(self._terrains_per_motion):
        #         self._terrain.hf[start_x_ind:start_x_ind+curr_terrain.dims[0], start_y_ind:start_y_ind+curr_terrain.dims[1]] = curr_terrain.hf[...]
        #         self._terrain.hf_mask[start_x_ind:start_x_ind+curr_terrain.dims[0], start_y_ind:start_y_ind+curr_terrain.dims[1]] = curr_terrain.hf_mask[...]
        #         start_y_ind += curr_terrain.dims[1] + 2 * num_padding_cells
        #     start_x_ind += curr_terrain.dims[0] + 2 * num_padding_cells

        save_data = {
            "terrain": self._terrain,
            "terrains_per_motion": self._terrains_per_motion,
            "motion_offsets": self._dm_motion_offsets,
            "all_terrain_verts": all_terrain_verts,
            "all_terrain_tris": all_terrain_tris
        }

        print("saving terrain data to", terrain_save_path)
        with open(terrain_save_path, "wb") as f:
            pickle.dump(save_data, f)

        return all_terrain_verts, all_terrain_tris
    
    def build_terrain_cube(self, env_config, terrain_save_path):

        return

    def build_terrain_wide(self, env_config, terrain_save_path, x_offset = 0.0, y_offset = 0.0):
        # Store a global terrain here
        hmap_config = env_config["dm"]["heightmap"]
        dx = float(hmap_config["horizontal_scale"])
        dy = dx
        padding = hmap_config["padding"]
        num_padding_cells = padding / dx
        assert (round(num_padding_cells) - num_padding_cells) < 1e-5
        num_padding_cells = int(num_padding_cells)


        # TODO: set up some repeating terrain code so motion ids are unique
        # something like mid = i % num_motions
        # and loop over range(num_terrains)
        # Make reset_ref_motion first select a random motion id, then select a random terrain corresponding to that id
        num_motions = self._motion_lib.num_motions()

        # To help reset agents into their respective terrain
        self._dm_motion_offsets = torch.zeros(size=[self._motion_lib.num_motions(), self._terrains_per_motion, 2], dtype=torch.float32, device=self._device)

        global_terrain_min_point = torch.zeros(size=[2], dtype=torch.float32, device=self._device)
        global_terrain_max_point = global_terrain_min_point.clone()

        all_terrain_verts = []
        all_terrain_tris = []

        og_y_offset = y_offset

        for i in range(num_motions):
            y_offset = og_y_offset
            curr_motion_all_terrain_verts = []
            curr_motion_all_terrain_tris = []
            for j in range(self._terrains_per_motion):
                print("building mesh", j, "for motion id:", i)
                curr_terrain = self._motion_lib._terrains[i]

                # TODO: option to randomize terrain here?

                #error_msg = str(curr_terrain.dxdy.cpu().numpy()) + ", " + str(dx) 
                #assert curr_terrain.dxdy[0].item() == curr_terrain.dxdy[1].item() == dx, error_msg
                np_terrain = curr_terrain.numpy_copy()
                
                # adjust terrain position so we can stack them along the x-axis
                self._dm_motion_offsets[i, j, 0] += x_offset - np_terrain.min_point[0]
                self._dm_motion_offsets[i, j, 1] += y_offset - np_terrain.min_point[1]
                np_terrain.min_point[0] = x_offset
                np_terrain.min_point[1] = y_offset

                y_offset += dy * np_terrain.dims[1] + padding * 2.0


                # todo: keep track of corner vert ids so we can make cheap padding meshes
                terrain_is_flat = abs(np_terrain.hf.max() - np_terrain.hf.min()) < 1e-5
                if terrain_is_flat:
                    # p2 ------- p3
                    # |           |
                    # |           |
                    # |           |
                    # p0 ------- p1
                    min_x = np_terrain.min_point[0]
                    min_y = np_terrain.min_point[1]
                    max_x = min_x + dx*(np_terrain.dims[0]-1)
                    max_y = min_y + dx*(np_terrain.dims[1]-1)
                    z = np_terrain.hf[0, 0]
                    p0 = np.array([min_x - dx/2 - padding, min_y - dx/2 - padding, z])
                    p1 = np.array([max_x + dx/2 + padding, min_y - dx/2 - padding, z])
                    p2 = np.array([min_x - dx/2 - padding, max_y + dx/2 + padding, z])
                    p3 = np.array([max_x + dx/2 + padding, max_y + dx/2 + padding, z])
                    verts = np.array([p0, p1, p2, p3], dtype=np.float32)
                    tris = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.uint32)
                else:
                    verts, tris = terrain_util.convert_heightfield_to_voxelized_trimesh(
                        hf = np_terrain.hf,
                        min_x = np_terrain.min_point[0],
                        min_y = np_terrain.min_point[1],
                        dx = np_terrain.dxdy[0],
                        padding=padding)
                    
                curr_motion_all_terrain_verts.append(verts)
                curr_motion_all_terrain_tris.append(tris)

                global_terrain_min_point[0] = min(global_terrain_min_point[0].item(), float(np_terrain.min_point[0]))
                global_terrain_min_point[1] = min(global_terrain_min_point[1].item(), float(np_terrain.min_point[1]))

                curr_terrain_max_point = np_terrain.min_point + np_terrain.dims * dx
                global_terrain_max_point[0] = max(global_terrain_max_point[0].item(), float(curr_terrain_max_point[0]))
                global_terrain_max_point[1] = max(global_terrain_max_point[1].item(), float(curr_terrain_max_point[1]))

            x_offset += dx * np_terrain.dims[0] + padding * 2.0
            all_terrain_verts.append(curr_motion_all_terrain_verts)
            all_terrain_tris.append(curr_motion_all_terrain_tris)  

        new_terrain_dims = torch.round((global_terrain_max_point - global_terrain_min_point) / dx).to(dtype=torch.int)

        print("global terrain dims:", new_terrain_dims)
        print("global terrain min point:", global_terrain_min_point.cpu().numpy())
        print("global terrain max point:", global_terrain_max_point.cpu().numpy())

        self._terrain = terrain_util.SubTerrain("heightmap", new_terrain_dims[0].item(), 
                                                new_terrain_dims[1].item(), dx, dx,
                                                global_terrain_min_point[0].item(), 
                                                global_terrain_min_point[1].item(),
                                                device=self._device)

        start_x_ind = 0
        for i in range(num_motions):
            curr_terrain = self._motion_lib._terrains[i]
            start_y_ind = 0
            for j in range(self._terrains_per_motion):
                self._terrain.hf[start_x_ind:start_x_ind+curr_terrain.dims[0], start_y_ind:start_y_ind+curr_terrain.dims[1]] = curr_terrain.hf[...]
                self._terrain.hf_mask[start_x_ind:start_x_ind+curr_terrain.dims[0], start_y_ind:start_y_ind+curr_terrain.dims[1]] = curr_terrain.hf_mask[...]
                start_y_ind += curr_terrain.dims[1] + 2 * num_padding_cells
            start_x_ind += curr_terrain.dims[0] + 2 * num_padding_cells

        cpu_terrain = self._terrain.torch_copy()
        cpu_terrain.set_device("cpu")
        cpu_dm_motion_offsets = self._dm_motion_offsets.cpu()
        save_data = {
            "terrain": cpu_terrain, # TODO: save with cpu device
            "terrains_per_motion": self._terrains_per_motion,
            "motion_offsets": cpu_dm_motion_offsets,
            "all_terrain_verts": all_terrain_verts,
            "all_terrain_tris": all_terrain_tris
        }

        print("saving terrain data to", terrain_save_path)
        with open(terrain_save_path, "wb") as f:
            pickle.dump(save_data, f)

        return all_terrain_verts, all_terrain_tris
    
    def load_terrain(self, terrain_save_path):
        with open(terrain_save_path, "rb") as f:
            print("loading:", terrain_save_path)
            save_data = pickle.load(f)

        self._terrain = save_data["terrain"]
        self._terrain.set_device(self._device)
        self._terrains_per_motion = save_data["terrains_per_motion"]
        self._dm_motion_offsets = save_data["motion_offsets"]
        self._dm_motion_offsets = self._dm_motion_offsets.to(device=self._device)

        all_terrain_verts = save_data["all_terrain_verts"]
        all_terrain_tris = save_data["all_terrain_tris"]

        return all_terrain_verts, all_terrain_tris
    
    def _refresh_obs_hfs(self, char_root_pos_xyz, char_heading):
        self._refresh_ray_obs_hfs(char_root_pos_xyz, char_heading)
        return
    
    def _refresh_obs_hfs_bc(self, char_root_pos_xyz, char_heading):
        self._refresh_ray_obs_hfs_bc(char_root_pos_xyz, char_heading)
        return

    def _reset_ref_motion(self, env_ids):
        #env_ids = self._extract_dm_env_ids(env_ids)
        # shouldnt need this^

        n = len(env_ids)

        if self._demo_mode:
            motion_ids = env_ids % self._motion_lib.num_motions()
        elif self._one_motion_mode:
            motion_ids = torch.ones_like(env_ids)
            motion_ids[...] = self._selected_motion_id
        elif self._ignore_fail_rates:
            motion_ids = self._motion_lib.sample_motions(n)
        else:
            # motions that fail more often get more samples
            weights_modifier = torch.clamp(self._motion_id_fail_rates, min = self._min_motion_weight) # TODO: try 0.01
            motion_weights = weights_modifier * self._motion_lib._motion_weights
            motion_ids = self._motion_lib.sample_motions(n, motion_weights)

        # if self.has_user_input_motion_id():
        #     motion_ids = torch.ones(size=(n,), device=self._device, dtype=torch.int64)
        #     motion_ids[:] = self.get_user_input_motion_id()

        motion_terrain_ids = torch.randint(high=self._terrains_per_motion, size=(n,), dtype=torch.int64, device=self._device)

        if (self._rand_reset):
            motion_times = self._motion_lib.sample_time(motion_ids)
        else:
            motion_lengths = self._motion_lib.get_motion_length(motion_ids)
            motion_times = motion_lengths * self._motion_start_time_fraction[env_ids]

        self._motion_ids[env_ids] = motion_ids
        self._motion_terrain_ids[env_ids] = motion_terrain_ids
        self._motion_time_offsets[env_ids] = motion_times

        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, contacts = self._motion_lib.calc_motion_frame(motion_ids, motion_times)
        root_pos[..., 0:2] = self._move_to_motion_terrain(root_pos[..., 0:2], env_ids)
        
        self._ref_contacts[env_ids] = contacts        
        self._ref_root_pos[env_ids] = root_pos
        self._ref_root_rot[env_ids] = root_rot
        self._ref_root_vel[env_ids] = root_vel
        self._ref_root_ang_vel[env_ids] = root_ang_vel
        self._ref_joint_rot[env_ids] = joint_rot
        self._ref_dof_vel[env_ids] = dof_vel
        
        dof_pos = self._motion_lib.joint_rot_to_dof(joint_rot)
        self._ref_dof_pos[env_ids] = dof_pos

        self._next_target_xy_time[env_ids] = 0.0
        self._update_motion_targets(env_ids)
        return
    
    def _update_ref_motion(self):
        motion_ids = self._motion_ids
        motion_times = self._get_motion_times()
        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, contacts = \
            self._motion_lib.calc_motion_frame(motion_ids, motion_times)
        root_pos[..., 0:2] = self._move_to_motion_terrain(root_pos[..., 0:2])
        self._ref_contacts[:] = contacts
        self._ref_root_pos[:] = root_pos
        self._ref_root_rot[:] = root_rot
        self._ref_root_vel[:] = root_vel
        self._ref_root_ang_vel[:] = root_ang_vel
        self._ref_joint_rot[:] = joint_rot
        self._ref_dof_vel[:] = dof_vel

        if (self._has_key_bodies()):
            ref_body_pos, _ = self._kin_char_model.forward_kinematics(self._ref_root_pos, 
                                                                      self._ref_root_rot,
                                                                      self._ref_joint_rot)
            self._ref_body_pos[:] = ref_body_pos

        dof_pos = self._motion_lib.joint_rot_to_dof(joint_rot)
        self._ref_dof_pos[:] = dof_pos

        if not self._visualize:
            self._update_motion_targets()
        return

    def _get_motion_times(self, env_ids=None):
        if (env_ids is None):
            motion_times = self._time_buf + self._motion_time_offsets
        else:
            motion_times = self._time_buf[env_ids] + self._motion_time_offsets[env_ids]
        return motion_times
    
    def _move_to_motion_terrain(self, pos, env_ids=None):
        if env_ids is None:
            offset = self._dm_motion_offsets[self._motion_ids, self._motion_terrain_ids] - self._env_offsets[..., 0:2]
        else:
            offset = self._dm_motion_offsets[self._motion_ids[env_ids], self._motion_terrain_ids[env_ids]] - self._env_offsets[env_ids, 0:2]

        shape_len_diff = len(pos.shape) - len(offset.shape)
        assert pos.shape[0] == offset.shape[0], str(pos.shape[0]) + " " + str(offset.shape[0])
        assert pos.shape[-1] == offset.shape[-1]
        for _ in range(shape_len_diff):
            offset = offset.unsqueeze(1)
        return pos + offset

    def _update_motion_targets(self, env_ids=None):
        # FOR DEEPMIMIC ENVS:
        # pick a random future timestep to extract the target position from for each env
        # keep a timestep timer for each env
        # when the envs reach that future timestep, then we reselect the target position
        
        # First filter for envs that can pick a new target

        if env_ids is None:
            env_ids = torch.arange(end=self._num_envs, dtype=torch.int64, device=self._device)

        env_ids = env_ids[self._time_buf[env_ids] >= self._next_target_xy_time[env_ids]]
        if len(env_ids) == 0:
            return

        motion_ids = self._motion_ids[env_ids]
        future_time_offset = torch.rand(size=(len(env_ids),), dtype=torch.float32, device=self._device)
        future_time_offset = future_time_offset * (self._target_xy_future_time_max - self._target_xy_future_time_min) + self._target_xy_future_time_min
        motion_times = self._get_motion_times()[env_ids] + future_time_offset

        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, contacts = self._motion_lib.calc_motion_frame(motion_ids, motion_times)
        root_pos[..., 0:2] = self._move_to_motion_terrain(root_pos[..., 0:2], env_ids)

        # check if the motion time is greater than the motion clip length:
        residual_times = motion_times - self._motion_lib._motion_lengths[motion_ids]
        
        # if so, then extrapolate the future time based on final root position and root velocity
        target_pos = root_pos #+ root_vel * torch.clamp_min(residual_times, torch.zeros_like(residual_times)).unsqueeze(-1)

        # also add some noise
        # TODO: make this a config param
        target_pos_xy = target_pos[..., 0:2]

        target_pos_xy = target_pos_xy + torch.randn_like(target_pos_xy) * 0.05

        self._target_xy[env_ids] = target_pos_xy
        self._next_target_xy_time[env_ids] = self._time_buf[env_ids] + future_time_offset
        return
    
    def reset(self, env_ids):
        if (len(env_ids) > 0):
            self._reset_ref_motion(env_ids)
            # TODO: based on a self._ variable, use MDM to generate start frame
            # Too much work^
            self._char_state_init_from_ref(env_ids)
            if self._record_motion_frame_hist:
                self._motion_frame_hist.clear()
            self.add_noise_to_char_state(env_ids)
            # TODO: pick a random point on plane. Move character up by init height + hf val at point
            if self._random_reset_pos: # TODO make this work with multiple envs
                x = torch.randint(2, self._terrain.dims[0].item()-2, [1], device=self._device)
                y = torch.randint(2, self._terrain.dims[1].item()-2, [1], device=self._device)
                grid_cell_xyz = self._terrain.get_xyz_point(torch.cat([x, y]))
                
                self._char_root_pos[0, 0:2] = grid_cell_xyz[0:2]
                self._char_root_pos[0, 2] = self._char_root_pos[0, 2] + grid_cell_xyz[2]
                self._target_xy[0] = self._char_root_pos[0, 0:2].clone()# + torch.randn(size=[2], dtype=torch.float32, device=self._device)
                self._target_xy[0, 0] += 2.0
                # NOTE: do i need to update the obs?

            self.apply_offsets_to_char_state(env_ids)
            
            self._timestep_buf[env_ids] = 0
            self._time_buf[env_ids] = 0.0
            self._done_buf[env_ids] = base_env.DoneFlags.NULL.value

            self._actors_need_reset[env_ids, SIM_CHAR_IDX] = True
        return
        
    def compute_tar_obs(self, tar_obs_steps, env_ids = None):
        if env_ids is not None:
            motion_ids = self._motion_ids[env_ids]
            motion_times = self._get_motion_times(env_ids)#self._time_buf[env_ids]
            #char_root_pos = self._char_root_pos[env_ids]
        else:
            motion_ids = self._motion_ids
            motion_times = self._get_motion_times()#self._time_buf
            #char_root_pos = self._char_root_pos
        tar_root_pos, tar_root_rot, tar_joint_rot, tar_contacts = mgdm_dm_util.fetch_tar_obs_data(motion_ids, 
                                                                                                  motion_times,
                                                                                                  self._motion_lib,
                                                                                                  self._timestep,
                                                                                                  tar_obs_steps)
        
        # tar root pos from motion lib does not keep track of its global position
        tar_root_pos[..., 0:2] = self._move_to_motion_terrain(tar_root_pos[..., 0:2], env_ids)# + char_root_pos[..., 0:2].unsqueeze(1)
        tar_root_pos_flat = torch.reshape(tar_root_pos, [tar_root_pos.shape[0] * tar_root_pos.shape[1], 
                                                            tar_root_pos.shape[-1]])
        tar_root_rot_flat = torch.reshape(tar_root_rot, [tar_root_rot.shape[0] * tar_root_rot.shape[1], 
                                                            tar_root_rot.shape[-1]])
        tar_joint_rot_flat = torch.reshape(tar_joint_rot, [tar_joint_rot.shape[0] * tar_joint_rot.shape[1], 
                                                            tar_joint_rot.shape[-2], tar_joint_rot.shape[-1]])
        tar_body_pos_flat, _ = self._kin_char_model.forward_kinematics(tar_root_pos_flat, tar_root_rot_flat,
                                                                        tar_joint_rot_flat)
        tar_body_pos = torch.reshape(tar_body_pos_flat, [tar_root_pos.shape[0], tar_root_pos.shape[1], 
                                                            tar_body_pos_flat.shape[-2], tar_body_pos_flat.shape[-1]])

        if (self._has_key_bodies()):
            tar_key_pos = tar_body_pos[..., self._key_body_ids, :]
        else:
            tar_key_pos = torch.zeros([0], device=self._device)
        return tar_root_pos, tar_root_rot, tar_joint_rot, tar_key_pos, tar_contacts
    
    def update_done(self, termination_height, episode_length, contact_body_ids, 
                    pose_termination, pose_termination_dist, global_obs, enable_early_termination,
                    track_root, root_pos_termination_dist, root_rot_termination_angle):
        super().update_done(termination_height, episode_length, contact_body_ids, 
                    pose_termination, pose_termination_dist, global_obs, enable_early_termination,
                    track_root, root_pos_termination_dist, root_rot_termination_angle)
        
        # before updating done buf based on motion time, check for fail, since there is a hack
        # where we set doneflag to fail for motion ends

        #if False:
        # get the envs that failed
        # failed = self._done_buf == base_env.DoneFlags.FAIL.value

        # # get the motion ids of the failed envs
        # failed_motion_ids, fail_counts = torch.unique(self._motion_ids[failed], return_counts = True)

        # # self._motion_id_fail_counts[failed_motion_ids] += fail_counts

        # for id in failed_motion_ids:
        #     for _ in range(fail_counts[id].item()):
        #         self._motion_id_fail_rates[id] *= (1.0 - self._ema_weight) + self._ema_weight * 1.0

        #if self._num_envs == 1:
        #    print(base_env.DoneFlags(self._done_buf.item()).name)

        motion_times = self._get_motion_times()
        motion_len = self._motion_lib.get_motion_length(self._motion_ids)
        motion_loop_mode = self._motion_lib.get_motion_loop_mode(self._motion_ids)
        motion_len_term = motion_loop_mode != motion_lib.LoopMode.WRAP.value
        motion_end = motion_times >= motion_len
        motion_end = torch.logical_and(motion_end, motion_len_term)
        #print("done_buf:", base_env.DoneFlags(self._done_buf[0].item()).name)
        #print("motion_end:", motion_end[0].item())
        #print("motion times:", motion_times[0].item())
        #print("motion_len:", motion_len[0].item())


        done = torch.logical_or(self._done_buf != base_env.DoneFlags.NULL.value, motion_end)
        done_ids = torch.nonzero(done).squeeze(0)

        # potentially a very slow for loop since we are interacting with gpu on cpu
        for env_id in done_ids:
            done_flag = self._done_buf[env_id]
            motion_id = self._motion_ids[env_id]
            if done_flag == base_env.DoneFlags.FAIL.value:
                self._motion_id_fail_rates[motion_id] = self._motion_id_fail_rates[motion_id] * (1.0 - self._ema_weight) + self._ema_weight
            elif done_flag == base_env.DoneFlags.TIME.value \
                or done_flag == base_env.DoneFlags.SUCC.value \
                or motion_end[env_id] == True:
                self._motion_id_fail_rates[motion_id] = self._motion_id_fail_rates[motion_id] * (1.0 - self._ema_weight)
            else:
                assert False



        # If the motion failed, then we don't end
        #self._motion_id_done_counts[done_motion_ids] += done_counts
        #self._motion_id_fail_rates[done_motion_ids]  

        # setting the done flag to fail at the end of the motion avoids the
        # local minimal of a character just standing still until the end of the motion
        self._done_buf[motion_end] = base_env.DoneFlags.FAIL.value
        return
    

    def get_extra_log_info(self):
        motion_names = self._motion_lib.get_motion_names()
        k = min(40, self._motion_lib.num_motions())

        #fail_rates = self._motion_id_fail_counts / self._motion_id_done_counts
        #self._motion_id_fail_rates[:] = fail_rates

        #top_fail_rates, motion_ids = torch.topk(fail_rates, k=k)

        #top_fail_rates, motion_ids = torch.sort(fail_rates, descending=True)
        top_fail_rates, motion_ids = torch.sort(self._motion_id_fail_rates, descending=True)
        
        title_str = "***** TOP " + str(k) + " FAILURE RATES BY MOTION ID *****"
        print(title_str)
        for i in range(k):
            motion_name = motion_names[motion_ids[i].item()]
            fail_rate = str(top_fail_rates[i].item() * 100.0)
            #fail_count = str(self._motion_id_fail_counts[motion_ids[i].item()].item())
            #done_count = str(self._motion_id_done_counts[motion_ids[i].item()].item())
            print("motion name: " + motion_name)
            print("motion_id: " + str(motion_ids[i].item()) +
                  ", fail rate (EMA): " + fail_rate + "%")
                  #", fail count: " + fail_count +
                  #", done count: " + done_count)
            print("__________________________________________________")
        print("*"*len(title_str))


        # Check for the 25, 50, and 75% fail rate percentiles
        
        fail_rate_at_quantiles = torch.quantile(top_fail_rates, self._fail_rate_quantiles)


        fail_rate_dict = dict()
        for i in range(len(motion_names)):
            fail_rate_dict[motion_names[i]] = self._motion_id_fail_rates[i].item() * 100.0

        extra_log_info = dict()
        extra_log_info["MOTION_FAIL_RATES"] = fail_rate_dict
        extra_log_info["Misc"] = {"top fail rate": top_fail_rates[0].item() * 100.0}
        
        for i in range(self._fail_rate_quantiles.shape[0]):
            info_name = "Fail Rate at " + str(round(self._fail_rate_quantiles[i].item() * 100.0)) + "% Quantile"
            print(info_name)
            print(fail_rate_at_quantiles[i].item() * 100.0)
            extra_log_info["Misc"][info_name] = fail_rate_at_quantiles[i].item() * 100.0


        # Also log the mean fail rate for each class:
        if self._has_motion_classes:
            for class_name in self._motion_classes:
                mean_fail_rate = torch.mean(self._motion_id_fail_rates[self._motion_class_to_ids_map[class_name]]).item() * 100.0
                max_fail_rate = torch.max(self._motion_id_fail_rates[self._motion_class_to_ids_map[class_name]]).item() * 100.0

                info_name = class_name + " mean fail rate"
                extra_log_info["Misc"][info_name] = mean_fail_rate

                info_name = class_name + " max fail rate"
                extra_log_info["Misc"][info_name] = max_fail_rate
        return extra_log_info
    
    def post_test_update(self):
        # set all to 1 because we want the default failure rate to be 100%,
        # and we divide by done counts so we want that to be > 0
        # self._motion_id_done_counts[:] = 1
        # self._motion_id_fail_counts[:] = 1
        # using EMA for failure rates now
        return
    
    def set_demo_mode(self, val=None):
        if val is None:
            val = not self._demo_mode
        self._demo_mode = val
        return self._demo_mode
    
    def set_motion_start_time_fraction(self, val: torch.Tensor):
        self._motion_start_time_fraction = val
        return
    
    def get_env_motion_length(self, env_ids):
        motion_ids = self._motion_ids[env_ids]
        return self._motion_lib.get_motion_length(motion_ids)
    
    def get_env_motion_time(self, env_ids):
        return self._get_motion_times(env_ids)
    
    def get_env_motion_name(self, env_id):
        motion_names = self._motion_lib.get_motion_names()
        motion_id = self._motion_ids[env_id].item()
        return motion_names[motion_id]
    
    # TODO
    # def disable_motion_ids(self, motion_ids):


    #     return

    def _get_char_motion_frames(self, eps=1e-5, ref=False, null=False):
        
        if null:
            ret = motion_util.MotionFrames()
            ret.init_blank_frames(self._kin_char_model, 1, self._num_envs)
            ret.body_pos = None
            ret.body_rot = None
        else:
            if ref:
                root_pos = self._ref_root_pos
                root_rot = self._ref_root_rot
                joint_rot = self._kin_char_model.dof_to_rot(self._ref_dof_pos)
                char_contacts = self._ref_contacts
            else:
                root_pos = self._char_root_pos
                root_rot = self._char_root_rot
                joint_rot = self._kin_char_model.dof_to_rot(self._char_dof_pos)
                char_contacts = torch.norm(self._char_contact_forces, dim=-1)
                char_contacts = char_contacts > eps
                char_contacts = char_contacts.to(dtype=torch.float32)

            ret = motion_util.MotionFrames(
                root_pos = root_pos.clone(),
                root_rot = root_rot.clone(),
                joint_rot = joint_rot.clone(),
                contacts = char_contacts.clone()
            )

            ret = ret.unsqueeze(1)
        
        return ret

    def astar_mdm(self):

        # 1. take a slice of the terrain around current char position to target position
        # 2. run a_star
        # 3. run mdm_path
        # 4. replace motion lib with mdm generated motion? (or make new motion lib?)
        # 4.5 save mdm generated motion so we can view whole thing in visualizer
        env_id = 0
        start_pos_xyz = self._char_root_pos[env_id]
        start_pos_xy = start_pos_xyz[0:2]

        end_pos_xy = self._target_xy[env_id]

        start_node = self._terrain.get_grid_index(start_pos_xy)
        end_node = self._terrain.get_grid_index(end_pos_xy)


        slice_pad = 10
        

        min_i = min(start_node[0].item(), end_node[0].item())
        min_j = min(start_node[1].item(), end_node[1].item())
        max_i = max(start_node[0].item(), end_node[0].item())
        max_j = max(start_node[1].item(), end_node[1].item())

        slice_start_i = max(min_i - slice_pad, 0)
        slice_start_j = max(min_j - slice_pad, 0)

        slice_end_i = min(max_i + slice_pad, self._terrain.hf.shape[0]-1)
        slice_end_j = min(max_j + slice_pad, self._terrain.hf.shape[1]-1)


        # TODO: fix slicing padding logic so that it actually pads

        terrain_slice = terrain_util.get_terrain_slice(self._terrain, 
                                                   slice_start_i, slice_start_j,
                                                   slice_end_i, slice_end_j)
        
        sliced_start_node_i = start_node[0].item() - slice_start_i
        sliced_start_node_j = start_node[1].item() - slice_start_j
        sliced_end_node_i = end_node[0].item() - start_node[0].item() + sliced_start_node_i
        sliced_end_node_j = end_node[1].item() - start_node[1].item() + sliced_start_node_j

        # TODO: check if we can switch to cpu and if its faster
        device=self._device
        sliced_start_node = torch.tensor([sliced_start_node_i, sliced_start_node_j], dtype=torch.int64, device=device)
        sliced_end_node = torch.tensor([sliced_end_node_i, sliced_end_node_j], dtype=torch.int64, device=device)

        assert sliced_end_node_i < terrain_slice.hf.shape[0]
        assert sliced_end_node_j < terrain_slice.hf.shape[1]

        print("Running ASTAR...")
        astar_settings = astar.AStarSettings()
        path_nodes = astar.run_a_star_on_start_end_nodes(terrain_slice, sliced_start_node, sliced_end_node, astar_settings)
        self._path_nodes = path_nodes
        # TODO: also save current and previous frame
        # TODO: make dm env optionally save previous frame with config bool

        
        prev_frames = motion_util.cat_motion_frames(self._motion_frame_hist[-2:]).squeeze(0)

        # TODO
        #mdm_path.generate_frames_until_end_of_path()

        mdm_path_settings = mdm_path.MDMPathSettings()
        mdm_path_settings.mdm_batch_size = 4
        mdm_path_settings.top_k = 1
        mdm_path_settings.max_motion_length = 20.0

        mdm_gen_settings = gen_util.MDMGenSettings()
        mdm_gen_settings.ddim_stride = 10
        mdm_gen_settings.use_cfg = True
        mdm_gen_settings.use_prev_state = torch.zeros(size=[mdm_path_settings.mdm_batch_size], dtype=torch.bool, device=self._device)
        mdm_gen_settings.prev_state_ind_key= torch.zeros(size=[mdm_path_settings.mdm_batch_size], dtype=torch.bool, device=self._device)

        

        gen_motion_frames, _, info = mdm_path.generate_frames_until_end_of_path(
            path_nodes=path_nodes,
            terrain=terrain_slice,
            char_model=self._kin_char_model,
            mdm_model=self._mdm_model,
            prev_frames=prev_frames,
            mdm_path_settings=mdm_path_settings,
            mdm_gen_settings=mdm_gen_settings,
            add_noise_to_loss=False,
            slice_terrain=False)

        motion_frames, contact_frames = gen_motion_frames[0].get_mlib_format(self._kin_char_model)

        self._motion_lib = motion_lib.MotionLib(motion_input=motion_frames,
                                                kin_char_model=self._kin_char_model,
                                                device=self._device,
                                                init_type="motion_frames",
                                                loop_mode=motion_lib.LoopMode.CLAMP,
                                                fps=30,
                                                contact_info=True,
                                                contacts=contact_frames)
        self._motion_lib._motion_names = ["astar_mdm_motion"]
        
        self._motion_ids[:] = 0
        self._motion_time_offsets[:] = 0.0
        self._one_motion_mode = True
        self._dm_motion_offsets[0][:] = 0.0

        kwargs = {"path_nodes": path_nodes}
        terrain_save_path = "output/ig_env_astar_terrain.pkl"
        medit_lib.save_motion_data(terrain_save_path,
                                   motion_frames,
                                   contact_frames,
                                   terrain_slice,
                                   30,
                                   "CLAMP",
                                   **kwargs)

        return
    
    def pre_physics_step(self):
        if self._record_motion_frame_hist:
            assert self._num_envs == 1
            # only for env id 0
            curr_frame = self._get_char_motion_frames()

            if len(self._motion_frame_hist) < 5:
                self._motion_frame_hist.append(curr_frame)
            else:
                self._motion_frame_hist.pop(0)
                self._motion_frame_hist.append(curr_frame)

        return