import abc
import gym
import isaacgym.gymapi as gymapi
import isaacgym.gymtorch as gymtorch
import isaacgym.gymutil as gymutil
import numpy as np
import re
import sys
import torch
import time

import envs.base_env as base_env

from util.logger import Logger
import util.torch_util as torch_util
import sys
import util.terrain_util as terrain_util

class IGEnv(base_env.BaseEnv):
    NAME = "isaacgym"

    def __init__(self, config, num_envs, device, visualize):
        super().__init__(visualize=visualize)
        
        self._viewer = None
        self._num_envs = num_envs
        self._device = device
        self._enable_viewer_sync = True
        self._config = config

        env_config = config["env"]
        self._episode_length = env_config["episode_length"] # episode length in seconds
        print("Episode length:", self._episode_length)
        self._env_spacing = 5 if "env_spacing" not in env_config else env_config["env_spacing"]
        self._env_style = env_config.get("env_style", "square")
        self._build_sim(config)
        self._build_envs(config)
        self._gym.prepare_sim(self._sim)
        
        self._build_sim_tensors(config)
        self._build_data_buffers()

        self._action_space = self._build_action_space()

        if (self._visualize):
            self._build_viewer()
            self._init_camera()
            
        return
    
    def get_num_envs(self):
        return self._num_envs

    def reset(self, env_ids=None):
        if (env_ids is None): # note, not the same as passing in empty tensor []
            num_envs = self.get_num_envs()
            env_ids = torch.arange(num_envs, device=self._device, dtype=torch.long)

        self._reset_envs(env_ids)
        
        reset_env_ids = self._reset_sim_tensors()
        self._refresh_sim_tensors()
        self._update_observations(reset_env_ids)
        self._update_info(reset_env_ids)

        return self._obs_buf, self._info
    
    def step(self, action):
        # apply actions
        self._pre_physics_step(action)

        # step physics and render each frame
        self._physics_step()
        
        # to fix!
        if (self._device == "cpu"):
            self._gym.fetch_results(self._sim, True)
            
        if (self._viewer):
            self._update_camera()
            self._render()
        
        # compute observations, rewards, resets, ...
        self._post_physics_step()
        
        return self._obs_buf, self._reward_buf, self._done_buf, self._info
    
    def get_obs_space(self):
        obs = self._compute_obs()
        obs_shape = list(obs.shape[1:])
        obs_dtype = torch_util.torch_dtype_to_numpy(obs.dtype)
        obs_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_shape,
            dtype=obs_dtype,
        )
        return obs_space
    
    def _build_sim(self, config):
        self._gym = gymapi.acquire_gym()
        self._pysics_engine = gymapi.SimType.SIM_PHYSX
        
        env_config = config["env"]
        sim_freq = env_config.get("sim_freq", 60)
        control_freq = env_config.get("control_freq", 10)
        self._control_freq = control_freq
        assert(sim_freq >= control_freq and sim_freq % control_freq == 0), \
            "Simulation frequency must be a multiple of the control frequency"
        sim_timestep = 1.0 / sim_freq

        self._timestep = 1.0 / control_freq
        self._sim_steps = int(sim_freq / control_freq)
        self._sim_params = self._parse_sim_params(config, sim_timestep)

        device_idx = self._get_device_idx()
        self._sim = self._gym.create_sim(device_idx, device_idx, self._pysics_engine, self._sim_params)
        assert(self._sim is not None), "Failed to create sim"

        return

    def _get_device_idx(self):
        re_idx = re.search(r"\d", self._device)
        if (re_idx is None):
            device_idx = 0
        else:
            num_idx = re_idx.start()
            device_idx = int(self._device[num_idx:])
        return device_idx

    def _parse_sim_params(self, config, sim_timestep):
        sim_params = gymapi.SimParams()
        sim_params.dt = sim_timestep
        sim_params.num_client_threads = 0
        
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity.x = 0
        sim_params.gravity.y = 0
        if "gravity_z" in config["env"]:
            sim_params.gravity.z = config["env"]["gravity_z"]
        else:
            sim_params.gravity.z = -9.81

        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.num_threads = 4
        sim_params.physx.num_subscenes = 0
        sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024

        if ("gpu" in self._device or "cuda" in self._device):
            sim_params.physx.use_gpu = True
            sim_params.use_gpu_pipeline = True
        elif ("cpu" in self._device):
            sim_params.physx.use_gpu = False
            sim_params.use_gpu_pipeline = False
        else:
            assert(False), "Unsupported simulation device: {}".format(self._device)

        # if sim options are provided in cfg, parse them and update/override above:
        if "sim" in config:
            gymutil.parse_sim_config(config["sim"], sim_params)
        
        return sim_params
    
    def _build_heightmap(self, config):

        env_config = config["env"]
        hm_config = env_config["heightmap"]

        padding = hm_config["padding"]
        min_x = -padding
        min_y = -padding
        horizontal_scale = hm_config["horizontal_scale"]
        num_padding_cells = int(round(padding / horizontal_scale))

        #x_len = config["env"]["terrain_length"] + padding * 2.0
        #y_len = config["env"]["env_spacing"] * 2.0 * self._num_envs + padding * 2.0

        

        gen_name = hm_config["generator_name"]
        #heightfield = np.zeros((num_rows, num_cols), dtype=np.int16)

        
        # sub_terrain = terrain_util.Terrain()
        # sub_terrain = terrain_util.linear_parkour_course(sub_terrain)

        # vertices, triangles = terrain_util.convert_terrain_to_trimesh(sub_terrain)

        
        # measured by number of heightmap cells
        real_gap_width = 1.15 # must be odd multiple of horizontal_scale
        gap_width = int(round(real_gap_width / horizontal_scale)) - 1 # this -1 is too tedious to explain
        real_gap_spacing = 4.9 # must be multiple of horizontal scale
        gap_spacing = int(round(real_gap_spacing/ horizontal_scale))

        #real_vault_width = 0.35 # must be odd multiple of horizontal_scale
        real_vault_width = 0.25
        vault_width = int(round(real_vault_width / horizontal_scale)) - 1
        #real_vault_spacing = 4.9 # must be multiple of horizontal scale
        #real_vault_spacing = 4.075
        real_vault_spacing = 6.673
        vault_spacing = int(round(real_vault_spacing / horizontal_scale))

        gap_height = -1.0
        vault_height = 1.00
        
        print("building heightmap and mesh")
        start_time = time.perf_counter()
        # TODO: make this more elegant?
        if gen_name == "gap":
            x_len = env_config["env_spacing"] * 2.0 * (self._num_envs-1.0) + padding * 2.0
            print("x_len", x_len)
            y_len = hm_config["terrain_length"] + padding * 2.0

            x_dim = int(x_len/horizontal_scale)
            y_dim = int(y_len/horizontal_scale)
            sub_terrain = terrain_util.SubTerrain(x_dim=x_dim, y_dim=y_dim, dx=horizontal_scale, dy=horizontal_scale,
                                              min_x=min_x, min_y=min_y)
            box_centers = [gap_spacing + num_padding_cells]
            box_heights = [gap_height]
            box_dims = [gap_width]
            new_terrain, vertices, triangles = terrain_util.linear_parkour_course(sub_terrain, box_centers, box_heights, box_dims)
        elif gen_name == "vault":
            x_len = env_config["env_spacing"] * 2.0 * (self._num_envs-1.0) + padding * 2.0
            y_len = hm_config["terrain_length"] + padding * 2.0
            x_dim = int(x_len/horizontal_scale)
            y_dim = int(y_len/horizontal_scale)
            sub_terrain = terrain_util.SubTerrain(x_dim=x_dim, y_dim=y_dim, dx=horizontal_scale, dy=horizontal_scale,
                                              min_x=min_x, min_y=min_y)
            
            real_second_vault_box_center = real_vault_spacing + 6.5
            box_centers = [vault_spacing + num_padding_cells] #, second_vault_box_center + num_padding_cells]
            box_heights = [vault_height] #, vault_height]
            box_dims = [vault_width] #, vault_width]
            new_terrain, vertices, triangles = terrain_util.linear_parkour_course(sub_terrain, box_centers, box_heights, box_dims)
        
            # THIS STUFF ONLY MATTERS FOR TERRAIN RUNNER
            self._spawn_min_x = 0.0
            self._spawn_min_y = 0.0
            self._spawn_max_x = -real_vault_spacing #x_len - padding*2.0 # gonna space envs manually
            self._spawn_max_y = real_second_vault_box_center #real_vault_spacing * 1.1
            print("max y spawn:", self._spawn_max_y)
            print("y_len:", y_len)
        elif gen_name == "vault_gap":
            num_env_per_row = self._num_envs# // 2
            x_len = env_config["env_spacing"] * 2.0 * (num_env_per_row-1.0) + padding * 2.0
            y_len = hm_config["terrain_length"] + padding * 2.0
            x_dim = int(x_len/horizontal_scale)
            y_dim = int(y_len/horizontal_scale)
            sub_terrain = terrain_util.SubTerrain(x_dim=x_dim, y_dim=y_dim, dx=horizontal_scale, dy=horizontal_scale,
                                              min_x=min_x, min_y=min_y)
            
            real_second_box_center = hm_config["vault_gap"]["second_row_y"] + real_gap_spacing
            second_box_center = int(round(real_second_box_center/horizontal_scale))
            box_centers = [vault_spacing + num_padding_cells, second_box_center + num_padding_cells]
            box_heights = [vault_height, gap_height]
            box_dims = [vault_width, gap_width]
            new_terrain, vertices, triangles = terrain_util.linear_parkour_course(sub_terrain, box_centers, box_heights, box_dims)
        
            # THIS STUFF ONLY MATTERS FOR TERRAIN RUNNER
            # self._spawn_min_x = 0.0
            # self._spawn_min_y = 0.0
            # self._spawn_max_x = -real_vault_spacing #x_len - padding*2.0 # gonna space envs manually
            # self._spawn_max_y = real_second_vault_box_center #real_vault_spacing * 1.1
            # print("max y spawn:", self._spawn_max_y)
            # print("y_len:", y_len)
        elif gen_name == "random_linear_parkour_course":
            x_len = hm_config["parkour_course"]["path_width"]
            envs_per_block = hm_config["parkour_course"]["envs_per_block"]
            course_padding = hm_config["parkour_course"]["course_padding"]
            num_blocks = self._num_envs // envs_per_block + course_padding
            approx_len_per_block = hm_config["parkour_course"]["approx_len_per_block"]
            y_len = approx_len_per_block * num_blocks + padding * 2.0
            x_dim = int(x_len/horizontal_scale)
            y_dim = int(y_len/horizontal_scale)
            sub_terrain = terrain_util.SubTerrain(x_dim=x_dim, y_dim=y_dim, dx=horizontal_scale, dy=horizontal_scale,
                                              min_x=min_x, min_y=min_y)
            
            new_terrain, vertices, triangles = terrain_util.random_linear_parkour_course(
                sub_terrain,
                gap_width, gap_height,
                vault_width, vault_height,
                num_padding_cells)
            
            # THIS STUFF ONLY MATTERS FOR TERRAIN RUNNER
            # TODO: move somewhere safer/cleaner
            self._spawn_min_x = 0.0
            self._spawn_min_y = 0.0
            self._spawn_max_x = 0.0#x_len - padding*2.0 # gonna space envs manually
            self._spawn_max_y = approx_len_per_block * (num_blocks - course_padding)
            print("max y spawn:", self._spawn_max_y)
            print("y_len:", y_len)
        elif gen_name == "hf_from_motion":
            # new_terrain = terrain_util.SubTerrain(x_dim=1000, y_dim=100, dx=dx, dy=dx,
            #                                   min_x=-5.0, min_y=-5.0)
            # terrain_util.random_heightfield(new_terrain)
            dx = hm_config["horizontal_scale"]
            motion_file = config["env"]["motion_file"]
            import anim.kin_char_model as kin_char_model
            import pickle
            char_model_file = "data/assets/humanoid_beyond.xml"
            with open(motion_file, "rb") as filestream:
                motion_data = pickle.load(filestream)
                motion_frames = torch.tensor(motion_data["frames"], dtype=torch.float32, device=self._device)
                #motion_contacts = torch.tensor(motion_data["contacts"], dtype=torch.float32, device=self._device)
                #motion_contacts = None
                if "floor_heights" in motion_data:
                    floor_heights = torch.tensor(motion_data["floor_heights"], dtype=torch.float32, device=self._device)
                    floor_heights = floor_heights.unsqueeze(0)
                else:
                    floor_heights = None
            char_model = kin_char_model.KinCharModel(self._device)
            char_model.load_char_file(char_model_file)

            num_neg_x = 250
            num_pos_x = 250
            num_neg_y = 250
            num_pos_y = 250
            new_hf, _ = terrain_util.hf_from_motion(motion_frames.unsqueeze(dim=0),
                                                        min_height=-1.0,
                                                        ground_height=0.0,
                                                        dx=dx,
                                                        num_neg_x=num_neg_x,
                                                        num_pos_x=num_pos_x,
                                                        num_neg_y=num_neg_y,
                                                        num_pos_y=num_pos_y,
                                                        floor_heights=floor_heights)
            
            # 3x3 maxpool filter
            maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

            new_hf = maxpool(new_hf.unsqueeze(dim=0)).squeeze(dim=0)

            
            grid_dim_x = num_neg_x + num_pos_x + 1
            grid_dim_y = num_neg_y + num_pos_y + 1

            min_x = motion_frames[0, 0] - dx * num_neg_x
            min_y = motion_frames[0, 1] - dx * num_neg_y

            new_terrain = terrain_util.SubTerrain(x_dim=grid_dim_x, y_dim=grid_dim_y, dx=dx, dy=dx,
                                                  min_x=min_x.item(), min_y=min_y.item(), device=self._device)
            new_terrain.hf = new_hf.squeeze(dim=0)
            
            np_terrain = new_terrain.numpy_copy()
            vertices, triangles = terrain_util.convert_heightfield_to_voxelized_trimesh(
                np_terrain.hf, min_x = np_terrain.min_point[0],
                min_y = np_terrain.min_point[1],
                dx = dx
            )
        elif gen_name == "saved_terrain":
            # If terrain is saved with the motion file
            motion_file = config["env"]["motion_file"]
            import anim.kin_char_model as kin_char_model
            import pickle
            #char_model_file = "data/assets/humanoid.xml"
            with open(motion_file, "rb") as filestream:
                motion_data = pickle.load(filestream)
                assert "terrain" in motion_data
                terrain = motion_data["terrain"]
                #terrain.hf[~terrain.hf_mask] = 1.5

            if hm_config["saved_terrain"]["augment"]:
                terrain.to_torch(self._device)
                heights=[0.6]
                hf_augmenter = terrain_util.SimpleHFAugmenter(heights=heights, device=self._device)
                terrain.hf = hf_augmenter.forward(terrain.hf, terrain.hf_mask, terrain.hf_maxmin, num_boxes=32)
                terrain.to_numpy()

            if hm_config["saved_terrain"]["repeat"]:
                # NOTE: this only works with x_line env style
                terrain_x_length = terrain.dxdy[0] * terrain.dims[0]
                num_repeats = 2 + int(round(self._num_envs * self._env_spacing * 2 / terrain_x_length))

                terrain.hf = np.repeat(terrain.hf, repeats=num_repeats, axis=0)

            padding_len = hm_config["saved_terrain"]["padding_len"]
            if padding_len > 0:

                terrain.hf = np.pad(terrain.hf, pad_width=padding_len, mode='constant')
                terrain.dims = terrain.hf.shape
                terrain.min_point = terrain.min_point - np.array([padding_len, padding_len]) * terrain.dxdy

            vertices, triangles = terrain_util.convert_heightfield_to_voxelized_trimesh(
                terrain.hf, min_x = terrain.min_point[0],
                min_y = terrain.min_point[1],
                dx = terrain.dxdy[0]
            )

            terrain.to_torch(self._device)
            new_terrain = terrain

        elif gen_name == "paths":
            dx = 0.25
            grid_dim_x = 201
            grid_dim_y = 201
            min_x = -10.0
            min_y = -10.0
            new_terrain = terrain_util.SubTerrain(x_dim=grid_dim_x, y_dim=grid_dim_y, dx=dx, dy=dx,
                                                  min_x=min_x, min_y=min_y, device=self._device)
            
            terrain_util.gen_paths_hf(new_terrain)
            np_terrain = new_terrain.numpy_copy()
            vertices, triangles = terrain_util.convert_heightfield_to_voxelized_trimesh(
                np_terrain.hf, min_x = np_terrain.min_point[0],
                min_y = np_terrain.min_point[1],
                dx = dx
            )

        elif gen_name == "terrain_runner":
            dx = 0.1

            #grid_dim_x = 1000
            #grid_dim_y = 1000
            #min_x = -100.0
            #min_y = -100.0

            sq_m_per_env = hm_config["sq_m_per_env"]
            safety_region = 40.0
            x_length = np.round(np.sqrt(self._num_envs)) * sq_m_per_env + safety_region * 2.0
            y_length = x_length

            grid_dim_x = int(x_length / dx)
            grid_dim_y = int(y_length / dx)

            min_x = -x_length / 2.0
            min_y = -y_length / 2.0

            new_terrain = terrain_util.SubTerrain(x_dim=grid_dim_x, y_dim=grid_dim_y, dx=dx, dy=dx,
                                                  min_x=min_x, min_y=min_y, device=self._device)
            
            boxes_per_sq_m = hm_config["boxes_per_sq_m"]
            num_boxes = int(np.round(x_length * y_length * boxes_per_sq_m))

            # TODO: read these params from file
            # terrain_util.add_boxes_to_hf(hf = new_terrain.hf,
            #                              hf_mask = new_terrain.hf_mask,
            #                              box_heights = [0.6],
            #                              hf_maxmin = new_terrain.hf_maxmin,
            #                              num_boxes = num_boxes,
            #                              box_max_len = 50, 
            #                              box_min_len = 5)

            # quadrants
            num_segments = 8
            x_dim_divided = new_terrain.dims[0].item() // num_segments
            x_dim_remainder = new_terrain.dims[0].item() % num_segments
            segments_x = []
            for i in range(num_segments + 1):
                val = i * x_dim_divided
                if i == num_segments:
                    val += x_dim_remainder
                segments_x.append(val)
            
            #[0, x_dim_divided, x_dim_divided * 2, x_dim_divided * 3, x_dim_divided*4 + x_dim_remainder]

            y_dim_divided = new_terrain.dims[1].item() // num_segments
            y_dim_remainder = new_terrain.dims[1].item() % num_segments
            #segments_y = [0, y_dim_divided, y_dim_divided * 2, y_dim_divided * 3, y_dim_divided*4 + y_dim_remainder]

            segments_y = []
            for i in range(num_segments + 1):
                val = i * y_dim_divided
                if i == num_segments:
                    val += y_dim_remainder
                segments_y.append(val)

            for i in range(num_segments):
                for j in range(num_segments):
                    if i % 2 == 0:
                        if j % 2 == 0:
                            val = 0.6
                        else: 
                            val = 0.0
                    else:
                        if j % 2 == 0:
                            val = 0.0
                        else:
                            val = 0.6
                    new_terrain.hf[segments_x[i]:segments_x[i+1], segments_y[j]:segments_y[j+1]] = val
            
            np_terrain = new_terrain.numpy_copy()
            vertices, triangles = terrain_util.convert_heightfield_to_voxelized_trimesh(
                np_terrain.hf, min_x = np_terrain.min_point[0],
                min_y = np_terrain.min_point[1],
                dx = dx
            )

            self._oob_region = safety_region / 10
            self._spawn_min_x = min_x + safety_region
            self._spawn_min_y = min_y + safety_region
            self._spawn_max_x = min_x + x_length - safety_region #x_len - padding*2.0 # gonna space envs manually
            self._spawn_max_y = min_y + y_length - safety_region #real_vault_spacing * 1.1


        else:
            assert False

        
            
        end_time = time.perf_counter()
        print("building heightmap and mesh time:", end_time-start_time, " seconds.")
        
        print("vertices.shape:", vertices.shape)
        print("triangles.shape:", triangles.shape)

        #vertices, triangles = terrain_util.convert_heightfield_to_trimesh(heightfield, horizontal_scale=horizontal_scale, vertical_scale=vertical_scale, slope_threshold=3.0)
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = vertices.shape[0]
        tm_params.nb_triangles = triangles.shape[0]
        tm_params.transform.p.x = 0.0#sub_terrain.min_point[0]
        tm_params.transform.p.y = 0.0#sub_terrain.min_point[1]
        tm_params.static_friction = env_config["plane"]["static_friction"]
        tm_params.dynamic_friction = env_config["plane"]["dynamic_friction"]
        tm_params.restitution = env_config["plane"]["restitution"]
        print("adding triangle mesh to gym")
        start_time = time.perf_counter()
        self._gym.add_triangle_mesh(self._sim, vertices.flatten(), triangles.flatten(), tm_params)
        end_time = time.perf_counter()
        print("adding mesh to gym time:", end_time-start_time, " seconds.")

        self._terrain = new_terrain
        
        return

    def _build_envs(self, config):
        self._envs = []
        env_config = config["env"]

        terrain_type = env_config.get("terrain", None)

        if terrain_type == "heightmap":
            hm_config = env_config["heightmap"]
            generator_name = hm_config["generator_name"]
            if generator_name == "random_linear_parkour_course":
            # DO SOMETHING DIFFERENT
                p_config = hm_config["parkour_course"]
                envs_per_block = p_config["envs_per_block"]
                env_spacing_x = p_config["env_spacing_x"]
                env_spacing_y = p_config["env_spacing_y"]
                lower = gymapi.Vec3(-env_spacing_x, -env_spacing_y, 0.0)
                upper = gymapi.Vec3(env_spacing_x, env_spacing_y, env_spacing_x)
                # we need a tiny bit of env spacing ^ to make sure no IG collision out of memory errors occur

                # for this env, all agents share the same terrain.
                # So we will handle their roots globally

                self._env_offsets = np.zeros(shape=(self._num_envs, 3), dtype=np.float32)

                

                for i in range(self._num_envs):
                    curr_block = int(i / envs_per_block)
                    self._env_offsets[i, :] = [(i % envs_per_block) * env_spacing_x * 2.0, curr_block * env_spacing_y * 2.0, 0.0]
                    Logger.print("Building {:d}/{:d} envs".format(i + 1, self._num_envs), end='\r')
                    env_ptr = self._gym.create_env(self._sim, lower, upper, envs_per_block)
                    self._build_env(i, env_ptr, config)
                    self._envs.append(env_ptr)
                self._env_offsets = torch.tensor(self._env_offsets, device=self._device)

                Logger.print("\n")
                return
            elif generator_name == "vault":
                num_env_per_row = self._num_envs
                env_spacing = env_config["env_spacing"]
                lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
                upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
                self._env_offsets = torch.zeros(size = (self._num_envs,3), dtype=torch.float32, device=self._device)
                self._env_offsets[:, 0] = torch.arange(0, self._num_envs, 1, dtype=torch.float32, device=self._device) * env_spacing * 2.0
            
                for i in range(self._num_envs):
                    Logger.print("Building {:d}/{:d} envs".format(i + 1, self._num_envs), end='\r')
                    env_ptr = self._gym.create_env(self._sim, lower, upper, num_env_per_row)
                    self._build_env(i, env_ptr, config)
                    self._envs.append(env_ptr)

                Logger.print("\n")
                return
            elif generator_name == "gap":
                assert False # TODO
            elif generator_name == "vault_gap":
                # 2 rows. 1st row is vault, second row is leap
                num_env_per_row = self._num_envs# // 2
                env_spacing = env_config["env_spacing"]

                #second_row_start_y = hm_config["vault_gap"]["second_row_y"]
                #second_row_spacing = second_row_start_y / 2.0
                
                
                #lower = gymapi.Vec3(-env_spacing, -second_row_spacing, 0.0)
                #upper = gymapi.Vec3(env_spacing, second_row_spacing, env_spacing)
                lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
                upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
                self._env_offsets = torch.zeros(size = (self._num_envs,3), dtype=torch.float32, device=self._device)
                self._env_offsets[:, 0] = torch.arange(0, num_env_per_row, 1, dtype=torch.float32, device=self._device) * env_spacing * 2.0
                #self._env_offsets[0:num_env_per_row, 0] = torch.arange(0, num_env_per_row, 1, dtype=torch.float32, device=self._device) * env_spacing * 2.0
                #self._env_offsets[num_env_per_row:, 0]  = torch.arange(0, num_env_per_row, 1, dtype=torch.float32, device=self._device) * env_spacing * 2.0
                
                #self._env_offsets[num_env_per_row:, 1] = second_row_start_y

                for i in range(self._num_envs):
                    Logger.print("Building {:d}/{:d} envs".format(i + 1, self._num_envs), end='\r')
                    env_ptr = self._gym.create_env(self._sim, lower, upper, num_env_per_row)
                    self._build_env(i, env_ptr, config)
                    self._envs.append(env_ptr)

                Logger.print("\n")
                return
            
            elif generator_name == "flat":
                # No env offsets
                self._env_offsets = np.zeros(shape=(self._num_envs, 3), dtype=np.float32)

                

                for i in range(self._num_envs):
                    lower = gymapi.Vec3(0.0, 0.0, 0.0)
                    upper = gymapi.Vec3(0.0, 0.0, 0.0)
                    envs_per_block = 1
                    curr_block = int(i / envs_per_block)
                    Logger.print("Building {:d}/{:d} envs".format(i + 1, self._num_envs), end='\r')
                    
                    
                    env_ptr = self._gym.create_env(self._sim, lower, upper, envs_per_block)
                    self._build_env(i, env_ptr, config)
                    self._envs.append(env_ptr)
                self._env_offsets = torch.tensor(self._env_offsets, device=self._device)

                Logger.print("\n")
                return

        env_spacing = self._get_env_spacing()
        if self._env_style == "square":
            num_env_per_row = int(np.sqrt(self._num_envs))
            lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
            upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
            self._env_offsets = torch.zeros(size = (self._num_envs,3), dtype=torch.float32, device=self._device)
            for i in range(self._num_envs):
                curr_col = i % num_env_per_row
                curr_row = i // num_env_per_row
                self._env_offsets[i, 0] = env_spacing * 2 * curr_col
                self._env_offsets[i, 1] = env_spacing * 2 * curr_row

            # create spawn boundaries
            self._spawn_max_x = self._env_spacing * 10.0
            self._spawn_min_x = -self._env_spacing * 10.0
            self._spawn_max_y = self._env_spacing * 10.0
            self._spawn_min_y = -self._env_spacing * 10.0
        elif self._env_style == "x_line":
            num_env_per_row = self._num_envs
            lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
            upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
            self._env_offsets = torch.zeros(size = (self._num_envs,3), dtype=torch.float32, device=self._device)
            self._env_offsets[:, 0] = torch.arange(0, self._num_envs, 1, dtype=torch.float32, device=self._device) * config["env"]["env_spacing"] * 2.0
        elif self._env_style == "y_line":
            assert False # TODO: test more
            num_env_per_row = 2
            lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
            length = config["env"]["terrain_length"]
            upper = gymapi.Vec3(length, env_spacing, env_spacing)
        else:
            assert False


        
        
        
        for i in range(self._num_envs):
            Logger.print("Building {:d}/{:d} envs".format(i + 1, self._num_envs), end='\r')
            env_ptr = self._gym.create_env(self._sim, lower, upper, num_env_per_row)
            self._build_env(i, env_ptr, config)
            self._envs.append(env_ptr)

        Logger.print("\n")

        return

    
    @abc.abstractmethod
    def _build_env(self, env_id, env_ptr, config):
        return
    
    @abc.abstractmethod
    def _reset_envs(self, env_ids):
        if (len(env_ids) > 0):
            self._timestep_buf[env_ids] = 0
            self._time_buf[env_ids] = 0
            self._done_buf[env_ids] = base_env.DoneFlags.NULL.value
        return

    def _reset_sim_tensors(self):
        # note: need_reset_buf and actors_need_reset are same tensor, different views
        actor_ids = self._need_reset_buf.nonzero(as_tuple=False)
        actor_ids = actor_ids.type(torch.int32).flatten()

        if (len(actor_ids) > 0):
            self._gym.set_actor_root_state_tensor_indexed(self._sim,
                                                         gymtorch.unwrap_tensor(self._root_state),
                                                         gymtorch.unwrap_tensor(actor_ids), len(actor_ids))
            if (self._dof_state is not None):
                has_dof = self._actor_dof_dims[actor_ids.type(torch.long)] > 0
                dof_actor_ids = actor_ids[has_dof]
                self._gym.set_dof_state_tensor_indexed(self._sim,
                                                      gymtorch.unwrap_tensor(self._dof_state),
                                                      gymtorch.unwrap_tensor(dof_actor_ids), len(dof_actor_ids))
                
                dof_pos = self._dof_state[..., :, 0]
                dof_pos = dof_pos.contiguous()
                self._gym.set_dof_position_target_tensor_indexed(self._sim,
                                                      gymtorch.unwrap_tensor(dof_pos),
                                                      gymtorch.unwrap_tensor(dof_actor_ids), len(dof_actor_ids))

            reset_env_ids = torch.sum(self._actors_need_reset, dim=-1).nonzero(as_tuple=False)
            reset_env_ids = reset_env_ids.flatten()
            self._actors_need_reset[:] = False
        else:
            reset_env_ids = []

        return reset_env_ids

    def _get_env_spacing(self):
        return self._env_spacing
    
    def _build_ground_plane(self, config):
        env_configs = config["env"]

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = env_configs["plane"]["static_friction"]
        plane_params.dynamic_friction = env_configs["plane"]["dynamic_friction"]
        plane_params.restitution = env_configs["plane"]["restitution"]
        if "z_offset" in env_configs["plane"]:
            plane_params.distance = env_configs["plane"]["z_offset"]
        self._gym.add_ground(self._sim, plane_params)
        return
    
    

    def _build_viewer(self):
        # subscribe to keyboard shortcuts
        self._viewer = self._gym.create_viewer(
            self._sim, gymapi.CameraProperties())
        self._gym.subscribe_viewer_keyboard_event(
            self._viewer, gymapi.KEY_ESCAPE, "QUIT")
        self._gym.subscribe_viewer_keyboard_event(
            self._viewer, gymapi.KEY_V, "toggle_viewer_sync")

        # set the camera position based on up axis
        sim_params = self._gym.get_sim_params(self._sim)
        if sim_params.up_axis == gymapi.UP_AXIS_Z:
            cam_pos = gymapi.Vec3(20.0, 25.0, 3.0)
            cam_target = gymapi.Vec3(10.0, 15.0, 0.0)
        else:
            cam_pos = gymapi.Vec3(20.0, 3.0, 25.0)
            cam_target = gymapi.Vec3(10.0, 0.0, 15.0)

        self._gym.viewer_camera_look_at(
                self._viewer, None, cam_pos, cam_target)

        return

    def _build_sim_tensors(self, config):
        root_state_tensor = self._gym.acquire_actor_root_state_tensor(self._sim)
        dof_state_tensor = self._gym.acquire_dof_state_tensor(self._sim)
        rigid_body_state = self._gym.acquire_rigid_body_state_tensor(self._sim)
        force_sensor_tensor = self._gym.acquire_force_sensor_tensor(self._sim)
        contact_force_tensor = self._gym.acquire_net_contact_force_tensor(self._sim)
        
        self._root_state = gymtorch.wrap_tensor(root_state_tensor)
        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        self._contact_forces = gymtorch.wrap_tensor(contact_force_tensor)
        
        if self._enable_dof_force_sensors():
            dof_force_tensor = self._gym.acquire_dof_force_tensor(self._sim)
            self._dof_forces = gymtorch.wrap_tensor(dof_force_tensor)

        return
            
    def _build_data_buffers(self):
        num_envs = self.get_num_envs()
        actors_per_env = self._get_actors_per_env()

        self._reward_buf = torch.zeros(num_envs, device=self._device, dtype=torch.float)
        self._done_buf = torch.zeros(num_envs, device=self._device, dtype=torch.int)
        self._timestep_buf = torch.zeros(num_envs, device=self._device, dtype=torch.int)
        self._time_buf = torch.zeros(num_envs, device=self._device, dtype=torch.float)

        self._need_reset_buf = torch.zeros(self._root_state.shape[0], device=self._device, dtype=torch.bool)
        self._actors_need_reset = self._need_reset_buf.view((num_envs, actors_per_env))
        
        obs_space = self.get_obs_space()
        obs_dtype = torch_util.numpy_dtype_to_torch(obs_space.dtype)
        self._obs_buf = torch.zeros([num_envs] + list(obs_space.shape), device=self._device, dtype=obs_dtype)

        self._actor_dof_dims = self._build_actor_dof_dims()

        self._info = dict()

        return

    def _build_actor_dof_dims(self):
        num_envs = self.get_num_envs()
        actors_per_env = self._get_actors_per_env()

        actor_dof_dims = torch.zeros(self._need_reset_buf.shape, device=self._device, dtype=torch.int)
        env_actor_dof_dims = actor_dof_dims.view([num_envs, actors_per_env])

        for e in range(num_envs):
            env_handle = self._envs[e]
            for a in range(actors_per_env):
                num_dofs = self._gym.get_actor_dof_count(env_handle, a)
                env_actor_dof_dims[e, a] = num_dofs

        return actor_dof_dims

    @abc.abstractmethod
    def _build_action_space(self):
        return
    
    def _get_actors_per_env(self):
        n = self._root_state.shape[0] // self.get_num_envs()
        return n
    
    def _pre_physics_step(self, actions):
        return
    
    def _physics_step(self):
        for i in range(self._sim_steps):
            self._step_sim()
        return

    def _step_sim(self):
        self._gym.simulate(self._sim)
        return
    
    def _post_physics_step(self):
        self._refresh_sim_tensors()
        
        self._update_time()
        self._update_misc()
        self._update_observations()
        self._update_info()
        self._update_reward()
        self._update_done()
        return

    def _refresh_sim_tensors(self):
        self._gym.refresh_dof_state_tensor(self._sim)
        self._gym.refresh_actor_root_state_tensor(self._sim)
        self._gym.refresh_rigid_body_state_tensor(self._sim)

        self._gym.refresh_force_sensor_tensor(self._sim)
        self._gym.refresh_net_contact_force_tensor(self._sim)
        
        if self._enable_dof_force_sensors():
            self._gym.refresh_dof_force_tensor(self._sim)
        return
    
    def _update_time(self, num_steps=1):
        self._timestep_buf += num_steps
        self._time_buf[:] = self._timestep * self._timestep_buf
        return

    def _update_misc(self):
        return

    def _update_observations(self, env_ids=None):
        if (env_ids is None or len(env_ids) > 0):
            obs = self._compute_obs(env_ids)
            if (env_ids is None):
                self._obs_buf[:] = obs
            else:
                self._obs_buf[env_ids] = obs
        return

    def _enable_dof_force_sensors(self):
        return False
    
    @abc.abstractmethod
    def _update_reward(self):
        return
    
    @abc.abstractmethod
    def _update_done(self):
        return

    def _update_info(self, env_ids=None):
        return
    
    @abc.abstractmethod
    def _compute_obs(env_ids=None):
        return

    def _render(self, sync_frame_time=False, clear_lines=True):
        # check for window closed
        if (self._gym.query_viewer_has_closed(self._viewer)):
            sys.exit()

        # check for keyboard events
        for evt in self._gym.query_viewer_action_events(self._viewer):
            if evt.action == "QUIT" and evt.value > 0:
                sys.exit()
            elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                self._enable_viewer_sync = not self._enable_viewer_sync

        # fetch results
        if (self._device != "cpu"):
            self._gym.fetch_results(self._sim, True)
                
        # step graphics
        if (self._enable_viewer_sync):
            self._gym.step_graphics(self._sim)
            self._gym.draw_viewer(self._viewer, self._sim, True)
        else:
            self._gym.poll_viewer_events(self._viewer)
        
        if clear_lines:
            self._gym.clear_lines(self._viewer)

        return

    def _init_camera(self):
        return

    def _update_camera(self):
        return

    def _get_vis_col_group(self):
        return self.get_num_envs()
    
    def _draw_point(self, env, pt, color, size=0.01):
        if isinstance(pt, torch.Tensor):
            pt = pt.cpu().detach().numpy()

        if isinstance(color, torch.Tensor):
            color = color.cpu().detach().numpy()

        vertices = []
        for i in range(3):
            axis = np.zeros(3)
            axis[i] = size
            vertices.append(pt)
            vertices.append(pt + axis)
            vertices.append(pt)
            vertices.append(pt - axis)

        vertices = np.array(vertices, dtype=np.float32)
        colors = np.broadcast_to(color, (6, 3)).astype(np.float32)
        self._gym.add_lines(self._viewer, env, 6, vertices, colors)
        return

    def _draw_line(self, env, p1, p2, color):
        if isinstance(p1, torch.Tensor):
            p1 = p1.cpu().detach().numpy()

        if isinstance(p2, torch.Tensor):
            p2 = p2.cpu().detach().numpy()

        if isinstance(color, torch.Tensor):
            color = color.cpu().detach().numpy()

        vertices = np.array([p1, p2], dtype=np.float32)
        colors = np.array([color], dtype=np.float32)
        self._gym.add_lines(self._viewer, env, 1, vertices, colors)
        return
    
    def _draw_flag(self, env, point, height, color):
        if isinstance(point, torch.Tensor):
            point = point.cpu().detach().numpy()
        # draw a flag at the point
        p1 = [point[0], point[1], point[2]]
        p2 = [point[0], point[1], point[2] + height]
        self._draw_line(env, p1, p2, color)
        self._draw_point(env, p1, color, 0.1)
        self._draw_point(env, p2, color, 0.1)

        return
