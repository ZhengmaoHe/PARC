import isaacgym.gymapi as gymapi
import torch
import numpy as np
import time
import pickle
import enum
import random
import os
import yaml

import envs.base_env as base_env
import envs.ig_parkour.mgdm_dm_util as mgdm_dm_util
import util.torch_util as torch_util
import util.terrain_util as terrain_util
import util.geom_util as geom_util
import diffusion.mdm as mdm
from diffusion.mdm import MDMKeyType
from diffusion.diffusion_util import MDMCustomGuidance
import anim.motion_lib as motion_lib
import diffusion.gen_util as gen_util
from util.motion_util import MotionFrames
import util.motion_util as motion_util
import copy

SIM_CHAR_IDX = 0
REF_CHAR_IDX = 1

class ReplanFlags(enum.Enum):
    REPLAN = 0
    HARD_RESET = 1

def load_mdm(model_path) -> mdm.MDM:
    with open(model_path, 'rb') as input_file:
        mgen = pickle.load(input_file)
    return mgen

class MotionGenDeepMimicEnv(mgdm_dm_util.RefCharEnv):
    def __init__(self, config, num_envs, device, visualize, char_model):
        env_config = config["env"]
        mgdm_config = env_config["mgdm"]
        
        super().__init__(config, num_envs, device, visualize, char_model)
        self._demo_mode = env_config["demo_mode"]
        self._plan_length = mgdm_config["plan_length"]
        self._ddim_stride = mgdm_config["ddim_stride"]
        self._max_replans = mgdm_config["max_replans"]
        self._replan_flag = True # important to start with True since reset() is the first function called
        self._cfg_scale = mgdm_config["cfg_scale"]
        self._target_dist_max = mgdm_config["target_dist_max"]
        self._target_dist_min = mgdm_config["target_dist_min"]
        self._target_dur_max = mgdm_config["target_dur_max"]
        self._target_dur_min = mgdm_config["target_dur_min"]
        self._target_radius = env_config["target_radius"]
        self._target_heading_scale = mgdm_config["target_heading_scale"]
        self._true_tensor = torch.ones(size=(num_envs,), dtype=torch.bool, device=device)
        self._dont_auto_update_targets = mgdm_config.get("dont_auto_update_targets", False)

        print("target radius", self._target_radius)
        print("target heading scale:", self._target_heading_scale)

        ## Load motion generator
        model_path = mgdm_config["model_path"]
        self._mgen = load_mdm(model_path)
        
        self._num_prev_states = self._mgen._num_prev_states
        if self._num_prev_states > 1:
            self._agent_state_hist = self._get_char_motion_frames(ref=False, null=True)
            self._ref_state_hist = self._get_char_motion_frames(ref=True, null=True)

        self._build_obs_hfs(env_config, num_envs, device)
        self._motion_ids = torch.arange(0, num_envs, device=self._device, dtype=torch.int64)

        # GLOBAL space target locations
        #self._target_xy = torch.zeros(size=(num_envs, self.get_target_dim()), device=self._device, dtype=torch.float)
        self._next_target_xy_time = torch.zeros(num_envs, device=self._device, dtype=torch.float)

        # For tracking the time in the current generated sequence
        self._mgdm_time_buf = torch.zeros(size=(1,), device=self._device, dtype=torch.float)
        self._replan_time_buf = self._mgdm_time_buf # just overloading name
        # for replanning
        #self._replan_time_buf = torch.zeros(size=(1,), dtype=torch.float32, device=self._device)

        # for determining if an env needs to hard reset or replan when the replan interval comes up
        self._replan_buf = torch.zeros(size=(num_envs,), device=self._device, dtype=torch.int64)
        self._replan_buf[:] = ReplanFlags.HARD_RESET.value
        self._replan_counter = torch.zeros(size=(num_envs,), device=self._device, dtype=torch.int64)
        return
    
    def reset(self, env_ids):
        # Handle scheduled replanning and hard resets
        # Right now we are assuming the simulator calls reset() every timestep, even
        # if it passes in an empty tensor of env ids

        if self._replan_flag:
            if self._demo_mode: # just so we can see long trajectories
                self._replan_buf[:] = ReplanFlags.REPLAN.value
            self.replan()
        
        else: # Handle soft resets # NOTE: figure out if this being an else statement is correct
            if (len(env_ids) > 0):
                self._timestep_buf[env_ids] = 0
                self._time_buf[env_ids] = 0
                self._done_buf[env_ids] = base_env.DoneFlags.NULL.value
                
            if (len(env_ids) > 0):
                self._reset_char(env_ids)
        return

    def pre_physics_step(self):
        # update history of agent, which must happen after history is used as input to MDM,
        # and before simulate ovewrites the agent state
        self._agent_state_hist = self._get_char_motion_frames(ref=False)
        return
    
    def update_time(self, timestep):
        self._mgdm_time_buf[0] += timestep
        return

    def update_misc(self):
        self._ref_state_hist = self._get_char_motion_frames(ref=True)

        # check if we need to update next target time
        target_update_check = self._time_buf > self._next_target_xy_time
        
        update_env_ids = target_update_check.nonzero()
        if len(update_env_ids) > 0:
            self.pick_new_xy_targets(update_env_ids.squeeze(-1))
        return
    
    def compute_tar_obs(self, tar_obs_steps, env_ids = None):
        if env_ids is not None:
            motion_ids = self._motion_ids[env_ids]
            motion_times = self._mgdm_time_buf.expand(len(env_ids))
        else:
            motion_ids = self._motion_ids
            motion_times = self._mgdm_time_buf.expand(self._num_envs)
        tar_root_pos, tar_root_rot, tar_joint_rot, tar_contacts = mgdm_dm_util.fetch_tar_obs_data(motion_ids, 
                                                                                                  motion_times,
                                                                                                  self._motion_lib,
                                                                                                  self._timestep,
                                                                                                  tar_obs_steps)
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
        # also reset agents that get too close to the terrain edges
        global_xy = self._char_root_pos[..., 0:2] + self._env_offsets[..., 0:2]

        min_point = self._terrain.min_point + self._oob_region
        max_point = self._terrain.min_point + self._terrain.dims * self._terrain.dxdy - self._oob_region

        oob_xmin = global_xy[..., 0] < min_point[0]
        oob_xmax = global_xy[..., 0] > max_point[0]
        oob_ymin = global_xy[..., 1] < min_point[1]
        oob_ymax = global_xy[..., 1] > max_point[1]

        oob_x = torch.logical_or(oob_xmin, oob_xmax)
        oob_y = torch.logical_or(oob_ymin, oob_ymax)
        oob = torch.logical_or(oob_x, oob_y)

        self._done_buf[oob] = base_env.DoneFlags.TIME.value
        self._replan_buf[oob] = ReplanFlags.HARD_RESET.value

        # Finally, reset agents who have jumped up way too high
        too_high = self._char_root_pos[..., 2] > 3.0
        self._done_buf[too_high] = base_env.DoneFlags.FAIL.value
        
        # if there are any fails, keep track of it in the replan buf
        fail_env_mask = self._done_buf == base_env.DoneFlags.FAIL.value
        self._replan_buf[fail_env_mask] = ReplanFlags.HARD_RESET.value

        if self._mgdm_time_buf[0] > self._plan_length:
            self._replan_flag = True
            print("Time to replan")
            print("replan time buf:", self._mgdm_time_buf[0])

            hard_reset_mask = self._compute_hard_reset_envs_mask()
            # set the TIME done flag for hard reset envs, but only if the env isn't failing
            done_mask = self._done_buf == base_env.DoneFlags.NULL.value
            done_mask = torch.logical_and(done_mask, hard_reset_mask)
            self._done_buf[done_mask] = base_env.DoneFlags.TIME.value
            ## If the character is too close to the edge, also hard reset?
            # for now, we will just use a replan counter and use a max number of replans,
            # which is accounted for in the replan function
        return

    def build_terrain(self, env_config, terrain_save_path):
        start_time = time.perf_counter()
        hm_config = env_config["mgdm"]["heightmap"]


        dx = hm_config["horizontal_scale"]
        sq_m_per_env = hm_config["sq_m_per_env"]
        safety_region = hm_config["safety_region"]

        if "terrain_file" in env_config["mgdm"]:
            load_terrain_path = env_config["mgdm"]["terrain_file"]

            path_ext = os.path.splitext(load_terrain_path)[1]
            if path_ext == ".pkl":
                with open(load_terrain_path, "rb") as f:
                    new_terrain = pickle.load(f)["terrain"]
                    np_terrain = copy.deepcopy(new_terrain)
                    new_terrain.to_torch(self._device)

                
            elif path_ext == ".yaml":

                self._hard_motion_lib = motion_lib.MotionLib(load_terrain_path, self._kin_char_model, self._device)
                with open(load_terrain_path, "r") as f:
                    motion_terrain_yaml = yaml.safe_load(f)

                    terrain_path = motion_terrain_yaml["terrain"]
                    print(terrain_path)
                    with open(terrain_path, "rb") as f2:
                        new_terrain = pickle.load(f2)["terrain"]
                        np_terrain = copy.deepcopy(new_terrain)
                        new_terrain.to_torch(self._device)

            min_x = new_terrain.min_point[0].item()
            min_y = new_terrain.min_point[1].item()

            x_length = new_terrain.dims[0].item() * new_terrain.dxdy[0].item()
            y_length = new_terrain.dims[1].item() * new_terrain.dxdy[1].item()

        else:

            num_envs = 2048
            x_length = np.round(np.sqrt(num_envs)) * sq_m_per_env + safety_region * 2.0
            y_length = x_length

            grid_dim_x = int(x_length / dx)
            grid_dim_y = int(y_length / dx)

            min_x = -x_length / 2.0
            min_y = -y_length / 2.0

            new_terrain = terrain_util.SubTerrain(x_dim=grid_dim_x, y_dim=grid_dim_y, dx=dx, dy=dx,
                                                    min_x=min_x, min_y=min_y, device=self._device)
            
            
            # TODO: read these params from file
            # boxes_per_sq_m = hm_config["boxes_per_sq_m"]
            # num_boxes = int(np.round(x_length * y_length * boxes_per_sq_m))
            # terrain_util.add_boxes_to_hf(hf = new_terrain.hf,
            #                              hf_mask = new_terrain.hf_mask,
            #                              box_heights = [0.6],
            #                              hf_maxmin = new_terrain.hf_maxmin,
            #                              num_boxes = num_boxes,
            #                              box_max_len = 50, 
            #                              box_min_len = 5)

            # quadrants
            num_segments = hm_config["num_segments"]
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

            platform_heights = hm_config["platform_heights"]

            for i in range(num_segments):
                for j in range(num_segments):
                    if i % 2 == 0:
                        if j % 2 == 0:
                            height_ind = random.randint(0, len(platform_heights) - 1)
                            val = platform_heights[height_ind]#0.6
                        else: 
                            val = 0.0
                    else:
                        if j % 2 == 0:
                            val = 0.0
                        else:
                            height_ind = random.randint(0, len(platform_heights) - 1)
                            val = platform_heights[height_ind]#0.6
                    new_terrain.hf[segments_x[i]:segments_x[i+1], segments_y[j]:segments_y[j+1]] = val
            
            np_terrain = new_terrain.numpy_copy()
        vertices, triangles = terrain_util.convert_heightfield_to_voxelized_trimesh(
            np_terrain.hf, 
            min_x = np_terrain.min_point[0],
            min_y = np_terrain.min_point[1],
            dx = dx
        )

        self._oob_region = safety_region / 10
        self._spawn_min_x = min_x + safety_region
        self._spawn_min_y = min_y + safety_region
        self._spawn_max_x = min_x + x_length - safety_region #x_len - padding*2.0 # gonna space envs manually
        self._spawn_max_y = min_y + y_length - safety_region #real_vault_spacing * 1.1

        end_time = time.perf_counter()
        print("building mgdm heightfield and mesh time:", end_time-start_time, " seconds.")
        print("vertices.shape:", vertices.shape)
        print("triangles.shape:", triangles.shape)

        self._terrain = new_terrain

        save_data = {
            "oob_region": self._oob_region,
            "spawn_min_x": self._spawn_min_x,
            "spawn_max_x": self._spawn_max_x,
            "spawn_min_y": self._spawn_min_y,
            "spawn_max_y": self._spawn_max_y,
            "terrain": self._terrain,
            "vertices": vertices,
            "triangles": triangles
        }

        with open(terrain_save_path, "wb") as f:
            print("writing to:", terrain_save_path)
            pickle.dump(save_data, f)

        return vertices, triangles, new_terrain.min_point
    
    def load_terrain(self, terrain_save_path):
        with open(terrain_save_path, "rb") as f:
            print("loading :", terrain_save_path)
            save_data = pickle.load(f)

        self._oob_region = save_data["oob_region"]
        self._spawn_min_x = save_data["spawn_min_x"]
        self._spawn_max_x = save_data["spawn_max_x"]
        self._spawn_min_y = save_data["spawn_min_y"]
        self._spawn_max_y = save_data["spawn_max_y"]
        self._terrain = save_data["terrain"]
        vertices = save_data["vertices"]
        triangles = save_data["triangles"]

        return vertices, triangles, self._terrain.min_point
    
    def _build_obs_hfs(self, env_config, num_envs, device):
        mgdm_config = env_config["mgdm"]
        hm_config = mgdm_config["heightmap"]
        # dx = hm_config["local_grid"]["dx"]
        # dy = hm_config["local_grid"]["dy"]
        # num_x_neg = hm_config["local_grid"]["num_x_neg"]
        # num_x_pos = hm_config["local_grid"]["num_x_pos"]
        # num_y_neg = hm_config["local_grid"]["num_y_neg"]
        # num_y_pos = hm_config["local_grid"]["num_y_pos"]
        dx = self._mgen._dx
        dy = self._mgen._dy
        num_x_neg = self._mgen._num_x_neg
        num_x_pos = self._mgen._num_x_pos
        num_y_neg = self._mgen._num_y_neg
        num_y_pos = self._mgen._num_y_pos
        self._mgdm_local_xy_points = geom_util.get_xy_grid_points(
                                    center=torch.zeros(size=(2,), 
                                                       dtype=torch.float32, 
                                                       device=device),
                                                       dx=dx,
                                                       dy=dy,
                                                       num_x_neg=num_x_neg,
                                                       num_x_pos=num_x_pos,
                                                       num_y_neg=num_y_neg,
                                                       num_y_pos=num_y_pos)
        
        num_points = (1 + num_x_neg + num_x_pos) * (1 + num_y_neg + num_y_pos)
        self._mgdm_hfs = torch.zeros(size=(num_envs, num_points), dtype=torch.float32, device=device)
        self._mgdm_floor_heights = torch.zeros(size=(num_envs,), dtype=torch.float32, device=device)
        return
    
    def refresh_obs_hfs(self, char_root_pos_xyz, char_heading):
        # char_root_pos_xy: global root positions

        self._refresh_ray_obs_hfs(char_root_pos_xyz, char_heading)

        #root_floor_height = terrain_util.get_local_hf_from_terrain(char_root_pos_xy, self._terrain)

        grid_dim_x = self._mgdm_local_xy_points.shape[0]
        grid_dim_y = self._mgdm_local_xy_points.shape[1]

        char_root_pos_xy = char_root_pos_xyz[..., 0:2]
        mgdm_char_root_pos_xy = char_root_pos_xy.unsqueeze(1).unsqueeze(1).expand(-1, grid_dim_x, grid_dim_y, -1)
        mgdm_local_xy_points = self._mgdm_local_xy_points.unsqueeze(0).expand(mgdm_char_root_pos_xy.shape[0], -1, -1, -1)

        # rotate the points around the character's root
        mgdm_char_heading = char_heading.unsqueeze(-1).unsqueeze(-1).expand(-1, grid_dim_x, grid_dim_y)
        mgdm_local_xy_points = torch_util.rotate_2d_vec(mgdm_local_xy_points, mgdm_char_heading) \
            + mgdm_char_root_pos_xy
        mgdm_local_xy_points = mgdm_local_xy_points.view(self._num_envs * grid_dim_x * grid_dim_y, 2)
        mgdm_hfs = terrain_util.get_local_hf_from_terrain(mgdm_local_xy_points, self._terrain) \
            .view(self._num_envs, grid_dim_x * grid_dim_y)

        
        self._mgdm_hfs = mgdm_hfs - self._char_root_pos[..., 2].unsqueeze(-1)
        self._mgdm_floor_heights = terrain_util.get_local_hf_from_terrain(char_root_pos_xy, self._terrain)
        # Note: using this function^ in a strange way, but it works

        if self._visualize:
            self._mgdm_xyz_points = torch.cat([mgdm_local_xy_points.view(self._num_envs, -1, 2), mgdm_hfs.unsqueeze(-1)], dim=-1)
        return

    def pick_new_xy_targets(self, env_ids = None):
        # select a random point within the XY bounds of the terrain
        # Only call this with mgdm env ids

        if (env_ids is None):
            env_ids = torch.arange(end=self._num_envs, dtype=torch.int64, device=self._device)
            num_envs = self._num_envs
        else:
            num_envs = len(env_ids)

        # pick a point around the character
        rel_traj_pos = torch.zeros(size=(num_envs, 2), dtype=torch.float32, device=self._device)
        rel_traj_pos[:, 0] = 1.0

        # pick a rotation
        target_heading = torch.rand(size=(num_envs,), dtype=torch.float32, device=self._device) * (torch.pi * 2) - torch.pi
        # TEMP: scale the rotation down
        # target_heading *= 0.5
        target_heading = target_heading * self._target_heading_scale

        target_distance = torch.rand(size=(num_envs,), dtype=torch.float32, device=self._device)
        target_distance = target_distance * (self._target_dist_max - self._target_dist_min) + self._target_dist_min
        rel_traj_pos *= target_distance.unsqueeze(dim=-1)
        rel_traj_pos = torch_util.rotate_2d_vec(rel_traj_pos, target_heading)
        
        
        char_heading = torch_util.calc_heading(self._char_root_rot[env_ids])
        rel_traj_pos = torch_util.rotate_2d_vec(rel_traj_pos, char_heading)

        # todo: check this char_root_pos between test iterations
        new_targets = self._char_root_pos[env_ids, 0:2] + rel_traj_pos

        self._target_xy[env_ids] = new_targets


        # ALSO pick the time that the next target should get picked
        next_target_times = torch.rand(size=(num_envs,), dtype=torch.float32, device=self._device)
        next_target_times = next_target_times * (self._target_dur_max - self._target_dur_min) + self._target_dur_min
        next_target_times = next_target_times + self._time_buf[env_ids]

        if self._dont_auto_update_targets:
            self._next_target_xy_time[env_ids] = 100000.0
        else:
            self._next_target_xy_time[env_ids] = next_target_times
        return

    def _update_ref_motion(self):
        motion_ids = self._motion_ids
        motion_times = self._mgdm_time_buf.expand(self._num_envs)
        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, contacts = \
            self._motion_lib.calc_motion_frame(motion_ids, motion_times)
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
        return
    
    def _reset_char(self, env_ids):
        ## This function is called whenever the environment is reset.
        # We only want to reset the char to the current frame
        # of the generated motion sequence. We do not want to regen
        # the motion sequence here, since the plan is to have that
        # happen at fixed time intervals across all envs.

        # env reset is called after taking a step. In the first env step,
        # the planner will generate the motion sequence and update the ref char tensors
        
        if len(env_ids) > 0:
            self._char_state_init_from_ref(env_ids)
            self._actors_need_reset[env_ids, SIM_CHAR_IDX] = True

            # use the ref char's history in place of the agent char's history
            # NOTE: the state history only matters if the reset happens right before a replan
            self._agent_state_hist.set_vals(self._ref_state_hist, env_ids)
        return
    
    
    
    def _get_char_motion_frames(self, eps=1e-5, ref=False, null=False):
        
        if null:
            ret = MotionFrames()
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

            ret = MotionFrames(
                root_pos = root_pos.clone(),
                root_rot = root_rot.clone(),
                joint_rot = joint_rot.clone(),
                contacts = char_contacts.clone()
            )

            ret = ret.unsqueeze(1)
        
        return ret

    def _get_state_dict_from_motion_lib(self, t, motion_ids):

        motion_times = torch.ones(motion_ids.shape, dtype=torch.float, device=self._device) * t

        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, contacts = self._motion_lib.calc_motion_frame(motion_ids, motion_times)     
        
        #joint_pos, joint_body_rot = self._kin_char_model.forward_kinematics(root_pos, root_rot, joint_rot)
        
        ret = MotionFrames(root_pos = root_pos,
                           root_rot = root_rot,
                           joint_rot = joint_rot,
                           contacts = contacts)
        ret = ret.unsqueeze(1)
        return ret

    def _compute_hard_reset_envs_mask(self):
        max_replan_mask = self._replan_counter >= self._max_replans
        self._replan_buf[max_replan_mask] = ReplanFlags.HARD_RESET.value
        # This should be fine...

        hard_reset_mask = self._replan_buf == ReplanFlags.HARD_RESET.value
        return hard_reset_mask

    @torch.no_grad()
    def replan(self):
        print("REPLAN")
        hard_reset_mask = self._compute_hard_reset_envs_mask()
        hard_reset_ids = hard_reset_mask.nonzero().squeeze(dim=-1)
        num_hard_resets = hard_reset_ids.shape[0]
        replan_ids = (~hard_reset_mask).nonzero().squeeze(dim=-1)
        num_replans = replan_ids.shape[0]

        ## Step 1: get the heightfield for the hard reset characters ##
        ## as well as their new starting xy location
        if num_hard_resets > 0:
            # TODO: hard reset to hard_motion_lib motion
            print("num hard resets:", num_hard_resets)
            #self._curve_level[hard_reset_ids] = torch.rand_like(self._curve_level[hard_reset_ids]) * self._curve_level_max
            # pick new targets for the hard reset envs
            
            # Make sure hard reset envs get their time and done buffers reset
            self._timestep_buf[hard_reset_ids] = 0
            self._time_buf[hard_reset_ids] = 0
            self._done_buf[hard_reset_ids] = base_env.DoneFlags.NULL.value

            new_x_points = torch.rand(size=(num_hard_resets,), dtype=torch.float32, device=self._device)
            new_x_points = new_x_points * (self._spawn_max_x - self._spawn_min_x) + self._spawn_min_x
            new_y_points = torch.rand(size=(num_hard_resets,), dtype=torch.float32, device=self._device)
            new_y_points = new_y_points * (self._spawn_max_y - self._spawn_min_y) + self._spawn_min_y

            num_points = self._mgdm_local_xy_points.shape[0] * self._mgdm_local_xy_points.shape[1]
            global_new_xy_points = torch.zeros_like(self._char_root_pos[hard_reset_ids, 0:2])
            global_new_xy_points[:, 0] += new_x_points
            global_new_xy_points[:, 1] += new_y_points

            new_heading = torch.rand(size=[num_hard_resets], dtype=torch.float32, device=self._device) * torch.pi * 2.0 - torch.pi
            local_xy_points = torch_util.rotate_2d_vec(self._mgdm_local_xy_points.unsqueeze(0).expand(num_hard_resets, -1, -1, -1), new_heading.unsqueeze(-1).unsqueeze(-1))

            # get local grid points for each env
            new_xy_grid_points = global_new_xy_points.unsqueeze(1).unsqueeze(1) + local_xy_points
            new_xy_grid_points = new_xy_grid_points.view(num_hard_resets * num_points, 2)
            
            new_hf = terrain_util.get_local_hf_from_terrain(new_xy_grid_points, self._terrain).view(num_hard_resets, num_points)
            
            # TODO: make more random?
            # hard_reset_root_heights = torch.ones_like(hard_reset_ids, dtype=torch.float32, device=self._device) * 0.7
            # new_hf = new_hf - hard_reset_root_heights.unsqueeze(-1)

            self._mgdm_hfs[hard_reset_ids] = new_hf

            new_floor_heights = terrain_util.get_local_hf_from_terrain(global_new_xy_points, self._terrain)
            self._mgdm_floor_heights[hard_reset_ids] = new_floor_heights

            # need to set these before targets are picked
            # root rot shouldn't matter though
            self._char_root_pos[hard_reset_ids, 0] = new_x_points - self._env_offsets[hard_reset_ids, 0]
            self._char_root_pos[hard_reset_ids, 1] = new_y_points - self._env_offsets[hard_reset_ids, 1]
            self.pick_new_xy_targets(hard_reset_ids)

            # TODO: don't hard code this
            self._char_root_pos[hard_reset_ids, 2] = new_floor_heights + torch.rand(size=[num_hard_resets], dtype=torch.float32, device=self._device) * 0.2 + 0.7

        curr_motion_frames = self._get_char_motion_frames()

        prev_frames = motion_util.cat_motion_frames([self._agent_state_hist, curr_motion_frames])
            
        
        mdm_gen_settings = gen_util.MDMGenSettings()
        mdm_gen_settings.ddim_stride = 100
        mdm_gen_settings.use_cfg = True
        # mdm_gen_settings.use_cfg = torch.ones(size=[self._num_envs], dtype=torch.bool, device=self._device)
        #mdm_gen_settings.use_cfg[hard_reset_ids] = False
        mdm_gen_settings.use_prev_state = torch.ones(size=[self._num_envs], dtype=torch.bool, device=self._device)
        mdm_gen_settings.use_prev_state[hard_reset_ids] = False
        mdm_gen_settings.prev_state_ind_key= torch.ones(size=[self._num_envs], dtype=torch.bool, device=self._device)
        mdm_gen_settings.prev_state_ind_key[hard_reset_ids] = False

        output_motion_frames = gen_util.gen_mdm_motion(
            self._target_xy, prev_frames, self._terrain,
            self._mgen,
            self._kin_char_model,
            mdm_gen_settings,
            verbose=False)

        # # Canonicalization
        # canon_root_pos = curr_agent_state_dict[mdm.MDMFrameType.ROOT_POS].clone()
        # canon_root_rot = curr_agent_state_dict[mdm.MDMFrameType.ROOT_ROT].clone()
        # heading_quat_inv = torch_util.calc_heading_quat_inv(canon_root_rot).unsqueeze(1)
        # heading = torch_util.calc_heading(canon_root_rot)
        # # Canonicalize the root rotation and rotate the root positions
        # #heading_quat_inv = torch_util.calc_heading_quat_inv(char_root_rot_quat).unsqueeze(dim=1).expand(-1, self._num_prev_states, -1)
        # #heading = torch_util.calc_heading(char_root_rot_quat) # need this for later
        # #prev_state[:, :, 0:2] -= self._char_root_pos[:, 0:2].unsqueeze(dim=1)
        # #prev_state[:, :, 0:3] = torch_util.quat_rotate(heading_quat_inv, prev_state[:, :, 0:3])

        # prev_state[mdm.MDMFrameType.ROOT_POS][..., 0:2] -= canon_root_pos[..., 0:2].unsqueeze(1)
        # prev_state[mdm.MDMFrameType.ROOT_POS] = torch_util.quat_rotate(heading_quat_inv, prev_state[mdm.MDMFrameType.ROOT_POS])
        # prev_state[mdm.MDMFrameType.ROOT_ROT] = torch_util.quat_multiply(heading_quat_inv, prev_state[mdm.MDMFrameType.ROOT_ROT])

        # # Root height is relative
        # #prev_state[mdm.MDMFrameType.ROOT_POS][..., 2] -= prev_state[mdm.MDMFrameType.ROOT_POS][..., 2].clone()
        
        # if self._mgen._target_type == mdm.TargetType.XY_DIR:

        #     rel_target_pos = self._target_xy - canon_root_pos[..., 0:2]
        #     target_distance = torch.norm(rel_target_pos, dim=-1, keepdim=True)
        #     target_dir = torch.where(target_distance >= self._target_radius, rel_target_pos / target_distance, torch.zeros_like(rel_target_pos))

        #     #char_heading = torch_util.calc_heading(self._char_root_rot)
        #     rel_target_dir = torch_util.rotate_2d_vec(target_dir, -heading)
        #     target_key = rel_target_dir
        # else:
        #     assert False

        

        # # canonicalize
        # #target_key = torch_util.rotate_2d_vec(rel_traj_pos, -heading)

        # conds = dict()
        # #if self._use_heightmap:
        # if True:
        #     #hf = self._mgdm_hfs - self._mgdm_floor_heights.unsqueeze(-1)
        #     #hf = self._mgdm_hfs - canon_root_pos
        #     hf = self._mgdm_hfs

        #     # TODO: optionally force non zero values to be +0.6 or -0.6 if we are using different platform height

        #     hf = hf.view(self._num_envs, self._mgen._grid_dim_x, self._mgen._grid_dim_y)

        #     conds[MDMKeyType.OBS_KEY] = hf
        #     conds[MDMKeyType.OBS_FLAG_KEY] = self._true_tensor
        # else:
        #     zero_hf = torch.zeros(size=(self._num_envs, self._mgen._grid_dim_x, self._mgen._grid_dim_y), 
        #                           dtype=torch.float32, device=self._device)
        #     conds[MDMKeyType.OBS_KEY] = zero_hf
        # conds[MDMKeyType.PREV_STATE_KEY] = prev_state
        # conds[MDMKeyType.TARGET_KEY] = target_key
        # conds[MDMKeyType.TARGET_FLAG_KEY] = self._true_tensor
        # conds[MDMKeyType.PREV_STATE_NOISE_IND_KEY] = ~hard_reset_mask
        # conds[MDMKeyType.PREV_STATE_FLAG_KEY] = ~hard_reset_mask
        # #torch.ones(size=[self._num_envs], dtype=torch.bool, device=self._device)

        # if self._cfg_scale is not None:
        #     guidance_params = MDMCustomGuidance()
        #     guidance_params.obs_cfg_scale = self._cfg_scale
        #     guidance_params.verbose=False
        #     conds[MDMKeyType.GUIDANCE_PARAMS] = guidance_params

        # if self._ddim_stride is not None:
        #     output_motion_dict = self._mgen.gen_sequence_with_contacts(conds, ddim_stride=self._ddim_stride)
        # else:
        #     output_motion_dict = self._mgen.gen_sequence_with_contacts(conds)

        # TODO: don't hard code this
        # motion_sequence = mdm_sequence[..., 0:34]
        # contact_sequence = mdm_sequence[..., 34:49]

        output_root_pos = output_motion_frames.root_pos
        output_root_rot = output_motion_frames.root_rot
        output_joint_rot = output_motion_frames.joint_rot
        contact_sequence = output_motion_frames.contacts
        
        ## Uncanonicalization step ## 
        motion_sequence = torch.cat([output_root_pos, torch_util.quat_to_exp_map(output_root_rot),
                                     self._kin_char_model.rot_to_dof(output_joint_rot)], dim=-1)
        
        # gen_seq_len = motion_sequence.shape[1]
        # if num_hard_resets > 0:
        #     axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=self._device).unsqueeze(dim=0).expand(num_hard_resets, -1)
        #     new_root_rot = torch_util.axis_angle_to_quat(axis, new_heading)
            
        #     new_root_rot = new_root_rot.unsqueeze(dim=1).expand(-1, gen_seq_len, -1)
        #     gen_root_rot = torch_util.exp_map_to_quat(motion_sequence[hard_reset_ids, :, 3:6])
        #     motion_sequence[hard_reset_ids, :, 3:6] = torch_util.quat_to_exp_map(torch_util.quat_mul(new_root_rot, gen_root_rot))
        #     motion_sequence[hard_reset_ids, :, 0:3] = torch_util.quat_rotate(new_root_rot, motion_sequence[hard_reset_ids, :, 0:3])

        #     motion_sequence[hard_reset_ids, :, 0] += new_x_points.unsqueeze(dim=1)
        #     motion_sequence[hard_reset_ids, :, 1] += new_y_points.unsqueeze(dim=1)
        #     motion_sequence[hard_reset_ids, :, 0:2] -= self._env_offsets[hard_reset_ids, 0:2].unsqueeze(1)
        #     #motion_sequence[hard_reset_ids, :, 2] += canon_root_pos[hard_reset_ids, 2].unsqueeze(1)


        # if num_replans > 0:
        #     gen_root_rot = torch_util.exp_map_to_quat(motion_sequence[replan_ids, :, 3:6])
        #     replan_char_heading = torch_util.calc_heading_quat(canon_root_rot[replan_ids]) # need for uncanonicalization
        #     replan_char_heading = replan_char_heading.unsqueeze(dim=1).expand(-1, gen_seq_len, -1)
        #     #print(replan_char_root_rot.shape, gen_root_rot.shape)
        #     uncanon_root_rot = torch_util.quat_mul(replan_char_heading, gen_root_rot)
        #     motion_sequence[replan_ids, :, 3:6] = torch_util.quat_to_exp_map(uncanon_root_rot)
        #     motion_sequence[replan_ids, :, 0:3] = torch_util.quat_rotate(replan_char_heading, motion_sequence[replan_ids, :, 0:3])
        #     motion_sequence[replan_ids, :, 0:2] += canon_root_pos[replan_ids, 0:2].unsqueeze(dim=1) # unsqueeze frame index dim for broadcasting
        #     #motion_sequence[replan_ids, :, 0:3] += canon_root_pos[replan_ids].unsqueeze(dim=1)

        # uncanonicalize by floor height
        #motion_sequence[:, :, 2] += self._mgdm_floor_heights.unsqueeze(-1)
        
        self._motion_lib = motion_lib.MotionLib(motion_sequence, self._kin_char_model, self._device,
                                     init_type= "motion_frames", loop_mode=motion_lib.LoopMode.CLAMP, 
                                     fps=self._mgen._sequence_fps,
                                     contact_info=True, 
                                     contacts=contact_sequence)


        ## First fill in the history
        
        frame_zero = self._get_state_dict_from_motion_lib(0.0, self._motion_ids)

        self._ref_state_hist = frame_zero
        # ref_state_hist will get updated in update_misc(), which will get called after replan()
        # before the next call to reset, which is when its important

        if len(hard_reset_ids) > 0:
            ## Then get the first frame from the generated motion to fill in the hard reset char states
            motion_ids = self._motion_ids[hard_reset_ids]
            motion_times = torch.ones(motion_ids.shape, dtype=torch.float, device=self._device) * self._timestep

            root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, contacts = self._motion_lib.calc_motion_frame(motion_ids, motion_times)
            dof_pos = self._kin_char_model.rot_to_dof(joint_rot)

            self._char_rigid_body_vel[hard_reset_ids] = 0.0
            self._char_rigid_body_ang_vel[hard_reset_ids] = 0.0

            self._char_root_pos[hard_reset_ids] = root_pos
            self._char_root_rot[hard_reset_ids] = root_rot
            self._char_root_vel[hard_reset_ids] = root_vel
            self._char_root_ang_vel[hard_reset_ids] = root_ang_vel
            
            self._char_dof_pos[hard_reset_ids] = dof_pos
            self._char_dof_vel[hard_reset_ids] = dof_vel

            print(hard_reset_ids.shape)
            #frame_zero_state = frame_zero_state[hard_reset_ids]
            #frame_zero_state_dict

            self._agent_state_hist.set_vals(frame_zero, hard_reset_ids)
            #self._agent_state_hist[hard_reset_ids] = frame_zero_state

            self._actors_need_reset[hard_reset_ids, SIM_CHAR_IDX] = True
            
        # with 2 prev states, our character is initialized to timestep*1,
        # and our ref is initialized to timestep*1.
        # Obs and actions are computed after the replan. Then simulation happens which advances character by 1 timestep.
        # Then the actual timestep advance is recorded in update_time, then the ref motion is updated in update_misc()
        
        self._mgdm_time_buf[0] = self._timestep*(self._num_prev_states-1)
        self._replan_buf[:] = ReplanFlags.REPLAN.value
        self._replan_counter[replan_ids] += 1
        self._replan_counter[hard_reset_ids] = 1


        self._replan_flag = False

        self._update_ref_motion()
        return

    def get_target_dim(self):
        if self._mgen._target_type == mdm.TargetType.XY_POS:
            target_dim = 2
        elif self._mgen._target_type == mdm.TargetType.XY_POS_AND_HEADING:
            target_dim = 3
        elif self._mgen._target_type == mdm.TargetType.XY_DIR:
            target_dim = 2
        else:
            assert False

        return target_dim
    
    def get_mgdm_time_buf(self):
        return self._mgdm_time_buf
    
    def get_replan_counter(self):
        return self._replan_counter

    def apply_hard_reset(self):
        self._replan_buf[:] = ReplanFlags.HARD_RESET.value
        self._replan_flag = True
        return
    
    def hard_reset_char(self, reset_ids):

        num_resets = reset_ids.shape[0]
        motion_ids = self._hard_motion_lib.sample_motions(num_resets)
        motion_times = self._motion_lib.sample_time(motion_ids)
        
        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, contacts = self._hard_motion_lib.calc_motion_frame(motion_ids, motion_times)

        # TODO?


        return