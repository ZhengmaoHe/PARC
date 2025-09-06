import os
import polyscope as ps
import polyscope.imgui as psim
import numpy as np
import platform
import cpuinfo
if platform.system() == "Linux":
    import envs.env_builder as env_builder
    import learning.agent_builder as agent_builder
import torch
import trimesh
import yaml
import pickle
import time
from collections import OrderedDict

import util.torch_util as torch_util
import MOTION_FORGE.polyscope_util as ps_util
import anim.kin_char_model as kin_char_model
import anim.motion_lib as motion_lib
import zmotion_editing_tools.motion_edit_lib as medit_lib
import util.terrain_util as terrain_util
import util.geom_util as geom_util
import util.misc_util as misc_util
import util.motion_util as motion_util


import diffusion.mdm as mdm
from diffusion.diffusion_util import MDMKeyType, MDMCustomGuidance
from diffusion.mdm_heightfield_contact_motion_sampler import MDMHeightfieldContactMotionSampler, RelativeZStyle
import diffusion.gen_util as gen_util
import tools.procgen.mdm_path as mdm_path
import tools.procgen.astar as astar

from MOTION_FORGE.include.singleton import SingletonClass

try:
    with open("MOTION_FORGE/motion_forge_config.yaml", "r") as config_stream:
        input_config = yaml.safe_load(config_stream)
        g_motion_filepath = input_config["motion_filepath"]
        g_input_folder_dir = input_config["input_folder_dir"]
        LOAD_MDM = input_config["load_mdm"]
        LOAD_SAMPLER = input_config["load_sampler"]

        g_mdm_filepaths = input_config["mdm_filepaths"]
        SAMPLER_FILEPATH = input_config["sampler_filepath"]
        g_char_filepath = input_config["char_file"]

        g_other_motion_filepaths = input_config["other_motion_filepaths"]
        g_retargeting_cfg_path = input_config.get("retargeting_cfg", None)
        g_load_other_motion_terrains = input_config["load_other_motion_terrains"]

        if "motion_yaml_filepath" in input_config:
            g_motion_yaml_filepath = input_config["motion_yaml_filepath"]
        else:
            g_motion_yaml_filepath = None
except:
    assert False



class MainVars(SingletonClass):
    device = "cpu"
    selected_grid_ind = torch.tensor([0, 0], dtype=torch.int64, device=device)
    mouse_world_pos = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
    paused = False
    max_play_time = 100.0
    looping = True
    motion_time = 0.0
    use_contact_info = input_config["use_contact_info"]
    local_hf_visibility = 1.0
    viewing_local_hf = True
    viewing_prev_state = True
    viewing_char = True
    viewing_motion_sequence = True
    viewing_shadow = True
    using_sliders = True
    seq_vis_fps = 15
    mouse_size = 0
    mouse_ball_visible = True
    loaded_target_xy = False#"target_xy" in motion_data._data
    motion_dt = 1.0 / 30.0
    curr_time = 0.0
    curr_dt = 1.0 / 30.0
    visualize_body_points = False

    save_path_with_motion = False
    save_motion_as_loop = False
    save_terrain_with_motion = True

    time_gui_opened = True
    motion_gui_opened = False
    contact_gui_opened = False
    terrain_gui_opened = False
    mdm_gui_opened = False
    intersection_gui_opened = False
    optimization_gui_opened = False
    motion_matching_gui_opened = False
    nav_mesh_gui_opened = False
    path_planning_gui_opened = False
    recording_gui_opened = False
    retargeting_gui_opened = False
    ig_obs_gui_opened = False
    isaac_gym_gui_opened = False
    motion_graph_opened = False

    saved_cam_params = None

    def set_motion_time(self, val):
        assert isinstance(val, float)
        self.motion_time = val
        return

    def update_time(self):
        next_curr_time = time.time()
        self.curr_dt = next_curr_time - self.curr_time
        self.curr_time = next_curr_time
        return

## LOAD CHARACTER MODEL DATA
#g_char_model = kin_char_model.KinCharModel(MainVars().device)
#g_char_model.load_char_file(g_char_filepath)

## MOTION EDITING GLOBALS ##
class MotionEditorSettings(SingletonClass):
    num_blend_frames = 5
    medit_heading_rot_angle = 0.0
    medit_start_frame = 0
    medit_end_frame = 5
    medit_scale = 0.9
    load_terrain_with_motion = True
    motion_slice_start_time = 0.0
    motion_slice_end_time = 5.0
    editing_full_sequence = True
    speed_scale = 0.8

    # TODO: move somewhere else
    x_scale = 1.0
    y_scale = 1.0
    z_scale = 1.0

    halflife_start = 0.1
    halflife_end = 0.1
    ratio = 0.5
    blendtime = 0.4

## TERRAIN EDITING GLOBALS ##
class TerrainEditorSettings(SingletonClass):
    height = 0.6
    edit_modes = ["heightfield", "mask", "max", "min"]
    curr_edit_mode = 0
    mask_mode = True
    viewing_terrain = True
    viewing_mask = True
    viewing_max = False
    viewing_min = False
    terrain_padding = 0.4 # meters
    num_boxes = 10
    max_num_boxes = 8
    min_num_boxes = 2
    box_max_len = 10
    box_min_len = 5
    maxpool_size = 1
    box_heights = 0.6
    max_box_angle = 0.0 #2.0 * torch.pi
    min_box_angle = 0.0
    min_box_h = -2.0
    max_box_h = 2.0
    use_maxmin = False
    slice_min_i = 0
    slice_min_j = 0
    slice_max_i = 0
    slice_max_j = 0
    new_terrain_dim_x = 16
    new_terrain_dim_y = 16
    num_terrain_paths = 4
    path_min_height = -2.8
    path_max_height = 3.0
    floor_height = -3.0

    min_stair_start_height = -3.0
    max_stair_start_height = 1.0
    min_step_height = 0.15
    max_step_height = 0.25
    num_stairs = 4
    min_stair_thickness = 2.0
    max_stair_thickness = 8.0

class PathPlanningSettings(SingletonClass):
    waypoints = []
    waypoints_ps = []
    manual_placement_mode = False

    extend_path_mode = False

    astar_settings = astar.AStarSettings()

    path_nodes = None

    mdm_path_settings = mdm_path.MDMPathSettings()

    use_prev_frames = False

    curviness = 0.5

    def clear_waypoints(self):

        for waypoint_ps in self.waypoints_ps:
            waypoint_ps.remove()

        self.waypoints_ps.clear()
        self.waypoints.clear()

        return

    def place_waypoint(self, node: torch.Tensor, color = None):
        z = g_terrain.get_hf_val(node)

        world_pos = g_terrain.get_point(node)
        world_pos = torch.cat([world_pos, z.unsqueeze(0)]).cpu().numpy()

        sphere_trimesh = trimesh.primitives.Sphere(0.1)

        if color is None:
            color = [0.1, 1.0, 0.1]

        # TODO: use a number texture?
        name = "PathWaypoint" + str(len(self.waypoints))
        sphere_ps_mesh = ps.register_surface_mesh(name, sphere_trimesh.vertices, sphere_trimesh.faces)
        sphere_ps_mesh.set_color(color)
        transform = np.eye(4)
        transform[:3, 3] = world_pos
        sphere_ps_mesh.set_transform(transform)

        self.waypoints.append(node.cpu().numpy())
        self.waypoints_ps.append(sphere_ps_mesh)
        return
    
    def visualize_path_nodes(self, ps_name: str, nodes_3dpos: torch.Tensor):
        global g_terrain
        edges = []
        for i in range(nodes_3dpos.shape[0] - 1):
            edges.append([i, i+1])
        edges = np.array(edges)
        # heights = g_terrain.hf[nodes[:, 0], nodes[:, 1]]
        # xy_points = g_terrain.get_point(nodes)
        # xyz_nodes = torch.cat([xy_points, heights.unsqueeze(-1)], axis=-1)
        ps.register_curve_network(ps_name, nodes=nodes_3dpos.cpu().numpy(), edges=edges, 
                                    enabled=True, color=[1.0, 0.0, 0.0], radius=0.0014)
        return

# MDM
# note: we can load these from the mdm using mdm._cfg["heightmap"]["local_grid"]
g_mdm_model = None
class MDMSettings(SingletonClass):
    loaded_mdm_models = dict()
    current_mdm_key = "main"
    
    append_mdm_motion_to_prev_motion = False
    prev_sampled_motion_ID = None
    prev_sampled_motion_start_time = None
    resample_prev_motion = False
    #cfg_scale = 0.65
    #use_cfg = True
    mdm_batch_size = 1
    local_hf_num_neg_x = 8
    local_hf_num_pos_x = 8
    local_hf_num_neg_y = 8
    local_hf_num_pos_y = 8
    sample_prev_states_only = False
    attention_debug = False
    hide_batch_motions = True
    #relative_z_style = RelativeZStyle.RELATIVE_TO_ROOT

    conv_layer_num = 0
    gen_settings = gen_util.MDMGenSettings()

    def select_mdm_helper(self, key) -> mdm.MDM:
        return self.loaded_mdm_models[key]

    def select_mdm(self, key):
        global g_mdm_model, g_motion, g_terrain
        self.current_mdm_key = key
        g_mdm_model = self.select_mdm_helper(key)
        MDMSettings().local_hf_num_neg_x = g_mdm_model._num_x_neg
        MDMSettings().local_hf_num_pos_x = g_mdm_model._num_x_pos
        MDMSettings().local_hf_num_neg_y = g_mdm_model._num_y_neg
        MDMSettings().local_hf_num_pos_y = g_mdm_model._num_y_pos

        # TODO: reconstruct char local hf
        MotionManager().get_curr_motion().char.reconstruct_local_hf_points(g_mdm_model)

        MainVars().motion_dt = 1.0 / g_mdm_model._sequence_fps
        MotionManager().get_curr_motion().char.update_local_hf(g_terrain)
        self.gen_settings.starting_diffusion_timestep = g_mdm_model._diffusion_timesteps
        return

class IntersectionSettings(SingletonClass):
    intersecting = False
    check_intersection_with_local_hf = False
    use_interior_sdf = True
    
class MotionMatchingSettings(SingletonClass):
    motion_database = None
    w_pose = 0.2
    w_xy = 0.8
    mm_time_offsets = [1.0/6.0, 2.0/6.0, 3.0/6.0, 4.0/6.0, 5.0/6.0, 6.0/6.0,
                        7.0/6.0, 8.0/6.0, 9.0/6.0]
    append_mm_to_prev_motion = False
    blend_mm_motions = True
    contact_matching = False
    num_mm_blend_frames = 3

class ContactEditingSettings(SingletonClass):
    contact_eps = 1e-2
    selected_body_id = 0
    start_frame_idx=0
    end_frame_idx=0

class OptimizationSettings(SingletonClass):
    step_size = 1e-3
    num_iters = 1000
    w_root_pos = 1.0
    w_root_rot = 10.0
    w_joint_rot = 1.0
    w_smoothness = 10.0
    w_penetration = 1000.0
    w_contact = 1000.0
    w_sliding = 10.0
    w_body_constraints = 1000.0
    w_jerk = 100.0
    max_jerk = 1000.0
    body_constraints = None
    body_constraints_ps_meshes = OrderedDict()
    use_wandb = True
    auto_compute_body_constraints = False

    # TODO: separate body constraint dict for different motions

    def create_body_constraint_ps_mesh(self, body_id, start_frame_idx, end_frame_idx, position, 
                                       char_model: kin_char_model.KinCharModel):
        sphere_trimesh = trimesh.primitives.Sphere(0.025)

        body_name = char_model.get_body_name(body_id)
        name_str = body_name + ":" + str(start_frame_idx) + "->" + str(end_frame_idx) 
        sphere_ps_mesh = ps.register_surface_mesh(name_str, sphere_trimesh.vertices, sphere_trimesh.faces)
        sphere_ps_mesh.set_color([0.0, 0.0, 1.0])
        transform = np.eye(4)
        transform[:3, 3] = position
        sphere_ps_mesh.set_transform(transform)
        self.body_constraints_ps_meshes[name_str] = sphere_ps_mesh
        return
    
    def create_body_constraint_ps_meshes(self):
        if self.body_constraints is not None:
            for b in range(len(self.body_constraints)):
                for c_idx in range(len(self.body_constraints[b])):
                    curr_body_constraint = self.body_constraints[b][c_idx]
                    self.create_body_constraint_ps_mesh(b, curr_body_constraint.start_frame_idx, 
                                                        curr_body_constraint.end_frame_idx,
                                                        curr_body_constraint.constraint_point.cpu(),
                                                        MotionManager().get_curr_motion().char.char_model)
        return
    
    def clear_body_constraints(self):
        if self.body_constraints is not None:
            for b in range(len(self.body_constraints)):
                self.body_constraints[b] = []
            
            for key, ps_mesh in self.body_constraints_ps_meshes.items():
                ps_mesh.remove()
            self.body_constraints_ps_meshes.clear()
    
class RetargetingSettings(SingletonClass):
    cfg_loaded = False
    src_motion_name = None
    tgt_motion_name = None
    def load_config(self, cfg):
        self.device = cfg["device"]
        self.src_scale = cfg["src_scale"]
        self.src_motion_rot = cfg["src_motion_rot"]
        self.w_root_pos = cfg["w_root_pos"]
        self.w_root_rot = cfg["w_root_rot"]
        self.w_joint_rot = cfg["w_joint_rot"]
        self.w_key_pos = cfg["w_key_pos"]
        self.w_temporal = cfg["w_temporal"]
        self.lr = cfg["lr"]
        self.num_iters = cfg["num_iters"]
        self.tar_fps = cfg["tar_fps"]
        self.src_t_pose = torch.tensor(cfg["src_t_pose"], dtype=torch.float32, device=MainVars().device)
        self.tgt_t_pose = torch.tensor(cfg["tgt_t_pose"], dtype=torch.float32, device=MainVars().device)
        self.cfg_loaded = True
        self.cfg = cfg
        self.key_body_correspondences = cfg["key_body_correspondences"]

        self.src_char_model = kin_char_model.KinCharModel(device=MainVars().device)
        self.src_char_model.load_char_file(cfg["src_char_file"])
        self.tgt_char_model = kin_char_model.KinCharModel(device=MainVars().device)
        self.tgt_char_model.load_char_file(cfg["tgt_char_file"])
        return

class RecordingSettings(SingletonClass):
    """
    Can hard code camera jsons here for recording comparison videos
    """

    # present_terrain
    view_json = '{"farClipRatio":20.0,"fov":45.0,"nearClipRatio":0.005,"projectionMode":"Perspective","viewMat":[0.928424537181854,-0.37151899933815,-1.87303297871644e-10,-5.68922710418701,0.106584832072258,0.266355097293854,0.957963883876801,-1.2079918384552,-0.355901926755905,-0.889398097991943,0.286888986825943,-9.89432144165039,0.0,0.0,0.0,1.0],"windowHeight":1016,"windowWidth":1860}'
    
    # present terrain other angle
    view_json2 = '{"farClipRatio":20.0,"fov":45.0,"nearClipRatio":0.005,"projectionMode":"Perspective","viewMat":[-0.954478859901428,-0.298272997140884,1.64057445406485e-09,5.38498306274414,0.112148329615593,-0.35887685418129,0.92662388086319,-0.525702238082886,-0.276387631893158,0.88444459438324,0.37599128484726,-12.1192407608032,0.0,0.0,0.0,1.0],"windowHeight":1016,"windowWidth":1860}'

    root_pos_spacing = 0.8

class IGObsSettings(SingletonClass):
    has_obs = False
    overlay_obs_on_motion = False
    view_tar_obs = True
    view_key_points = True
    record_obs = False

    def setup(self, obs, obs_shapes):
        self.has_obs = True
        self.obs = obs
        self.obs_shapes = obs_shapes

        self.char_obs, self.tar_obs, self.tar_contacts, self.char_contacts, self.hf_obs_points = extract_obs(self.obs, self.obs_shapes)

        ps.register_point_cloud("hf_obs_points", self.hf_obs_points[0], radius=0.001)

        self.hf_obs_points = torch.from_numpy(self.hf_obs_points).to(dtype=torch.float32, device=MainVars().device)

        self.root_pos_tar_obs, self.root_rot_tar_obs, self.joint_rot_tar_obs, self.key_pos_tar_obs = misc_util.inverse_tar_obs(torch.from_numpy(self.tar_obs))

        self.num_tar_obs = 6 # TODO dont hardcode
        self.ps_tar_char_meshes = []
        self.tar_char_color = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        for i in range(self.num_tar_obs):
            ps_char_meshes = ps_util.create_char_mesh("tar_obs_" + str(i), color=self.tar_char_color, transparency=0.3, char_model=MotionManager().get_curr_motion().mlib._kin_char_model)
            self.ps_tar_char_meshes.append(ps_char_meshes)

        num_frames = self.key_pos_tar_obs.shape[0]
        self.key_pos_tar_obs = self.key_pos_tar_obs.view(num_frames, -1, 3)
        self.ps_tar_key_pos_pc = ps.register_point_cloud("tar_obs_key_pos", points = self.key_pos_tar_obs[0].numpy(),
                                                            color=[0.0, 0.0, 1.0], transparency=0.6, radius=0.003)


        self.proprio_char_obs = misc_util.extract_obs(torch.from_numpy(self.char_obs), False)
        self.proprio_char_color = np.array([0.9, 0.9, 0.9], dtype=np.float32)
        self.ps_proprio_char_mesh = ps_util.create_char_mesh("char_obs" + str(i), color=self.proprio_char_color, transparency=1.0, char_model=MotionManager().get_curr_motion().mlib._kin_char_model)
        self.proprio_key_pos_obs = self.proprio_char_obs["key_pos"].view(num_frames, -1, 3)
        self.ps_proprio_key_pos_pc = ps.register_point_cloud("proprio_key_pos", points=self.proprio_key_pos_obs[0].numpy(), 
                                                             color=[0.0, 0.7, 1.0], transparency=0.8, radius=0.004)
        return
    
    def SetViewTarObs(self, val):
        for i in range(self.num_tar_obs):
            for j in range(len(self.ps_tar_char_meshes[i])):
                self.ps_tar_char_meshes[i][j].set_enabled(val)
        self.ps_tar_key_pos_pc.set_enabled(val)
        return


class IsaacGymManager(SingletonClass):
    env = None
    agent = None
    device = None
    paused = True
    ps_char = None
    recorded_frames = []
    is_recording = False
    is_closed_loop_generating = False
    replan_time = 1.0
    num_replan_loops = 4
    start_time_fraction = 0.0

    def start_isaac_gym(self, env_file, num_envs=1, device="cuda:0", visualize=False):
        if platform.system() == "Linux":
            self.env = env_builder.build_env(env_file, num_envs=num_envs, device=device, visualize=visualize)
            self.device = device
        return
    
    def load_agent(self, agent_file, model_file):
        if platform.system() == "Linux":
            self.agent = agent_builder.build_agent(agent_file, self.env, self.device)
            self.agent.load(model_file)
            self.agent.eval_mode()
            self.agent.hard_reset_envs()
        return
    
    def create_MOTION_FORGE_character(self):
        char_model = self.env._kin_char_model.get_copy(MainVars.device)
        self.ps_char = ps_util.CharacterPS("isaac_gym_sim_char", color=[0.9, 0.9, 1.0], char_model=char_model)


        ## OBSERVATIONS ##

        self.obs_shapes = self.env._compute_obs(ret_obs_shapes=True)

        self.obs_slices = dict()

        curr_idx = 0
        for key in self.obs_shapes:
            print(key)
            print(self.obs_shapes[key])

            flat_shape = self.obs_shapes[key]["shape"].numel()
            self.obs_slices[key] = slice(curr_idx, curr_idx + flat_shape)
            curr_idx += flat_shape

        #tar_obs = self.env._obs_buf
        #self.root_pos_tar_obs, self.root_rot_tar_obs, self.joint_rot_tar_obs, self.key_pos_tar_obs = misc_util.inverse_tar_obs(torch.from_numpy(self.tar_obs))

        self.num_tar_obs = self.obs_shapes["tar_obs"]["shape"][0]
        self.ps_tar_char_meshes = []
        self.tar_char_color = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        for i in range(self.num_tar_obs):
            ps_char_meshes = ps_util.create_char_mesh("ig_tar_obs_" + str(i), color=self.tar_char_color, transparency=0.3, char_model=char_model)
            self.ps_tar_char_meshes.append(ps_char_meshes)
        return
    
    def is_ready(self):
        return self.env != None and self.agent != None
    
    def reset(self):
        self.recorded_frames.clear()
        self.env.reset()
        if self.is_recording:
            self.record_char_state()
        return

    def reset_to_time(self, time: float):
        self.env.get_dm_env()._rand_reset = False
        self.env.get_dm_env()._motion_start_time_fraction[:] = time
        self.reset()
        self.env.get_dm_env()._motion_start_time_fraction[:] = 0.0
        self.env.get_dm_env()._rand_reset = True
        return
    
    def reset_to_frame_0(self):
        self.env.get_dm_env()._rand_reset = False
        self.reset()
        self.env.get_dm_env()._rand_reset = True
        return
    
    def step(self):
        self.agent.step()

        ## VISUALIZATION CODE ##
        if self.ps_char is not None:
            global_root_pos = self.env._char_root_pos[0].cpu()
            global_root_rot = self.env._char_root_rot[0].cpu()

            sim_dof = self.env._char_dof_pos[0].cpu()
            joint_rot = self.ps_char.char_model.dof_to_rot(sim_dof)

            shadow_height = g_terrain.get_hf_val_from_points(global_root_pos[0:2])
            self.ps_char.forward_kinematics(root_pos=global_root_pos, root_rot=global_root_rot, joint_rot=joint_rot, shadow_height=shadow_height)

            global_heading_quat = torch_util.calc_heading_quat(global_root_rot)
            
            obs = self.env._obs_buf.cpu()

            tar_obs = obs[0, self.obs_slices["tar_obs"]].reshape(self.num_tar_obs, -1)

            tar_root_pos, tar_root_rot, tar_joint_rot, tar_key_pos = misc_util.inverse_tar_obs(tar_obs)

            for idx in range(self.num_tar_obs):
                root_pos = torch_util.quat_rotate(global_heading_quat, tar_root_pos[idx]) + global_root_pos
                root_rot = torch_util.quat_multiply(global_heading_quat, tar_root_rot[idx])
                joint_rot = tar_joint_rot[idx]
                body_pos, body_rot = self.ps_char.char_model.forward_kinematics(root_pos, root_rot, joint_rot)
                ps_util.update_char_motion_mesh(body_pos, body_rot, self.ps_tar_char_meshes[idx], self.ps_char.char_model)

                # for body_id in range(0, char_model.get_num_joints()):
                #     body_contact = tar_contacts[round_frame_idx, i, body_id].item()
                #     ig_obs.ps_tar_char_meshes[i][body_id].set_color(red * body_contact + ig_obs.tar_char_color * (1.0 - body_contact))
        
        if self.is_recording:
            self.record_char_state()

            if self.is_closed_loop_generating:
                dm_env = self.env.get_dm_env()
                motion_time = dm_env._get_motion_times(0)

                if motion_time > self.replan_time:
                    prev_frames = self.recorded_frames[-2:]
                    prev_frames = motion_util.cat_motion_frames(prev_frames)

                    path_nodes = PathPlanningSettings().path_nodes.to(device=self.device)

                    mdm_path_settings = PathPlanningSettings().mdm_path_settings
                    mdm_gen_settings = MDMSettings().gen_settings
                    #mdm_gen_settings.ddim_stride = 100

                    total_gen_motion_frames = []
                    for i in range(self.num_replan_loops):
                        gen_motion_frames, done = mdm_path.generate_frames_along_path(prev_frames=prev_frames,
                                                                path_nodes_xyz=path_nodes,
                                                                terrain=self.env.get_dm_env()._terrain,
                                                                char_model=self.env._kin_char_model,
                                                                mdm_model=g_mdm_model,
                                                                mdm_settings=mdm_gen_settings,
                                                                path_settings=mdm_path_settings,
                                                                verbose=False)
                        
                        total_gen_motion_frames.append(gen_motion_frames)
                        num_gen_frames = gen_motion_frames.root_pos.shape[1]
                        prev_frames = gen_motion_frames.get_slice(slice(num_gen_frames-2, num_gen_frames))

                    
                    total_gen_motion_frames = motion_util.cat_motion_frames(total_gen_motion_frames)
                    mlib_motion_frames, mlib_contact_frames = total_gen_motion_frames.get_mlib_format(self.env._kin_char_model)

                    mlib_motion_frames = mlib_motion_frames.squeeze(0)
                    mlib_contact_frames = mlib_contact_frames.squeeze(0)
                    new_mlib = motion_lib.MotionLib(mlib_motion_frames, 
                                        self.env._kin_char_model, 
                                        device=self.device, 
                                        init_type="motion_frames",
                                        loop_mode=motion_lib.LoopMode.CLAMP,
                                        fps=30,
                                        contact_info=MainVars().use_contact_info,
                                        contacts=mlib_contact_frames)
                    
                    dm_env._motion_lib = new_mlib

                    dm_env._motion_time_offsets[0] = 1.0 / 30.0
                    dm_env._time_buf[0] = 0.0
                    dm_env._timestep_buf[0] = 0
        return
    
    def loop_function(self):
        self.step()
        return
    
    def compute_critic_value(self):
        norm_obs = self.agent._obs_norm.normalize(self.agent._curr_obs)
        critic_val = self.agent._model.eval_critic(norm_obs)

        return critic_val.item()
    
    def record_char_state(self):
        root_pos = self.env._char_root_pos[0].clone()
        root_rot = self.env._char_root_rot[0].clone()
        joint_dof = self.env._char_dof_pos[0].clone()
        joint_rot = self.env._kin_char_model.dof_to_rot(joint_dof)
        char_contacts = torch.norm(self.env._char_contact_forces[0], dim=-1)
        char_contacts = char_contacts > 0.001
        char_contacts = char_contacts.to(dtype=torch.float32)

        curr_frame = motion_util.MotionFrames(root_pos=root_pos, root_rot=root_rot, joint_rot=joint_rot, contacts=char_contacts)

        self.recorded_frames.append(curr_frame.unsqueeze(0).unsqueeze(0))
        return
    
    def create_motion_from_recorded_frames(self):

        if len(self.recorded_frames) < 2:
            print("not enough recorded frames")
            return
        
        motion_frames = motion_util.cat_motion_frames(self.recorded_frames)

        mlib_motion_frames, mlib_contact_frames = motion_frames.get_mlib_format(self.env._kin_char_model)
        mlib_motion_frames = mlib_motion_frames.to(device=MainVars().device)
        mlib_contact_frames = mlib_contact_frames.to(device=MainVars().device)

        MotionManager().make_new_motion(mlib_motion_frames, mlib_contact_frames, "dm_recorded_motion", motion_fps=30, vis_fps=5)
        return


ps.set_up_dir("z_up")
ps.set_front_dir("neg_y_front")
ps.set_ground_plane_mode("none")
ps.set_background_color([0.0, 0.0, 0.0])
ps.set_program_name("Motion Forge")
ps.init()
ps_util.create_origin_axis_mesh()

def _load_input_folder_dir():
    global g_motion_filepaths, g_input_folder_dir
    
    if os.path.isdir(g_input_folder_dir):
        files = os.listdir(g_input_folder_dir)
        g_motion_filepaths = [os.path.join(g_input_folder_dir, f) for f in files if os.path.splitext(f)[1] == ".pkl"]
        g_motion_filepaths.sort()
    else:
        g_motion_filepaths = []
    return
_load_input_folder_dir()

g_motion_data = medit_lib.load_motion_file(g_motion_filepath)
g_motion_data.set_device(MainVars().device)
MainVars().motion_dt = 1.0 / g_motion_data.get_fps()

## LOAD MDM
def load_mdm(mdm_path) -> mdm.MDM:
    if LOAD_MDM:
        with open(mdm_path, 'rb') as input_filestream:
            ret_mdm = pickle.load(input_filestream)
            if ret_mdm.use_ema:
                print('Using EMA model...')
                ret_mdm._denoise_model = ret_mdm._ema_denoise_model
            ret_mdm.update_old_mdm()

            print("MDM uses heightmap: ", ret_mdm._use_heightmap_obs)
            print("MDM uses target: ", ret_mdm._use_target_obs)
        return ret_mdm
    else:
        return None

for key in g_mdm_filepaths:
    MDMSettings().loaded_mdm_models[key] = load_mdm(g_mdm_filepaths[key])


def load_mdm_sampler(sampler_config_path) ->  MDMHeightfieldContactMotionSampler:
    with open(sampler_config_path, "r") as stream:
        gen_config = yaml.safe_load(stream)
    gen_config["device"] = "cpu"
    motion_sampler = MDMHeightfieldContactMotionSampler(gen_config)
    with open(SAMPLER_FILEPATH, "wb") as stream:
        pickle.dump(motion_sampler, stream)
    return motion_sampler

def load_mdm_sampler_pkl(sampler_path) -> MDMHeightfieldContactMotionSampler:
    with open(sampler_path, "rb") as stream:
        motion_sampler = pickle.load(stream)
    return motion_sampler

def no_sampler() -> MDMHeightfieldContactMotionSampler:
    return None

if LOAD_SAMPLER:
    print("loading sampler...")
    if os.path.exists(SAMPLER_FILEPATH):
        g_sampler = load_mdm_sampler_pkl(SAMPLER_FILEPATH)
    else:
        g_sampler = load_mdm_sampler("diffusion/mdm.yaml")
    print("sampler loaded")
else:
    g_sampler = no_sampler()


class TerrainMeshManager(SingletonClass):
    def reset(self):
        self.build_ps_hf_mesh()
        self.hf_ps_mesh.set_enabled(TerrainEditorSettings().viewing_terrain)
        self.build_ps_mask_mesh()
        self.hf_mask_mesh.set_enabled(TerrainEditorSettings().viewing_mask)
        self.build_ps_max_mesh()
        self.hf_max_mesh.set_enabled(TerrainEditorSettings().viewing_max)
        self.build_ps_min_mesh()
        self.hf_min_mesh.set_enabled(TerrainEditorSettings().viewing_min)
        return
    
    def rebuild(self):
        self.build_ps_hf_mesh()
        self.build_ps_mask_mesh()
        self.build_ps_max_mesh()
        self.build_ps_min_mesh()
        return

    def soft_rebuild(self):
        self.build_ps_hf_mesh()
        self.build_ps_mask_mesh()
        return

    def build_ps_hf_mesh(self):
        global g_terrain
        verts, tris = terrain_util.convert_heightfield_to_voxelized_trimesh(
            g_terrain.hf, 
            g_terrain.min_point[0].item(), 
            g_terrain.min_point[1].item(),
            g_terrain.dxdy[0].item())
        
        hf_trimesh = trimesh.Trimesh(vertices=verts, faces=tris)
        
        info = cpuinfo.get_cpu_info()
        if "intel" in info['brand_raw'].lower():
            hf_embree = trimesh.ray.ray_pyembree.RayMeshIntersector(hf_trimesh)
        else:
            hf_embree = trimesh.ray.ray_triangle.RayMeshIntersector(hf_trimesh)

        hf_ps_mesh = ps.register_surface_mesh("heightfield", verts, tris)

        heights = verts[..., 2]
        max_h = np.max(heights)
        min_h = np.min(heights)
        if max_h > min_h + 1e-3:
            heights = (heights - min_h) / (max_h - min_h)
            hf_ps_mesh.add_scalar_quantity("height", heights, enabled=True)
        else:
            hf_ps_mesh.set_color([0.5, 0.5, 0.5])
        self.hf_ps_mesh = hf_ps_mesh
        self.hf_embree = hf_embree
        return

    def build_ps_mask_mesh(self):
        global g_terrain
        verts, tris = terrain_util.convert_hf_mask_to_flat_voxels(
            g_terrain.hf_mask, g_terrain.hf, 
            g_terrain.min_point[0].item(), 
            g_terrain.min_point[1].item(),
            g_terrain.dxdy[0].item(),
            voxel_w_scale=0.8)
        hf_mask_ps_mesh = ps.register_surface_mesh("heightfield mask", verts, tris)
        hf_mask_ps_mesh.set_color([0.4, 0.3, 0.7])
        hf_mask_ps_mesh.set_transparency(0.5)
        self.hf_mask_mesh = hf_mask_ps_mesh
        return

    def build_ps_max_mesh(self):
        global g_terrain
        true_hf = torch.ones_like(g_terrain.hf_mask, dtype=torch.bool)
        verts, tris = terrain_util.convert_hf_mask_to_flat_voxels(
            true_hf, g_terrain.hf_maxmin[..., 0], 
            g_terrain.min_point[0].item(), 
            g_terrain.min_point[1].item(),
            g_terrain.dxdy[0].item(),
            voxel_w_scale=0.8)
        hf_max_ps_mesh = ps.register_surface_mesh("heightfield max", verts, tris)
        hf_max_ps_mesh.set_color([1.0, 0.2, 0.2])
        hf_max_ps_mesh.set_transparency(0.2)
        self.hf_max_mesh = hf_max_ps_mesh
        return

    def build_ps_min_mesh(self):
        global g_terrain
        true_hf = torch.ones_like(g_terrain.hf_mask, dtype=torch.bool)
        verts, tris = terrain_util.convert_hf_mask_to_flat_voxels(
            true_hf, g_terrain.hf_maxmin[..., 1], 
            g_terrain.min_point[0].item(), 
            g_terrain.min_point[1].item(),
            g_terrain.dxdy[0].item(),
            voxel_w_scale=0.8)
        hf_min_ps_mesh = ps.register_surface_mesh("heightfield min", verts, tris)
        hf_min_ps_mesh.set_color([0.2, 0.2, 1.0])
        hf_min_ps_mesh.set_transparency(0.2)
        self.hf_min_mesh = hf_min_ps_mesh
        return



def create_mouse_ball_ps_meshes(size):

    dim = size * 2 + 1
    mouse_ball_trimesh = trimesh.primitives.Sphere(0.05)

    meshes = []

    for i in range(dim):
        for j in range(dim):
            name_str = "mouse " + str(i) + "," + str(j)
            mouse_ball_ps_mesh = ps.register_surface_mesh(name_str, mouse_ball_trimesh.vertices,
                                                        mouse_ball_trimesh.faces)
            mouse_ball_ps_mesh.set_color([1.0, 0.0, 0.0])
            meshes.append(mouse_ball_ps_mesh)
    return meshes

g_mouse_ball_meshes = create_mouse_ball_ps_meshes(MainVars().mouse_size)


def update_mouse_ball_ps_meshes(size):
    global g_mouse_ball_meshes
    for mesh in g_mouse_ball_meshes:
        ps.remove_surface_mesh(mesh.get_name())
    g_mouse_ball_meshes = create_mouse_ball_ps_meshes(size)
    return

def build_selected_pos_flag_mesh() -> ps.SurfaceMesh:

    flag_mesh = ps_util.create_vector_mesh([0.0, 0.0, 1.0], name="selected pos flag", color=[1.0, 0.0, 1.0])

    return flag_mesh
g_flag_mesh = build_selected_pos_flag_mesh()

def update_selected_pos_flag_mesh(xyz: torch.Tensor):
    global g_flag_mesh

    transform = np.eye(4)
    transform[:3, 3] = xyz.cpu().numpy()
    g_flag_mesh.set_transform(transform)
    return

# def update_mdm_char_frames(mdm_mlib, mdm_char_bodies):
#     for i in range(15):
#         body_pos, body_rot = ps_util.compute_body_transforms_from_mlib(i / 15.0, mdm_mlib)
#         ps_util.update_char_motion_mesh(body_pos, body_rot, mdm_char_bodies[i], MainVars().char_model)
#     return

class MDMCharacterPS(ps_util.CharacterPS):
    def __init__(self, name, color, char_model: kin_char_model.KinCharModel, history_length = 1,
                 x_scale: float = 1.0, y_scale: float = 1.0, z_scale: float = 1.0):
        global g_terrain, g_mdm_model
        super().__init__(name, color, char_model, history_length, 
                         x_scale=x_scale, y_scale=y_scale, z_scale=z_scale)
        
        self.reconstruct_local_hf_points(g_mdm_model)
        
        self.center_ind = [2, 20]
        
        self.hf_z = None
        self._hf_xyz_points = None

        self._ps_body_point_samples = None
        self._body_points = None

        #all_points = geom_util.get_char_point_samples(char_model)
        # TODO: apply scales to this i guess?
        #all_points = geom_util.get_minimal_char_point_samples(char_model)
        all_points = geom_util.get_char_point_samples(char_model)
        self.init_body_point_samples(all_points)

        
        return
    
    def reconstruct_local_hf_points(self, mdm_model):
        if mdm_model is not None:
            dx = mdm_model._dx
            dy = mdm_model._dy
        else:
            dx = 0.4
            dy = 0.4
        self.dx=dx
        self.dy=dy
        self.num_neg_x = MDMSettings().local_hf_num_neg_x
        self.num_neg_y = MDMSettings().local_hf_num_neg_y
        self.local_hf_points = geom_util.get_xy_grid_points(
            center=torch.zeros(size=(2,), dtype=torch.float32, device=MainVars().device),
            dx=dx,
            dy=dy,
            num_x_neg=MDMSettings().local_hf_num_neg_x,
            num_x_pos=MDMSettings().local_hf_num_pos_x,
            num_y_neg=MDMSettings().local_hf_num_neg_y,
            num_y_pos=MDMSettings().local_hf_num_pos_y)
        return
    
    def get_global_hf_xy(self, terrain: terrain_util.SubTerrain):
        assert isinstance(terrain.hf, torch.Tensor)
        heading = torch_util.calc_heading(self.get_body_rot(0))
        xy_points = torch_util.rotate_2d_vec(self.local_hf_points, heading) + self.get_body_pos(0)[0:2]

        return xy_points
    
    def update_local_hf(self, terrain: terrain_util.SubTerrain, hf_z = None):
        xy_points = self.get_global_hf_xy(terrain)

        #inds = terrain.get_grid_index(xy_points)
        #self.hf_z = terrain.hf[inds[..., 0], inds[..., 1]]
        if hf_z is None:
            self.hf_z = terrain.get_hf_val_from_points(xy_points)
        else:
            self.hf_z = hf_z


        xyz_points = torch.cat([xy_points, self.hf_z.unsqueeze(-1)], dim=-1).view(-1, 3)
        self._xyz_points = xyz_points.cpu().numpy()

        self.update_local_hf_ps()
        return
    
    def update_local_hf_ps(self):
        local_hf_ps_name = self.name + " local hf"
        self._local_hf_ps = ps.register_point_cloud(local_hf_ps_name, self._xyz_points)
        self._local_hf_ps.set_radius(0.001)
        self._local_hf_ps.set_color([0.8, 0.0, 0.0])


        N, M = self.hf_z.shape
    
        # Create range of indices for rows and columns
        row_indices = torch.arange(N)
        col_indices = torch.arange(M)
        
        # Create a meshgrid of row and column indices
        rows, cols = torch.meshgrid(row_indices, col_indices, indexing='ij')

        self._local_hf_ps.add_scalar_quantity("i", values=rows.reshape(-1))
        self._local_hf_ps.add_scalar_quantity("j", values=cols.reshape(-1))

        return
    
    def set_local_hf_transparency(self, val):
        self._local_hf_ps.set_transparency(val)
        return
    
    def set_local_hf_enabled(self, val):
        self._local_hf_ps.set_enabled(val)
        return
    
    def get_hf_below_root(self, terrain: terrain_util.SubTerrain):
        ind = terrain.get_grid_index(self.get_body_pos(0)[0:2])
        z = terrain.hf[ind[0], ind[1]].item()
        return z
    
    def remove(self):
        super().remove()
        self._local_hf_ps.remove()
        return
    
    def set_enabled(self, val):
        #self._local_hf_ps.set_enabled(val)
        #if val is False:
        #    self.set_body_points_enabled(val)
        super().set_enabled(val)
        return
    
    def deselect(self):
        self._local_hf_ps.set_enabled(False)
        return
    
    def select(self, viewing_local_hf, viewing_shadow):
        self._local_hf_ps.set_enabled(viewing_local_hf)
        self.set_shadow_enabled(viewing_shadow)
        return
    
    def has_body_point_samples(self):
        return self._body_points is not None
    
    def get_body_point_samples(self, device=None):
        if device is None:
            return self._body_points
        else:
            ret_body_points = []
            for b in range(len(self._body_points)):
                ret_body_points.append(self._body_points[b].to(device=device))
            return ret_body_points
        
    def get_transformed_body_point_samples(self):
        transformed_body_points = []
        body_point_slices = []
        num_bodies = self.char_model.get_num_joints()
        num_points = 0
        for b in range(num_bodies):
            curr_body_points = self._body_points[b]
            curr_body_rot = self.motion_frames.body_rot[-1][b].unsqueeze(0)
            curr_body_pos = self.motion_frames.body_pos[-1][b]

            curr_body_points = torch_util.quat_rotate(curr_body_rot, curr_body_points) + curr_body_pos
            transformed_body_points.append(curr_body_points)

            body_point_slices.append(slice(num_points, num_points + curr_body_points.shape[0]))
            num_points += curr_body_points.shape[0]

        transformed_body_points = torch.cat(transformed_body_points, dim=0)
        return transformed_body_points, body_point_slices
        
    def init_body_point_samples(self, all_points):
        num_bodies = len(all_points)
        self._body_points = all_points
        assert num_bodies == self.char_model.get_num_joints()
        self._ps_body_point_samples = []
        for b in range(num_bodies):
            name = self.name + self.char_model.get_body_name(b) + " point samples"
            ps_point_cloud = ps.register_point_cloud(name, all_points[b].cpu().numpy(), radius = 0.0005,
                                                     color = self.color,
                                                     enabled=False)
            self._ps_body_point_samples.append(ps_point_cloud)

        self._body_points_enabled = False
        self._sampled_points_enabled = False
        return
    
    def update_body_point_samples(self):
        # assuming forward kinematics is already called

        for b in range(self.char_model.get_num_joints()):
            body_pos = self.motion_frames.body_pos[-1][b]
            body_rot = self.motion_frames.body_rot[-1][b]

            pose = np.eye(4)
            pose[:3, 3] = body_pos.cpu().numpy()

            rot_mat = torch_util.quat_to_matrix(body_rot).cpu().numpy()
            pose[:3, :3] = rot_mat

            self._ps_body_point_samples[b].set_transform(pose)
        return
    
    def set_body_point_colors(self, b, colors):
        self._ps_body_point_samples[b].add_color_quantity("sdf < 0", colors.cpu().numpy(), enabled=True)
        return

    def update_transforms(self, shadow_height=0):
        super().update_transforms(shadow_height)
        if self._ps_body_point_samples is not None:
            self.update_body_point_samples()
        return
    
    # def set_body_points_enabled(self, val):
    #     # TODO: rename this sampled surface points
    #     # for b in range(self.char_model.get_num_joints()):
    #     #     self._ps_body_point_samples[b].set_enabled(val)
    #     # self._body_points_enabled = val
    #     return
    
    def get_body_points_enabled(self):
        return self._body_points_enabled
    
    def get_sampled_points_enabled(self):
        return self._sampled_points
    
    def get_normalized_local_hf(self, terrain: terrain_util.SubTerrain, max_h: float):
        hf_z = self.hf_z
        #z_below_root = self.get_hf_below_root(terrain)
        hf_z = hf_z - self.get_body_pos(0)[2].item() # relative to root
        hf_z = torch.clamp(hf_z, min=-max_h, max=max_h) / max_h
        return hf_z
    
class MDMMotionPS(ps_util.MotionPS):
    def __init__(self, name, mlib: motion_lib.MotionLib, char_color,
                 vis_fps: int = 15, start_time: float = 0.0):

        self.name = name
        self.char_color = char_color
        self.mlib = mlib
        self.char = MDMCharacterPS(name, char_color, mlib._kin_char_model, 2)

        # have a character frame every 2/30 seconds
        motion_length = mlib._motion_lengths[0].item()
        #print("MOTION LENGTH:", motion_length)

        
        if motion_length > 7.0:
           vis_fps = 1
        num_frames = int(round(vis_fps * motion_length))
        if num_frames < 2:
            num_frames = 2

        if vis_fps == -1:
            num_frames = mlib._motion_frames.shape[0]
        self.sequence = ps_util.MotionSequencePS(name + " motion sequence", char_color, [0.0, 0.0, 0.0], 
                                                 num_frames, 0.0, motion_length, self.mlib)

        self.root_offset = np.array([0.0, 0.0, 0.0])
        self.root_heading_angle = 0.0

        self.time_offset = 0.0

        self.start_retarget_time = 0.0
        self.end_retarget_time = self.mlib._motion_lengths[0].item()

        self.set_to_time(start_time)
        # if update_visualization:
        self.update_transforms(shadow_height=self.char.get_hf_below_root(g_terrain))
        #     num_vis_frames = int(round(vis_fps * new_mlib._motion_lengths[0].item()))
        #     MotionManager().get_curr_motion().update_sequence(0.0, new_mlib._motion_lengths[0].item(), num_vis_frames)
        self.char.update_local_hf(g_terrain)
        return
    
    def remove(self):
        self.char.remove()
        self.sequence.remove()
        return
    
    def deselect(self):
        self.char.deselect()
        return
    
    def select(self):
        self.char.select(MainVars().viewing_local_hf, MainVars().viewing_shadow)
        return
    
    def set_motion_sequence_colors(self, frame_colors, num_frames):
        motion_length = self.mlib._motion_lengths[0].item()
        self.sequence = ps_util.MotionSequencePS(self.name + " motion sequence", [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 
                                                 num_frames, 0.0, motion_length, self.mlib,
                                                 frame_colors=frame_colors)
        
        return

class MotionManager(SingletonClass):
    loaded_motions = OrderedDict()
    ik_motions = OrderedDict()
    curr_motion = None
    curr_ik_motion = None

    def load_motion(self, filepath, char_model_path, name, root_offset = None, vis_fps = 15) -> MDMMotionPS:
        char_model = kin_char_model.KinCharModel(MainVars().device)
        char_model.load_char_file(char_model_path)
        mlib = motion_lib.MotionLib(filepath, char_model, device=MainVars().device, contact_info=MainVars().use_contact_info)
        if root_offset is not None:
            mlib._motion_frames[:, 0:2] += root_offset
            mlib._frame_root_pos[:, 0:2] += root_offset
        motion = MDMMotionPS(name, mlib, [0.2, 0.2, 0.8], vis_fps=vis_fps)
        motion.set_to_time(0.0)
        motion.char.update_local_hf(g_terrain)

        self.loaded_motions[name] = motion
        return motion
    
    def add_motion(self, motion: MDMMotionPS, name):
        self.loaded_motions[name] = motion
        return
    
    def set_curr_motion(self, name, sequence_val=True, char_val=True):
        self.curr_motion = self.loaded_motions[name]
        self.curr_motion.set_enabled(sequence_val, char_val)
        return
    
    def get_curr_motion(self) -> MDMMotionPS:
        return self.curr_motion
    
    def get_loaded_motions(self):
        return self.loaded_motions
    
    def make_new_motion(self, 
                        motion_frames: torch.Tensor, 
                        contact_frames: torch.Tensor, 
                        new_motion_name: str,
                        motion_fps: int, 
                        vis_fps: int,
                        loop_mode = motion_lib.LoopMode.CLAMP,
                        new_color = [0.8, 0.2, 0.2],
                        new_char_model = None):
        MainVars().use_contact_info=True
        if new_char_model is None:
            new_char_model = MotionManager().get_curr_motion().char.char_model
        new_mlib = motion_lib.MotionLib(motion_frames, 
                                        new_char_model, 
                                        device=MainVars().device, 
                                        init_type="motion_frames",
                                        loop_mode=loop_mode,
                                        fps=motion_fps,
                                        contact_info=MainVars().use_contact_info,
                                        contacts=contact_frames)

        # TODO: fix
        #MotionManager().get_curr_motion().set_enabled(False)
        dt = 1.0 / motion_fps
        MainVars().motion_time = dt * (MotionManager().get_curr_motion().char._history_length - 1.0)
        #MainVars().motion_time = new_mlib._motion_lengths[0].item()


        self.get_curr_motion().deselect()
        new_motion = MDMMotionPS(new_motion_name, new_mlib, new_color, 
                            vis_fps=vis_fps, start_time=MainVars().motion_time)
        self.add_motion(new_motion, new_motion_name)
        self.set_curr_motion(new_motion_name)

        self.get_curr_motion().select()
        return



# Get Terrain
if g_motion_data.has_terrain():
    g_terrain = g_motion_data.get_terrain()
else:
    g_terrain = terrain_util.SubTerrain(x_dim=16, y_dim=16, dx=0.4, dy=0.4, min_x = -3.0, min_y=-3.0, device=MainVars().device)

if g_motion_yaml_filepath is None:
    if "vis_fps" in input_config:
        #g_motion = load_motion(g_motion_filepath, g_char_filepath, os.path.basename(g_motion_filepath), vis_fps=input_config["vis_fps"])
        MotionManager().load_motion(g_motion_filepath, g_char_filepath, os.path.basename(g_motion_filepath), vis_fps=input_config["vis_fps"])
    else:
        MotionManager().load_motion(g_motion_filepath, g_char_filepath, os.path.basename(g_motion_filepath))
    MotionManager().set_curr_motion(os.path.basename(g_motion_filepath))
    #     g_motion = load_motion(g_motion_filepath, g_char_filepath, os.path.basename(g_motion_filepath))
    # g_loaded_motions = OrderedDict()
    # g_loaded_motions[os.path.basename(g_motion_filepath)] = g_motion
else:
    #g_loaded_motions = OrderedDict()
    with open(g_motion_yaml_filepath, "r") as f:
        motion_yaml = yaml.safe_load(f)
        num_motions = len(motion_yaml["motions"])
        if num_motions > 5:
            vis_fps = 1
        else:
            vis_fps = 15

        if motion_yaml.get("view_every_frame", False):
            vis_fps = -1

        for motion_path in motion_yaml["motions"]:
            name = os.path.basename(motion_path["file"])

            with open(motion_path["file"], "rb") as f:
                motion_data = pickle.load(f)
                if "min_point_offset" in motion_data:
                    offset = motion_data["min_point_offset"]
                else:
                    offset = None
            # g_motion = load_motion(motion_path["file"], g_char_filepath, name, root_offset=offset, vis_fps=vis_fps)
            # g_loaded_motions[name] = g_motion

            MotionManager().load_motion(motion_path["file"], g_char_filepath, name, root_offset=offset, vis_fps=vis_fps)
            MotionManager().set_curr_motion(name)
            MotionManager().get_curr_motion().char.set_local_hf_enabled(False)
            MotionManager().get_curr_motion().char.set_shadow_enabled(False)

        
        terrain_path = motion_yaml["terrain"]
        with open(terrain_path, "rb") as f2:
            g_terrain = pickle.load(f2)["terrain"]
        g_terrain.to_torch(device=MainVars().device)

for elem in g_other_motion_filepaths:
    other_motion_path = elem[0]
    other_char_file = elem[1]

    name = os.path.basename(other_motion_path)
    if name in MotionManager().get_loaded_motions():
        name = name + "_1"

    with open(other_motion_path, "rb") as f:
        motion_data = pickle.load(f)
        if "min_point_offset" in motion_data:
            offset = motion_data["min_point_offset"]
        else:
            offset = None
    other_motion = MotionManager().load_motion(other_motion_path, other_char_file, name, root_offset=offset)

    if g_load_other_motion_terrains:
        other_motion_data = medit_lib.load_motion_file(other_motion_path)
        other_motion_data.set_device(MainVars().device)
        other_terrain = other_motion_data.get_terrain()

        if offset is not None:
            other_terrain.min_point += offset

            # TODO: search for the proper overlap
        

            orig_min_point = other_terrain.min_point.clone()
            for i in range(0, 1):
                for j in range(0, 1):
                    j = -1
                    print(i, j)
                    curr_offset = torch.tensor([i, j], dtype=torch.float32, device=other_terrain.hf.device) * other_terrain.dxdy
                    print("curr offset:", curr_offset)
                    other_terrain.min_point = orig_min_point + curr_offset
                    print(other_terrain.min_point)
                    xy_points = other_terrain.get_grid_node_xy_points()
                    g_terrain_hf_sample = g_terrain.get_hf_val_from_points(xy_points)

                    abs_hf_diff = torch.sum(torch.abs(g_terrain_hf_sample - other_terrain.hf))
                    print("ABS HF DIFF:", abs_hf_diff)

                    if abs_hf_diff < 1e-3:
                        print(curr_offset)


        verts, tris = terrain_util.convert_heightfield_to_voxelized_trimesh(
            other_terrain.hf, 
            other_terrain.min_point[0].item(), 
            other_terrain.min_point[1].item(),
            other_terrain.dxdy[0].item())
        
        hf_trimesh = trimesh.Trimesh(vertices=verts, faces=tris)

        hf_ps_mesh = ps.register_surface_mesh("heightfield_other", verts, tris)

        heights = verts[..., 2]
        max_h = np.max(heights)
        min_h = np.min(heights)
        if max_h > min_h + 1e-3:
            heights = (heights - min_h) / (max_h - min_h)
            hf_ps_mesh.add_scalar_quantity("height", heights, enabled=True)
        else:
            hf_ps_mesh.set_color([0.5, 0.5, 0.5])
    
    MotionManager().add_motion(other_motion, name)

# Use motion to compute terrain hf mask
if MotionManager().get_curr_motion().mlib._hf_mask_inds[0] is not None:
    g_terrain.hf_mask = terrain_util.compute_hf_mask_from_inds(g_terrain,
                                                               MotionManager().get_curr_motion().mlib._hf_mask_inds[0])
TerrainMeshManager().reset()

def get_motion(name) -> MDMMotionPS:
    return MotionManager().get_loaded_motions()[name]

g_dir_mesh = ps_util.create_vector_mesh([1.0, 0.0, 0.0], name="direction", radius = 0.02)

if LOAD_MDM:
    MDMSettings().select_mdm("main")

if "opt:body_constraints" in g_motion_data._data:
    OptimizationSettings().body_constraints = g_motion_data._data["opt:body_constraints"]
    OptimizationSettings().create_body_constraint_ps_meshes()

if "path_nodes" in g_motion_data._data:
    PathPlanningSettings().path_nodes = g_motion_data._data["path_nodes"]
    PathPlanningSettings().visualize_path_nodes(MotionManager().get_curr_motion().name + "_path", PathPlanningSettings().path_nodes)

if "loss" in g_motion_data._data:
    print("Loss:", g_motion_data._data["loss"])

def extract_obs(obs, obs_shapes):
    num_frames = obs.shape[0]

    char_obs_start = 0
    char_obs_end = obs_shapes["char_obs"]["shape"][0]
    char_obs = obs[:, char_obs_start:char_obs_end]

    tar_obs_start = char_obs_end
    tar_obs_end = tar_obs_start + obs_shapes["tar_obs"]["shape"][0] * obs_shapes["tar_obs"]["shape"][1]
    tar_obs = np.reshape(obs[:, tar_obs_start:tar_obs_end], newshape=[num_frames, obs_shapes["tar_obs"]["shape"][0], -1])

    tar_contacts_start = tar_obs_end
    tar_contacts_end = tar_contacts_start + obs_shapes["tar_contacts"]["shape"][0] * obs_shapes["tar_contacts"]["shape"][1]
    tar_contacts = np.reshape(obs[:, tar_contacts_start:tar_contacts_end], newshape=[num_frames, obs_shapes["tar_contacts"]["shape"][0], -1])

    char_contacts_start = tar_contacts_end
    char_contacts_end = char_contacts_start + obs_shapes["char_contacts"]["shape"][0]
    char_contacts = obs[:, char_contacts_start:char_contacts_end]

    hf_start = char_contacts_end
    hf_end = hf_start + obs_shapes["hf"]["shape"][0]
    hf = obs[:, hf_start:hf_end]
    num_points = hf.shape[1]

    ray_points_behind = 2
    ray_points_ahead = 60
    ray_num_left = 3
    ray_num_right = 3
    ray_dx = 0.05
    ray_angle = 0.26179938779

    ray_xy_points = geom_util.get_xy_points_cone(
        center=torch.zeros(size=(2,), dtype=torch.float32, device="cpu"),
        dx=ray_dx,
        num_neg=ray_points_behind,
        num_pos=ray_points_ahead,
        num_rays_neg=ray_num_left,
        num_rays_pos=ray_num_right,
        angle_between_rays=ray_angle).numpy()
    
    assert ray_xy_points.shape[0] == num_points

    hf_points = np.zeros(shape=[num_frames, num_points, 3])
    hf_points[:, :, 0:2] = np.expand_dims(ray_xy_points, 0)
    hf_points[:, :, 2] = hf

    return char_obs, tar_obs, tar_contacts, char_contacts, hf_points

if "obs" in g_motion_data._data:
    IGObsSettings().setup(g_motion_data._data["obs"], g_motion_data._data["obs_shapes"])

def update_dir_mesh():
    global g_dir_mesh

    transform = np.eye(4)
    transform[:3, 3] = MotionManager().get_curr_motion().char.get_body_pos(0).cpu()

    dir = MainVars().mouse_world_pos - MotionManager().get_curr_motion().char.get_body_pos(0)
    dir[2] = 0.0
    norm = torch.norm(dir)

    if norm < 0.05:
        g_dir_mesh.set_transparency(0.0)
        return

    else:
        dir = dir / norm
        angle = torch.atan2(dir[1], dir[0])
        z_axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=MainVars().device)
        quat = torch_util.axis_angle_to_quat(z_axis, angle).squeeze(0)
        rot_mat = torch_util.quat_to_matrix(quat)
        
        transform[:3, :3] = rot_mat.cpu().numpy()
        g_dir_mesh.set_transform(transform)
        g_dir_mesh.set_transparency(1.0)

        return
    
def update_current_motion_mlib(new_mlib, mesh_fps=15):
    MotionManager().get_curr_motion().mlib = new_mlib
    MainVars().motion_time = 0.0
    MotionManager().get_curr_motion().char.set_to_time(MainVars().motion_time, MainVars().motion_dt, MotionManager().get_curr_motion().mlib)
    motion_length = MotionManager().get_curr_motion().mlib._motion_lengths[0].item()
    if motion_length > 5.0:
        mesh_fps = 1
    num_frames = int(round(motion_length * mesh_fps)) + 1
    MotionManager().get_curr_motion().update_sequence(0.0, MotionManager().get_curr_motion().mlib._motion_lengths[0].item(), num_frames)
    MotionManager().get_curr_motion().update_transforms(shadow_height=MotionManager().get_curr_motion().char.get_hf_below_root(g_terrain))
    return