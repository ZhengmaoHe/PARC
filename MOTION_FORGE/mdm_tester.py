import sys
sys.path.insert(1, sys.path[0] + ("/../.."))

import os
import polyscope as ps
import polyscope.imgui as psim
import trimesh
import numpy as np
import torch
import pickle
import time
import yaml
import util.terrain_util as terrain_util
import util.torch_util as torch_util

import MOTION_FORGE.polyscope_util as ps_util
import anim.kin_char_model as kin_char_model
import anim.motion_lib as motion_lib

import diffusion.mdm as mdm
from diffusion.diffusion_util import MDMKeyType, MDMFrameType
from diffusion.mdm_heightfield_contact_motion_sampler import MDMHeightfieldContactMotionSampler

# TODO: button to move camera to each env
# View the motion id and losses for each env

## GLOBALS ##
g_batch_size = 16
g_losses = None
g_total_losses = None
g_conds = None
g_selected_env = 0

ps.set_up_dir("z_up")
ps.set_front_dir("neg_y_front")
ps.set_ground_plane_mode("none")
ps.set_background_color([0.0, 0.0, 0.0])
ps.init()

sampler_file_path = "../Data/samplers/mdm_batch_05_dm.pkl"
input_mdm_path = "../tests/train_mdm/parkour_dataset_v_21_test01/checkpoints/model_7500.pkl"

def load_mdm(mdm_path) -> mdm.MDM:
    with open(mdm_path, 'rb') as input_filestream:
        ret_mdm = pickle.load(input_filestream)
        ret_mdm.update_old_mdm()

        print("MDM uses heightmap: ", ret_mdm._use_heightmap_obs)
        print("MDM uses target: ", ret_mdm._use_target_obs)
    return ret_mdm

g_mdm = load_mdm(input_mdm_path)
g_local_hf_num_neg_x = g_mdm._num_x_neg
g_local_hf_num_pos_x = g_mdm._num_x_pos
g_local_hf_num_neg_y = g_mdm._num_y_neg
g_local_hf_num_pos_y = g_mdm._num_y_pos

def load_mdm_sampler(sampler_config_path) ->  MDMHeightfieldContactMotionSampler:
    with open(sampler_config_path, "r") as stream:
        gen_config = yaml.safe_load(stream)
    gen_config["device"] = "cuda:0"
    motion_sampler = MDMHeightfieldContactMotionSampler(gen_config)
    with open(sampler_file_path, "wb") as stream:
        pickle.dump(motion_sampler, stream)
    return motion_sampler

def load_mdm_sampler_pkl(sampler_path) -> MDMHeightfieldContactMotionSampler:
    with open(sampler_path, "rb") as stream:
        motion_sampler = pickle.load(stream)
    return motion_sampler

if os.path.exists(sampler_file_path):
    g_sampler = load_mdm_sampler_pkl(sampler_file_path)
else:
    g_sampler = load_mdm_sampler("diffusion/mdm.yaml")


def build_ps_hf_mesh(hf, min_x, min_y, dx, name) -> ps.SurfaceMesh:
    verts, tris = terrain_util.convert_heightfield_to_voxelized_trimesh(hf, 
                                                      min_x, 
                                                      min_y,
                                                      dx)
    
    hf_trimesh = trimesh.Trimesh(vertices=verts, faces=tris)
    hf_embree = trimesh.ray.ray_pyembree.RayMeshIntersector(hf_trimesh)

    hf_ps_mesh = ps.register_surface_mesh(name, verts, tris)
    hf_ps_mesh.set_color([0.5, 0.5, 0.5])
    return hf_ps_mesh, hf_embree

def get_mlib_format(motion_dict):

    motion_frames = torch.cat([
        motion_dict[MDMFrameType.ROOT_POS],
        torch_util.quat_to_exp_map(motion_dict[MDMFrameType.ROOT_ROT]),
        g_mdm._kin_char_model.rot_to_dof(motion_dict[MDMFrameType.JOINT_ROT])
    ], dim=-1)
    contacts = motion_dict[MDMFrameType.CONTACTS]

    return motion_frames, contacts

g_curr_time = time.time()
def main_loop():
    global g_curr_time
    global g_batch_size, g_selected_env
    global g_losses, g_total_losses
    padding = 4.0

    next_curr_time = time.time()
    dt = next_curr_time - g_curr_time
    g_curr_time = next_curr_time

    changed, g_batch_size = psim.InputInt("batch size", g_batch_size)
    if changed:
        g_batch_size = int(np.sqrt(g_batch_size)) ** 2

    changed, g_selected_env = psim.InputInt("selected env", g_selected_env)
    if changed:
        g_selected_env = min(max(g_selected_env, 0), g_batch_size-1)


        i = g_selected_env % int(np.sqrt(g_batch_size))
        j = g_selected_env // int(np.sqrt(g_batch_size))

        dy = g_sampler._grid_dim_y * g_mdm._dx + padding
        dx = g_mdm._grid_dim_x * g_mdm._dx + padding
        target = [i * dx, j * dy, 0.0]
        camera_location = [i * dx + 3.0, j * dy  + 3.0, 0.0  + 3.0]
        ps.look_at(camera_location=camera_location, target=target)
    

    if psim.Button("Sample MDM training motions"):
        print("Calling MDM")

        #batch_size = 4
        

        losses, hf, x0, gen_x0, conds = g_mdm.sample_and_compute_losses(
            max_t=g_mdm._diffusion_timesteps,
            motion_sampler=g_sampler,
            batch_size=g_batch_size,
            loss_fn=mdm.squared_l2_loss_fn,
            get_info=True,
            test=True,
            tf_chance=0.0,
            ret_extra=True)
        
        g_losses = losses
        g_total_losses = torch.zeros(size=[g_batch_size], dtype=torch.float32, device=g_mdm._device)
        for key in losses:
            g_total_losses += losses[key]
        
        g_conds = conds
        # for key in losses:
        #     print(mdm.LossType(key).name, ":", losses[key])
        
        
        ## Unnormalize
        x0 = g_mdm.extract_motion_features(x0)
        gen_x0 = g_mdm.extract_motion_features(gen_x0)

        dx = g_mdm._dx
        min_x = -g_mdm._num_x_neg * dx
        min_y = -g_mdm._num_y_neg * dx

        x_offset = 0.0
        y_offset = 0.0

        x0_motion, x0_contacts = get_mlib_format(x0)
        gen_x0_motion, gen_x0_contacts = get_mlib_format(gen_x0)

        for i in range(g_batch_size):
            print("building meshes for sample", i)
            if i > 0 and i % int(np.sqrt(g_batch_size)) == 0:
                y_offset += g_mdm._grid_dim_y * dx + padding
                x_offset = 0.0
            build_ps_hf_mesh(hf[i, 0], min_x + x_offset, min_y + y_offset, dx, "hf" + str(i).zfill(3))
            

            # x0_motion = x0[i, :, g_mdm._mframe_slice].unsqueeze(0)
            # x0_motion[..., 0] += x_offset
            # x0_motion[..., 1] += y_offset
            x0_motion[i, ..., 0] += x_offset
            x0_motion[i, ..., 1] += y_offset
            #x0_contacts = x0[i, :, g_mdm._ftype_slices[mdm.MDMFrameInfoType.CONTACTS]].unsqueeze(0)
            

            
            
            x0_mlib = motion_lib.MotionLib(motion_input = x0_motion[i].unsqueeze(0), 
                                           kin_char_model = g_sampler._kin_char_model,
                                           device = g_sampler._device,
                                           init_type = "motion_frames",
                                           loop_mode = motion_lib.LoopMode.CLAMP,
                                           fps = g_mdm._sequence_fps,
                                           contact_info = True,
                                           contacts = x0_contacts[i].unsqueeze(0))
            
            ps_util.MotionSequencePS(name = "x0_" + str(i).zfill(3),
                                     start_color = [0.5, 0.5, 1.0],
                                     end_color = [0.5, 0.5, 1.0],
                                     start_time = 0.0,
                                     end_time = g_mdm._sequence_duration,
                                     num_frames = g_mdm._seq_len,
                                     mlib = x0_mlib)
            
            gen_x0_motion[i, ..., 0] += x_offset
            gen_x0_motion[i, ..., 1] += y_offset
            
            gen_x0_mlib = motion_lib.MotionLib(motion_input = gen_x0_motion[i].unsqueeze(0), 
                                           kin_char_model = g_sampler._kin_char_model,
                                           device = g_sampler._device,
                                           init_type = "motion_frames",
                                           loop_mode = motion_lib.LoopMode.CLAMP,
                                           fps = g_mdm._sequence_fps,
                                           contact_info = True,
                                           contacts = gen_x0_contacts[i].unsqueeze(0))
            
            ps_util.MotionSequencePS(name = "gen_x0_" + str(i).zfill(3),
                                     start_color = [0.5, 1.0, 0.5],
                                     end_color = [0.5, 1.0, 0.5],
                                     start_time = 0.0,
                                     end_time = g_mdm._sequence_duration,
                                     num_frames = g_mdm._seq_len,
                                     mlib = gen_x0_mlib)
            

            # gen_x0_floor_heights = gen_x0[i, :, g_mdm._ftype_slices[mdm.MDMFrameInfoType.FLOOR_HEIGHTS]]
            # # visualize floor heights
            # floor_height_pts = torch.cat([gen_x0_motion[..., 0:2].squeeze(0), gen_x0_floor_heights], dim=-1)
            # floor_height_pts = floor_height_pts.cpu().numpy()
            # ps_floor_height_pts = ps.register_point_cloud("gen floor heights " + str(i).zfill(3), floor_height_pts, radius=0.0014)
            # ps_floor_height_pts.set_color([0.0, 1.0, 1.0])

            # print("gen floor heights (" + str(i) + "):",  gen_x0_floor_heights)


            # x0_floor_heights = x0[i, :, g_mdm._ftype_slices[mdm.MDMFrameInfoType.FLOOR_HEIGHTS]]
            # # visualize floor heights
            # floor_height_pts = torch.cat([x0_motion[..., 0:2].squeeze(0), x0_floor_heights], dim=-1)
            # floor_height_pts = floor_height_pts.cpu().numpy()
            # ps_floor_height_pts = ps.register_point_cloud("true floor heights " + str(i).zfill(3), floor_height_pts, radius=0.0014)
            # ps_floor_height_pts.set_color([1.0, 0.0, 1.0])


            x_offset += g_mdm._grid_dim_x * dx + padding

    if g_losses is not None:
        assert g_total_losses is not None
        
        total_loss_str = "total_loss: " + str(g_total_losses[g_selected_env].item())
        psim.TextUnformatted(total_loss_str)
        for key in g_losses:
            loss_str = mdm.LossType(key).name + ": " + np.array2string(g_losses[key][g_selected_env].cpu().numpy())
            psim.TextUnformatted(loss_str)


    if psim.Button("Sample motions"):
        padding = 4.0
        motion_samples, hf, future_pos, future_rot = g_sampler.sample_motion_data(
            ret_future_pos=True,
            ret_future_rot=True,
            ret_contacts=True,
            ret_floor_heights=True,
            num_samples=g_batch_size)
        
        x0 = motion_samples

        ## Unnormalize
        x0[..., :g_sampler.get_num_dof()] *= g_sampler._dof_high

        ## Unnormalize hf as well
        hf *= g_sampler._max_h

        dx = g_sampler._dx
        min_x = -g_sampler._num_x_neg * dx
        min_y = -g_sampler._num_y_neg * dx

        x_offset = 0.0
        y_offset = 0.0

        for i in range(g_batch_size):
            print("building meshes for sample", i)
            if i > 0 and i % int(np.sqrt(g_batch_size)) == 0:
                y_offset += g_sampler._grid_dim_y * dx + padding
                x_offset = 0.0
            build_ps_hf_mesh(hf[i], min_x + x_offset, min_y + y_offset, dx, "hf" + str(i).zfill(3))
            

            x0_motion = x0[i, :, :g_sampler.get_num_dof()].unsqueeze(0)
            x0_motion[..., 0] += x_offset
            x0_motion[..., 1] += y_offset
            x0_contacts = x0[i, :, 34:49].unsqueeze(0)
            x0_mlib = motion_lib.MotionLib(motion_input = x0_motion, 
                                           kin_char_model = g_sampler._kin_char_model,
                                           device = g_sampler._device,
                                           init_type = "motion_frames",
                                           loop_mode = motion_lib.LoopMode.CLAMP,
                                           fps = g_mdm._sequence_fps,
                                           contact_info = True,
                                           contacts = x0_contacts)
            
            ps_util.MotionSequencePS(name = "x0_" + str(i).zfill(3),
                                     start_color = [0.5, 1.0, 0.5],
                                     end_color = [0.5, 1.0, 0.5],
                                     start_time = 0.0,
                                     end_time = g_mdm._sequence_duration,
                                     num_frames = g_mdm._seq_len,
                                     mlib = x0_mlib)
            
            x_offset += g_sampler._grid_dim_x * dx + padding



ps.set_user_callback(main_loop)
ps.show()