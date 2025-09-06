import sys
sys.path.insert(1, sys.path[0] + ("/../..")) # for running this script in vs code in the root dir

import os
import polyscope as ps
import polyscope.imgui as psim
import trimesh
import time
import numpy as np
import torch
import pickle
import yaml

import MOTION_FORGE.polyscope_util as ps_util
import anim.kin_char_model as kin_char_model
import anim.motion_lib as motion_lib
import util.terrain_util as terrain_util
import util.torch_util as torch_util
import zmotion_editing_tools.motion_edit_lib as medit
import tools.retargeter.retargeter as rtgt_lib

import matplotlib.pyplot as plt

ps.set_up_dir("z_up")
ps.set_front_dir("neg_y_front")
ps.init()
ps_util.create_origin_axis_mesh()

## GLOBALS
g_device = "cpu"
g_motion_time = 0.0
g_paused = False
g_rtgt_cfg_path = "tools/retargeter/configs/rl_forge_to_rl_forge_config.yaml"
g_temporal_w = 0.05
g_use_sliders = True
g_A_motion_match_start_time = 2.7
g_A_motion_match_end_time = 3.0
g_B_motion_match_start_time = 0.5
g_B_motion_match_end_time = 1.0
g_num_blend_frames = 30
g_frame_ind_A = 0
g_frame_ind_B = 0

## LOAD CHARACTER MODEL DATA
g_char_model = kin_char_model.KinCharModel(g_device)
g_char_model.load_char_file("data/assets/humanoid.xml")


def load_motion_from_file(name, filepath, char_color, char_model: kin_char_model.KinCharModel):
    mlib = motion_lib.MotionLib(filepath, char_model, g_device, contact_info=False)
    return ps_util.MotionPS(name, mlib, char_color)

g_motions = [
    load_motion_from_file("motion 1", "../Data/parkour_dataset/v_05/running/dm_beyond_running_008.pkl", [0.2, 0.8, 0.2],  g_char_model),
    load_motion_from_file("unreal", "../Data/parkour_dataset/v_05_5/unreal/dm/unreal_take_004_6.pkl", [0.2, 0.2, 0.8], g_char_model)
]
g_num_motions = len(g_motions)
g_max_time = g_motions[0].mlib.get_motion_length(0)
g_pos_offsets = np.zeros(shape=(g_num_motions, 3))


def build_ground_plane():
    # TODO: do something about this
    extents = np.array([40.0, 40.0, 0.1])
    transform = np.eye(4)
    transform[2, 3] = -0.05
    plane = trimesh.primitives.Box(extents=extents, transform=transform)
    plane_ps = ps.register_surface_mesh("ground plane", plane.vertices, plane.faces)
    plane_ps.set_color([0.5, 0.5, 0.5])
    ps.set_ground_plane_mode("none")
    ps.set_background_color([0.0, 0.0, 0.0])

build_ground_plane()

def find_local_minima(tensor):
    # Check if the tensor has at least one dimension
    if tensor.ndimension() != 2:
        raise ValueError("The input tensor must be 2-dimensional")

    # Add padding to avoid boundary issues
    padded_tensor = torch.nn.functional.pad(tensor, (1, 1, 1, 1), mode='constant', value=float('inf'))

    # Create shifted versions of the tensor to represent neighbors
    shift_up = padded_tensor[0:-2, 1:-1]
    shift_down = padded_tensor[2:, 1:-1]
    shift_left = padded_tensor[1:-1, 0:-2]
    shift_right = padded_tensor[1:-1, 2:]

    shift_upleft = padded_tensor[0:-2, 0:-2]
    shift_upright = padded_tensor[0:-2, 2:]
    shift_downleft = padded_tensor[2:, 0:-2]
    shift_downright = padded_tensor[2:, 2:]

    # Current tensor
    current = padded_tensor[1:-1, 1:-1]

    # Check if each point is a local minimum
    is_local_minimum = (current <= shift_up) & (current <= shift_down) & \
                       (current <= shift_left) & (current <= shift_right) & \
                       (current <= shift_upleft) & (current <= shift_upright) & \
                       (current <= shift_downleft) & (current <= shift_downright)

    # Get the indices of the local minima
    local_minima_indices = torch.nonzero(is_local_minimum, as_tuple=True)

    # Get the values of the local minima
    local_minima_values = tensor[local_minima_indices]

    return local_minima_indices, local_minima_values


curr_time = time.time()
def main_loop():
    global curr_time, g_motion_time, g_paused
    global g_char_model
    global g_pos_offsets
    global g_temporal_w
    global g_A_motion_match_start_time, g_A_motion_match_end_time
    global g_B_motion_match_start_time, g_B_motion_match_end_time
    global g_use_sliders
    global g_num_blend_frames
    global g_frame_ind_A, g_frame_ind_B
    next_curr_time = time.time()
    dt = next_curr_time - curr_time
    curr_time = next_curr_time

    if not g_paused:

        for motion in g_motions:
            motion.char.set_to_time(g_motion_time + motion.time_offset, dt, motion.mlib)
            motion.update_transforms()
        

        g_motion_time += dt
        if g_motion_time > g_max_time:
            g_motion_time = 0.0

    changed, g_use_sliders = psim.Checkbox("use sliders", g_use_sliders)

    if(psim.TreeNode("Motion Data")):
        for i in range(len(g_motions)):
            curr_motion = g_motions[i]
            motion_name_str = os.path.basename(curr_motion.name)
            if(psim.TreeNode(motion_name_str)):
                dur_str = "motion duration: " + str(curr_motion.mlib._motion_lengths[0].item())
                psim.TextUnformatted(dur_str)
                time_str = "current time: " + str(curr_motion.time_offset + g_motion_time)
                psim.TextUnformatted(time_str)
                fps_str = "fps: " + str(curr_motion.mlib._motion_fps[0].item())
                psim.TextUnformatted(fps_str)


                if g_use_sliders:
                    changedx, curr_motion.root_offset[0] = psim.SliderFloat("offset x", curr_motion.root_offset[0], v_min = -10.0, v_max = 10.0)
                    changedy, curr_motion.root_offset[1] = psim.SliderFloat("offset y", curr_motion.root_offset[1], v_min = -10.0, v_max = 10.0)
                    changedz, curr_motion.root_offset[2] = psim.SliderFloat("offset z", curr_motion.root_offset[2], v_min = -10.0, v_max = 10.0)
                    changed_rot, curr_motion.root_heading_angle = psim.SliderFloat("heading angle offset", curr_motion.root_heading_angle, v_min = -np.pi, v_max = np.pi)
                    changed_time, curr_motion.time_offset = psim.SliderFloat("time offset", curr_motion.time_offset, v_min = -10.0, v_max = 10.0)
                else:
                    changedx, curr_motion.root_offset[0] = psim.InputFloat("offset x", curr_motion.root_offset[0])
                    changedy, curr_motion.root_offset[1] = psim.InputFloat("offset y", curr_motion.root_offset[1])
                    changedz, curr_motion.root_offset[2] = psim.InputFloat("offset z", curr_motion.root_offset[2])
                    changed_rot, curr_motion.root_heading_angle = psim.InputFloat("heading angle offset", curr_motion.root_heading_angle)
                    changed_time, curr_motion.time_offset = psim.InputFloat("time offset", curr_motion.time_offset)
                changed = changedx or changedy or changedz or changed_rot or changed_time or changed_time
                if changed:
                    curr_motion.char.set_to_time(g_motion_time + curr_motion.time_offset, dt, curr_motion.mlib)
                    curr_motion.update_transforms()

                changed, curr_motion.sequence.visibility = psim.SliderFloat("visibility", curr_motion.sequence.visibility, v_min=0.0, v_max=1.0)
                if changed:
                    curr_motion.sequence.mesh.set_transparency(curr_motion.sequence.visibility)


                changed, curr_motion.start_retarget_time = psim.InputFloat("Retargetings start time", curr_motion.start_retarget_time)
                changed, curr_motion.end_retarget_time = psim.InputFloat("Retargeting end time", curr_motion.end_retarget_time)

                changed, g_temporal_w = psim.InputFloat("temporal w", g_temporal_w)

                if psim.Button("Retarget motion"):
                    #print("TODO")
                    
                    # TODO: get the exact indices that are being extracted
                    src_fps = curr_motion.mlib._motion_fps[0].item()
                    src_motion_frames, start_idx, end_idx = medit.slice_motion(curr_motion.mlib._motion_frames.clone(), 
                                                                            curr_motion.start_retarget_time, 
                                                                            curr_motion.end_retarget_time,
                                                                            fps=src_fps,
                                                                            ret_idx_info=True)
                    
                    src_root_pos = src_motion_frames[0, 0:2].clone()
                    #print("offset:", -src_root_pos)

                    with open(g_rtgt_cfg_path, "r") as filestream:
                        rtgt_cfg = yaml.safe_load(filestream)
                    rtgt_cfg['boundary_constraints'] = True
                    rtgt_cfg['num_iters'] = 500
                    rtgt_cfg['w_temporal'] = g_temporal_w
                    logger = rtgt_lib.build_logger(log_file=None, use_wandb=False)

                    og_src_motion_frames = src_motion_frames.clone()
                    tgt_motion_frames = rtgt_lib.retarget_motion_frames(rtgt_cfg, src_motion_frames.to(device="cuda:0"), src_fps, logger)

                    tgt_motion_frames = tgt_motion_frames.to(device=g_device)
                    tgt_motion_frames[:, 0:2] += src_root_pos

                    print(og_src_motion_frames[-1, 0:3])
                    print(og_src_motion_frames[-2, 0:3])
                    print(tgt_motion_frames[-1, 0:3])
                    print(tgt_motion_frames[-2, 0:3])
                    
                    # Note: the retargeter aligns the src frames so that they are at the origin

                    final_motion_frames = curr_motion.mlib._motion_frames.clone()
                    final_motion_frames[start_idx:end_idx+1] = tgt_motion_frames

                    print("rtgt motion shape:", tgt_motion_frames.shape)
                    print("new motion shape:", final_motion_frames.shape)
                    motion_data = {
                        "frames": final_motion_frames.cpu().numpy(),
                        "loop_mode": "CLAMP",
                        "fps": curr_motion.mlib._motion_fps[0].item()
                    }
                    with open("retargeted_motion.pkl", "wb") as filestream:
                        pickle.dump(motion_data, filestream)
                    
                    g_motions.append(load_motion_from_file("retargeted motion", "retargeted_motion.pkl", [0.8, 0.2, 0.8], g_char_model))

                if psim.Button("apply transforms to motion data"):
                    curr_motion.apply_transforms_to_motion_data()

                if psim.Button("save to file"):

                    motion_frames = medit.slice_motion(curr_motion.mlib._motion_frames, curr_motion.start_retarget_time,
                                                       curr_motion.end_retarget_time, curr_motion.mlib._motion_fps[0].item())
                    motion_data = {
                        "frames": motion_frames.cpu().numpy(),
                        "loop_mode": "CLAMP",
                        "fps": curr_motion.mlib._motion_fps[0].item(),
                    }

                    with open("output/_motions/new_motion.pkl", "wb") as filestream:
                        pickle.dump(motion_data, filestream)
                    print("saved to new_motion.pkl")

                if psim.Button("Remove Motion"):
                    curr_motion.char.remove()
                    curr_motion.sequence.remove()
                    del g_motions[i]
                psim.TreePop()

        psim.TreePop()

    changed, g_paused = psim.Checkbox("Paused", g_paused)

    changed, g_motion_time = psim.SliderFloat("Motion time", g_motion_time, v_min = 0.0, v_max = g_max_time)
    if changed and g_paused:
        for motion in g_motions:
            motion.char.set_to_time(g_motion_time + motion.time_offset, dt, motion.mlib)
            motion.update_transforms()

    if len(g_motions) >= 2:
        changed, g_A_motion_match_start_time = psim.InputFloat("A motion match start time", g_A_motion_match_start_time)
        changed, g_A_motion_match_end_time = psim.InputFloat("A motion match end time", g_A_motion_match_end_time)
        changed, g_B_motion_match_start_time = psim.InputFloat("B motion match start time", g_B_motion_match_start_time)
        changed, g_B_motion_match_end_time = psim.InputFloat("B motion match end time", g_B_motion_match_end_time)
        if(psim.Button("Search for motion matching frames")):
            
            t_A, t_B, heading_diff, root_pos_diff = medit.search_for_matching_motion_frames(
                                                    g_motions[0].mlib, g_motions[1].mlib,
                                                    g_A_motion_match_start_time, g_A_motion_match_end_time,
                                                    g_B_motion_match_start_time, g_B_motion_match_end_time)
            
            print("best matching times:", t_A, ", ", t_B)

            #motion_ids = torch.tensor([0], dtype=torch.int64)
            #motion_times = torch.tensor([t_A], dtype=torch.float32)
            #g_motions[0].mlib.calc_motion_frame(motion_ids, motion_times)

            g_motion_time = 0.0
            g_motions[0].time_offset = t_A
            g_motions[1].time_offset = t_B

            g_motions[1].root_heading_angle = heading_diff.item()
            g_motions[1].root_offset = root_pos_diff.numpy()

            for motion in g_motions:
                motion.update_transforms() # this sets the pos and rot offset in the char
                motion.char.set_to_time(g_motion_time + motion.time_offset, dt, motion.mlib)
                motion.update_transforms() # and this also updates the mesh... kinda messy, TODO fix


        changed, g_num_blend_frames = psim.InputInt("num blend frames", g_num_blend_frames)
        if g_num_blend_frames < 0:
            g_num_blend_frames = 0

        if(psim.Button("Stitch Motions together")):
            fps = g_motions[0].mlib._motion_fps[0].item()
            assert fps == g_motions[1].mlib._motion_fps[0].item()
            dt = 1.0 / fps
            
            # 1. Get the motion frames with the transformations
            motion_frames = []
            for i in range(2):
                curr_motion_frames = g_motions[i].mlib._motion_frames
                curr_rot_quat = g_motions[i].compute_rot_quat()

                curr_motion_frames = medit.rotate_motion(curr_motion_frames, curr_rot_quat, torch.tensor([0.0, 0.0, 0.0]))
                curr_motion_frames = medit.translate_motion(curr_motion_frames, torch.tensor(g_motions[i].root_offset))
                motion_frames.append(curr_motion_frames)
            

           
            # 2. stitch them
            first_motion_end_time = g_motion_time + g_motions[0].time_offset
            second_motion_start_time = g_motion_time + g_motions[1].time_offset
            if g_num_blend_frames < 2:
                new_motion = medit.stitch_motions(motion_frames[0], None, first_motion_end_time,
                                                  motion_frames[1], second_motion_start_time, None, fps=fps)
            else:
                new_motion = medit.blend_motions(motion_frames[0], None, first_motion_end_time,
                                                motion_frames[1], second_motion_start_time, None, fps=fps,
                                                num_blend_frames=g_num_blend_frames)
            
            print("new motion shape:", new_motion.shape)
            motion_data = {
                "frames": new_motion.cpu().numpy(),
                "loop_mode": "CLAMP",
                "fps": fps
            }
            with open("output/_motions/stitched_motion.pkl", "wb") as filestream:
                pickle.dump(motion_data, filestream)
            
            g_motions.append(load_motion_from_file("stitched motion", "output/_motions/stitched_motion.pkl", [0.8, 0.2, 0.2], g_char_model))

        if(psim.Button("Test motion matching function")):
            motion_vectors = medit.compute_motion_feature_vectors([g_motions[0].mlib, g_motions[1].mlib])

            for mv in motion_vectors:
                print(mv.shape)

            if len(motion_vectors) == 2:
                mA = motion_vectors[0]
                mB = motion_vectors[1]
                diff = mA.unsqueeze(1) - mB.unsqueeze(0)
                diff = torch.sum(torch.abs(diff), dim=-1)
                
                # normalized between 0 and 1
                diff = diff / torch.max(diff)

                indices, values = find_local_minima(diff)

                #print(indices[allowable_mask], values[allowable_mask])
                print(indices, values)

                plt.imshow(diff.numpy(), cmap='viridis')  # choose a color map for visualization
                plt.axis('off')  # hide axes for a cleaner image
                plt.savefig('transition_matrix.png', bbox_inches='tight', pad_inches=0)  # save as PNG file
                plt.close()  # close the plot to avoid displaying it in interactive environments


                local_min_image = torch.zeros_like(diff)
                local_min_image[indices[0], indices[1]] = 255.0
                plt.imshow(local_min_image.numpy(), cmap='gray', vmin=0, vmax=255)  # choose a color map for visualization
                plt.axis('off')  # hide axes for a cleaner image
                plt.savefig('local_mins.png', bbox_inches='tight', pad_inches=0)  # save as PNG file
                plt.close()  # close the plot to avoid displaying it in interactive environments

        changed, g_frame_ind_A = psim.InputInt("motion 1 frame", g_frame_ind_A)
        changed, g_frame_ind_B = psim.InputInt("motion 2 frame", g_frame_ind_B)
        if psim.Button("Localize 2nd motion frame to 1st motion frame"):
                g_motion_time = 0.0
                tA = g_frame_ind_A / g_motions[0].mlib._motion_fps[0].item()
                tB = g_frame_ind_B / g_motions[1].mlib._motion_fps[0].item()

                print(tA, tB)
                
                g_motions[0].time_offset = tA
                g_motions[1].time_offset = tB

                motion_id = torch.zeros(size=(1,), dtype=torch.int64, device=g_device)
                root_pos_A, root_rot_A, root_vel, root_ang_vel, joint_rot, dof_vel = g_motions[0].mlib.calc_motion_frame(motion_id, torch.tensor([tA]))
                root_pos_B, root_rot_B, root_vel, root_ang_vel, joint_rot, dof_vel = g_motions[1].mlib.calc_motion_frame(motion_id, torch.tensor([tB]))

                root_pos_A = root_pos_A.squeeze(0)
                root_rot_A = root_rot_A.squeeze(0)
                root_pos_B = root_pos_B.squeeze(0)
                root_rot_B = root_rot_B.squeeze(0)


                heading_A = torch_util.calc_heading(root_rot_A)
                heading_B = torch_util.calc_heading(root_rot_B)
                heading_diff = heading_A - heading_B

                z_axis = torch.tensor([0.0, 0.0, 1.0])
                heading_diff_quat = torch_util.axis_angle_to_quat(z_axis, heading_diff).squeeze(0)

                root_pos_B = torch_util.quat_rotate(heading_diff_quat, root_pos_B)
                root_pos_diff = root_pos_A - root_pos_B
                root_pos_diff = root_pos_diff
                root_pos_diff[2] = 0.0
                #root_pos_diff = root_pos_A[..., 0:3]# - root_pos_B[..., 0:3]
                #root_pos_diff[..., 2] = 0.0
                #root_pos_diff = root_pos_diff.squeeze(0)

                g_motions[1].root_heading_angle = heading_diff.item()
                g_motions[1].root_offset = root_pos_diff.numpy()

                for motion in g_motions:
                    motion.update_transforms() # this sets the pos and rot offset in the char
                    motion.char.set_to_time(g_motion_time + motion.time_offset, dt, motion.mlib)
                    motion.update_transforms() # and this also updates the mesh... kinda messy, TODO fix

ps.set_user_callback(main_loop)
ps.show()