import numpy as np
import torch
import polyscope as ps
import polyscope.imgui as psim
import trimesh

import util.torch_util as torch_util
import anim.motion_lib as motion_lib
import diffusion.mdm as mdm
from diffusion.diffusion_util import MDMKeyType, MDMCustomGuidance
import zmotion_editing_tools.motion_edit_lib as medit_lib
import tools.motion_opt.motion_optimization as moopt
import MOTION_FORGE.include.global_header as g

import diffusion.gen_util as gen_util


def process_IO():
    io = psim.GetIO()

    terrain_meshes = g.TerrainMeshManager()
    main_vars = g.MainVars()
    terrain_editor_settings = g.TerrainEditorSettings()
    curr_motion = g.MotionManager().get_curr_motion()
    
    #if io.MouseClicked[0]:
    screen_coords = io.MousePos
    cam_params = ps.get_view_camera_parameters()

    world_cam_pos = np.expand_dims(cam_params.get_position(), axis=0)
    world_ray = np.expand_dims(ps.screen_coords_to_world_ray(screen_coords), axis=0)

    locs, _, __ =  terrain_meshes.hf_embree.intersects_location(ray_origins=world_cam_pos, ray_directions=world_ray)

    if len(locs) > 0:
        world_pos = torch.tensor(locs[0], dtype=torch.float32, device = main_vars.device)
        main_vars.mouse_world_pos = world_pos

        grid_ind = torch.round((world_pos[0:2] - g.g_terrain.min_point) / g.g_terrain.dxdy).to(dtype=torch.int64)
        
        grid_ind = torch.clamp(grid_ind, torch.zeros_like(g.g_terrain.dims), g.g_terrain.dims-1)
        main_vars.selected_grid_ind = grid_ind

        dim = main_vars.mouse_size * 2 + 1
        for i in range(dim):
            for j in range(dim):
                list_ind = i * dim + j

                curr_grid_ind = grid_ind.clone()
                curr_grid_ind[0] += i - main_vars.mouse_size
                curr_grid_ind[1] += j - main_vars.mouse_size

                grid_ind_loc_xy = curr_grid_ind * g.g_terrain.dxdy + g.g_terrain.min_point
                transform = np.eye(4)
                transform[:2, 3] = grid_ind_loc_xy.cpu().numpy()
                transform[2, 3] = g.g_terrain.get_hf_val(curr_grid_ind).item()
        

                g.g_mouse_ball_meshes[list_ind].set_transform(transform)

        if psim.IsKeyReleased(psim.ImGuiKey_A):
            for i in range(dim):
                for j in range(dim):
                    list_ind = i * dim + j

                    curr_grid_ind = main_vars.selected_grid_ind.clone()
                    curr_grid_ind[0] += i - main_vars.mouse_size
                    curr_grid_ind[1] += j - main_vars.mouse_size

                    if terrain_editor_settings.edit_modes[terrain_editor_settings.curr_edit_mode] == "heightfield":
                        g.g_terrain.set_hf_val(curr_grid_ind, terrain_editor_settings.height)
                    elif terrain_editor_settings.edit_modes[terrain_editor_settings.curr_edit_mode] == "mask":
                        g.g_terrain.set_hf_mask_val(curr_grid_ind, terrain_editor_settings.mask_mode)
                    elif terrain_editor_settings.edit_modes[terrain_editor_settings.curr_edit_mode] == "max":
                        g.g_terrain.set_hf_max_val(curr_grid_ind, terrain_editor_settings.height)
                    elif terrain_editor_settings.edit_modes[terrain_editor_settings.curr_edit_mode] == "min":
                        g.g_terrain.set_hf_min_val(curr_grid_ind, terrain_editor_settings.height)

            if terrain_editor_settings.edit_modes[terrain_editor_settings.curr_edit_mode] == "heightfield" or \
                terrain_editor_settings.edit_modes[terrain_editor_settings.curr_edit_mode] == "mask":
                terrain_meshes.soft_rebuild()

            if terrain_editor_settings.edit_modes[terrain_editor_settings.curr_edit_mode] == "max":
                terrain_meshes.build_ps_max_mesh()
            if terrain_editor_settings.edit_modes[terrain_editor_settings.curr_edit_mode] == "min":
                terrain_meshes.build_ps_min_mesh()    

        if psim.IsKeyReleased(psim.ImGuiKey_G):
            # TODO: return generated motion instead of automatically creating new ps motion
            batch_size = g.MDMSettings().mdm_batch_size

            
            target_world_pos = main_vars.mouse_world_pos.unsqueeze(0).expand(batch_size, -1)

            if not g.MDMSettings().gen_settings.input_root_pos:
                prev_frames = curr_motion.char.motion_frames.unsqueeze(0).expand_first_dim(batch_size)
            else:
                prev_frames = curr_motion.char.compute_motion_frames(motion_time=g.MainVars().motion_time, 
                                                                    dt=1.0/g.g_mdm_model._sequence_fps,
                                                                    seq_len=g.g_mdm_model._seq_len,
                                                                    hist_len=g.g_mdm_model._num_prev_states,
                                                                    mlib=curr_motion.mlib).unsqueeze(0).expand_first_dim(batch_size)

            gen_motion_frames = gen_util.gen_mdm_motion(target_world_pos=target_world_pos,
                                                        prev_frames=prev_frames,
                                                        terrain=g.g_terrain,
                                                        char_model=curr_motion.char.char_model,
                                                        mdm_model=g.g_mdm_model,
                                                        mdm_settings=g.MDMSettings().gen_settings)

            # TODO: if appending motion frames, let's figure out a faster way to update the motion_lib without
            # reconstructing it from scratch

            # root_pos = gen_motion_frames.root_pos
            # root_rot = gen_motion_frames.root_rot
            # joint_rot = gen_motion_frames.joint_rot
            # contacts = gen_motion_frames.contacts

            # mlib_motion_frames = torch.cat([root_pos, torch_util.quat_to_exp_map(root_rot),
            #                                 curr_motion.char.char_model.rot_to_dof(joint_rot)], dim=-1)
            # mlib_contact_frames = contacts
            mlib_motion_frames, mlib_contact_frames = gen_motion_frames.get_mlib_format(curr_motion.char.char_model)

            # CONCAT OLD MOTION WITH NEW MOTION
            if g.MDMSettings().append_mdm_motion_to_prev_motion:
                fps = curr_motion.mlib._motion_fps[0].item()
                curr_frame_idx = main_vars.motion_time * fps
                curr_frame_idx = int(round(curr_frame_idx))

                mlib_motion_frames = torch.cat([curr_motion.mlib._motion_frames[0:curr_frame_idx+1].unsqueeze(0).expand(batch_size, -1, -1), 
                                                mlib_motion_frames[:, curr_motion.char._history_length:]], dim=1)
                mlib_contact_frames = torch.cat([curr_motion.mlib._frame_contacts[0:curr_frame_idx+1].unsqueeze(0).expand(batch_size, -1, -1), 
                                                 mlib_contact_frames[:, curr_motion.char._history_length:]], dim=1)

            for i in range(batch_size):
                g.MotionManager().make_new_motion(motion_frames=mlib_motion_frames[i].unsqueeze(0),
                                                contact_frames=mlib_contact_frames[i].unsqueeze(0),
                                                new_motion_name="mdm_motion" + str(i).zfill(len(str(batch_size))),
                                                motion_fps=g.g_mdm_model._sequence_fps,
                                                vis_fps=15)#,

            g.update_selected_pos_flag_mesh(main_vars.mouse_world_pos)

            
        if psim.IsKeyReleased(psim.ImGuiKey_E):
            g.MainVars().motion_time = curr_motion.mlib.get_motion_length(0).item()
            curr_motion.set_to_time(g.MainVars().motion_time)
            curr_motion.update_transforms(shadow_height=curr_motion.char.get_hf_below_root(g.g_terrain))
            curr_motion.char.update_local_hf(g.g_terrain)

        if psim.IsKeyReleased(psim.ImGuiKey_C):
            optimization_settings = g.OptimizationSettings()
            contact_editing_settings = g.ContactEditingSettings()
            

            if optimization_settings.body_constraints is None:
                optimization_settings.body_constraints = [None] * curr_motion.char.char_model.get_num_joints()
                for i in range(curr_motion.char.char_model.get_num_joints()):
                    optimization_settings.body_constraints[i] = []
                #print(optimization_settings.body_constraints)
            
            body_id = contact_editing_settings.selected_body_id
            start_frame_idx = contact_editing_settings.start_frame_idx
            end_frame_idx = contact_editing_settings.end_frame_idx
            constraint_point = world_pos.clone()

            body_constraints = optimization_settings.body_constraints
            new_constraint = moopt.BodyConstraint()
            new_constraint.start_frame_idx = start_frame_idx
            new_constraint.end_frame_idx = end_frame_idx
            new_constraint.constraint_point = constraint_point
            body_constraints[body_id].append(new_constraint)

            optimization_settings.create_body_constraint_ps_mesh(
                body_id, start_frame_idx, end_frame_idx, constraint_point,
                curr_motion.char.char_model)

        if psim.IsKeyReleased(psim.ImGuiKey_N):

            if g.PathPlanningSettings().manual_placement_mode:
                g.PathPlanningSettings().place_waypoint(main_vars.selected_grid_ind)
                
                
                node_3dpos = g.g_terrain.get_xyz_point(main_vars.selected_grid_ind)
                if g.PathPlanningSettings().path_nodes is None:
                    g.PathPlanningSettings().path_nodes = node_3dpos.unsqueeze(0)
                else:
                    new_tensor = node_3dpos
                    prev_tensor = g.PathPlanningSettings().path_nodes
                    if len(prev_tensor.shape) == 1:
                        prev_tensor = prev_tensor.unsqueeze(0)
                    g.PathPlanningSettings().path_nodes = torch.cat([prev_tensor, new_tensor.unsqueeze(0)], dim=0)

                if g.PathPlanningSettings().path_nodes.shape[0] > 1:
                    g.PathPlanningSettings().visualize_path_nodes("manual_path", g.PathPlanningSettings().path_nodes)
            else:
                g.PathPlanningSettings().place_waypoint(main_vars.selected_grid_ind)


    if psim.IsKeyReleased(psim.ImGuiKey_Space):
        main_vars.motion_time = 0.0
        main_vars.paused = False
    return