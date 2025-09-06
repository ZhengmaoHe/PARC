import sys
import platform

if platform.system() == "Linux":
    print("Running on Linux, importing isaac gym")
    g_running_on_linux = True
    import envs.env_builder as env_builder
else:
    g_running_on_linux = False
    
import os
import polyscope as ps
import polyscope.imgui as psim
import torch
import math
import yaml

import anim.motion_lib as motion_lib
import util.terrain_util as terrain_util
import zmotion_editing_tools.motion_edit_lib as medit_lib

import MOTION_FORGE.include.global_header as g
import MOTION_FORGE.include.motion_editing_gui as motion_editing_gui
import MOTION_FORGE.include.contact_editing_gui as contact_editing_gui
import MOTION_FORGE.include.io as ps_io
import MOTION_FORGE.include.diffusion_gui as diffusion_gui
import MOTION_FORGE.include.terrain_gui as terrain_gui
import MOTION_FORGE.include.optimization_gui as optimization_gui
import MOTION_FORGE.include.path_planning_gui as path_planning_gui
import MOTION_FORGE.include.recording_gui as recording_gui
import MOTION_FORGE.include.ig_obs_gui as ig_obs_gui

if g_running_on_linux:
    import MOTION_FORGE.include.isaac_gym_gui as isaac_gym_gui


g.MainVars().update_time()

def time_gui():
    window_width, window_height = ps.get_window_size()
    psim.SetNextWindowPos([0, window_height - 150])
    psim.SetNextWindowSize([window_width, 150])
    psim.Begin("Motion Playback", True)
    main_vars = g.MainVars()

    curr_motion = g.MotionManager().get_curr_motion()
    
    loop_mode = curr_motion.mlib._motion_loop_modes[0]
    if not main_vars.paused:
        curr_motion.char.set_to_time(main_vars.motion_time, main_vars.motion_dt, curr_motion.mlib)
        curr_motion.update_transforms(shadow_height=curr_motion.char.get_hf_below_root(g.g_terrain))
        curr_motion.char.update_local_hf(g.g_terrain)

        for key in g.MotionManager().ik_motions:
            ik_motion = g.MotionManager().ik_motions[key]
            ik_motion.set_to_time(main_vars.motion_time)

        if main_vars.loaded_target_xy:
            curr_frame = int(math.floor(main_vars.motion_time / main_vars.motion_dt))
            if curr_frame > g.g_motion_data._data["target_xy"].shape[0] - 1:
                curr_frame = g.g_motion_data._data["target_xy"].shape[0] - 1
            curr_target_xy = torch.tensor(g.g_motion_data._data["target_xy"][curr_frame])
            curr_target_xyz = torch.cat([curr_target_xy, torch.tensor([0.0])])
            g.update_selected_pos_flag_mesh(curr_target_xyz)

        main_vars.motion_time += main_vars.curr_dt
        
        
        if (main_vars.motion_time > curr_motion.mlib.get_motion_length(0) and loop_mode != motion_lib.LoopMode.WRAP.value) or \
            main_vars.motion_time > main_vars.max_play_time:
            if main_vars.looping:
                main_vars.motion_time = 0.0
            else:
                main_vars.motion_time = min(curr_motion.mlib.get_motion_length(0).item(), main_vars.max_play_time)
                main_vars.paused = True


    g.update_dir_mesh()

    changed, main_vars.paused = psim.Checkbox("Paused", main_vars.paused)
    psim.SameLine()
    changed, main_vars.looping = psim.Checkbox("Looping", main_vars.looping)
    fps = int(curr_motion.mlib._motion_fps[0].item())
    main_vars.motion_dt = 1.0 / fps

    curr_frame_idx = main_vars.motion_time * fps

    motion_changed = False
    psim.SameLine()
    if psim.Button("Previous Frame"):
        curr_frame_idx = round(curr_frame_idx) - 1.0
        main_vars.motion_time = curr_frame_idx / fps
        motion_changed = True

    psim.SameLine()
    if psim.Button("Next Frame"):
        curr_frame_idx = round(curr_frame_idx) + 1.0
        main_vars.motion_time = curr_frame_idx / fps
        motion_changed = True

    


    slider_v_max = curr_motion.mlib._motion_lengths[0].item() if loop_mode != motion_lib.LoopMode.WRAP.value else main_vars.max_play_time
    changed1, main_vars.motion_time = psim.SliderFloat("Motion time", main_vars.motion_time, v_min = 0.0, v_max = slider_v_max)
    
    changed2, main_vars.max_play_time = psim.InputFloat("Max play time", main_vars.max_play_time)

    motion_changed = motion_changed or changed1
    if motion_changed:
        curr_motion.char.set_to_time(main_vars.motion_time, main_vars.motion_dt, curr_motion.mlib)
        curr_motion.update_transforms(shadow_height=curr_motion.char.get_hf_below_root(g.g_terrain))
        curr_motion.char.update_local_hf(g.g_terrain)

        for key in g.MotionManager().ik_motions:
            ik_motion = g.MotionManager().ik_motions[key]
            ik_motion.set_to_time(main_vars.motion_time)


    motion_info_str = "num frames: " + str(curr_motion.mlib._motion_num_frames[0].item())
    motion_info_str += ", fps: " + str(curr_motion.mlib._motion_fps[0].item())
    motion_info_str += ", curr frame: " + str(curr_frame_idx)
    psim.TextUnformatted(motion_info_str)
    psim.End()
    return

def main_loop():
    main_vars = g.MainVars()
    main_vars.update_time()
    curr_motion = g.MotionManager().get_curr_motion()
    loaded_motions = g.MotionManager().get_loaded_motions()    

    ps_io.process_IO()
    
    if psim.TreeNode("GUIs"):
        changed, main_vars.time_gui_opened = psim.Checkbox("Time GUI", main_vars.time_gui_opened)
        changed, main_vars.motion_gui_opened = psim.Checkbox("Motion GUI", main_vars.motion_gui_opened)
        changed, main_vars.contact_gui_opened = psim.Checkbox("Contact GUI", main_vars.contact_gui_opened)
        changed, main_vars.terrain_gui_opened = psim.Checkbox("Terrain GUI", main_vars.terrain_gui_opened)
        changed, main_vars.mdm_gui_opened = psim.Checkbox("MDM GUI", main_vars.mdm_gui_opened)
        changed, main_vars.optimization_gui_opened = psim.Checkbox("Optimization GUI", main_vars.optimization_gui_opened)
        changed, main_vars.path_planning_gui_opened = psim.Checkbox("Path Planning GUI", main_vars.path_planning_gui_opened)
        changed, main_vars.recording_gui_opened = psim.Checkbox("Recording GUI", main_vars.recording_gui_opened)
        changed, main_vars.isaac_gym_gui_opened = psim.Checkbox("Isaac Gym GUI", main_vars.isaac_gym_gui_opened)
        psim.TreePop()

    if main_vars.time_gui_opened:
        time_gui()

    if main_vars.motion_gui_opened and psim.Begin(name="Motion GUI", open=main_vars.motion_gui_opened):
        motion_editing_gui.motion_editor_gui()
        psim.End()

    if main_vars.contact_gui_opened and psim.Begin(name="Contact GUI", open=main_vars.contact_gui_opened):
        contact_editing_gui.contact_editing_gui()
        psim.End()
    
    if main_vars.terrain_gui_opened and psim.Begin(name="Terrain GUI", open=main_vars.terrain_gui_opened):
        terrain_gui.terrain_editing_gui()
        psim.End()
        
    if main_vars.mdm_gui_opened and psim.Begin(name="MDM GUI", open=main_vars.mdm_gui_opened):
        diffusion_gui.diffusion_gui()
        psim.End()

    if main_vars.optimization_gui_opened and psim.Begin(name="Optimization GUI", open=main_vars.optimization_gui_opened):
        optimization_gui.motion_optimization_gui()
        psim.End()

    if main_vars.path_planning_gui_opened and psim.Begin(name="Path Planning GUI", open=main_vars.path_planning_gui_opened):
        path_planning_gui.path_planning_gui()
        psim.End()

    if main_vars.recording_gui_opened and psim.Begin(name="Recording GUI", open=main_vars.recording_gui_opened):
        recording_gui.recording_gui()
        psim.End()

    if main_vars.ig_obs_gui_opened and psim.Begin(name="Obs GUI", open=main_vars.ig_obs_gui_opened):
        ig_obs_gui.ig_obs_gui()
        psim.End()

    if main_vars.isaac_gym_gui_opened and psim.Begin(name="Isaac Gym GUI", open=main_vars.isaac_gym_gui_opened):
        if g_running_on_linux:
            isaac_gym_gui.isaac_gym_gui()
        psim.End()

    
    if psim.TreeNode("Visibility"):
        if psim.Button("Hide all local hfs"):
            main_vars.viewing_local_hf = False
            for key in loaded_motions:
                loaded_motions[key].char.set_local_hf_enabled(False)

        if psim.Button("Hide all shadows"):
            for key in loaded_motions:
                loaded_motions[key].char.set_shadow_enabled(False)

        if psim.Button("Hide char"):
            curr_motion.char.set_enabled(False)

        changed, new_motion_seq_transparency = psim.SliderFloat("Motion Sequence Visibility", curr_motion.sequence.mesh.get_transparency(), v_min=0.0, v_max=1.0)
        if (changed):
            curr_motion.sequence.mesh.set_transparency(new_motion_seq_transparency)

        changed, main_vars.viewing_motion_sequence = psim.Checkbox("View motion sequence", main_vars.viewing_motion_sequence)
        if changed:
            curr_motion.sequence.mesh.set_enabled(main_vars.viewing_motion_sequence)

        changed, main_vars.local_hf_visibility = psim.SliderFloat("Local Heightfield Visibility", main_vars.local_hf_visibility, v_min=0.0, v_max=1.0)
        if (changed):
            curr_motion.char.set_local_hf_transparency(main_vars.local_hf_visibility)

        changed, main_vars.viewing_local_hf = psim.Checkbox("View local hf", main_vars.viewing_local_hf)
        if changed:
            curr_motion.char.set_local_hf_enabled(main_vars.viewing_local_hf)

        changed, main_vars.viewing_shadow = psim.Checkbox("View shadow", main_vars.viewing_shadow)
        if changed:
            curr_motion.char.set_shadow_enabled(main_vars.viewing_shadow)

        dir_mesh_visible = g.g_dir_mesh.is_enabled()
        changed, dir_mesh_visible = psim.Checkbox("Target direction visible", dir_mesh_visible)
        if changed:
            g.g_dir_mesh.set_enabled(dir_mesh_visible)

        changed, main_vars.mouse_ball_visible = psim.Checkbox("Mouse spheres visible", main_vars.mouse_ball_visible)
        if changed:
            for mesh in g.g_mouse_ball_meshes:
                mesh.set_enabled(main_vars.mouse_ball_visible)

        body_points_enabled = curr_motion.char.get_body_points_enabled()
        changed, body_points_enabled = psim.Checkbox("Body points visible", body_points_enabled)
        if changed:
            curr_motion.char.set_body_points_enabled(body_points_enabled)
        
        changed, main_vars.viewing_char = psim.Checkbox("Viewing character", main_vars.viewing_char)
        if changed:
            curr_motion.char.set_enabled(main_vars.viewing_char)

        changed, main_vars.viewing_prev_state = psim.Checkbox("Prev state visible", main_vars.viewing_prev_state)
        if changed:
            curr_motion.char.set_prev_state_enabled(main_vars.viewing_prev_state)

        changed, val = psim.Checkbox("Flag mesh", g.g_flag_mesh.is_enabled())
        if changed:
            g.g_flag_mesh.set_enabled(val)

        psim.TreePop()

        

    if psim.TreeNode("Saving"):
        changed, main_vars.save_path_with_motion = psim.Checkbox("Save path with motion", main_vars.save_path_with_motion)

        changed, main_vars.save_motion_as_loop = psim.Checkbox("Save motion as loop", main_vars.save_motion_as_loop)

        changed, main_vars.save_terrain_with_motion = psim.Checkbox("Save terrain with motion", main_vars.save_terrain_with_motion)

        if psim.Button("Set Saved Camera Params"):
            cam_params = ps.get_view_as_json()
            print(cam_params)

            g.MainVars().saved_cam_params = cam_params

        if psim.Button("Save Current Motion"):

            kwargs = dict()
            if g.OptimizationSettings().body_constraints is not None:
                kwargs["opt:body_constraints"] = g.OptimizationSettings().body_constraints

            if curr_motion.mlib._hf_mask_inds is not None:
                if curr_motion.mlib._hf_mask_inds[0] is not None:
                    kwargs["hf_mask_inds"] = curr_motion.mlib._hf_mask_inds[0]

            if main_vars.save_path_with_motion and \
                g.PathPlanningSettings().path_nodes is not None:
                kwargs["path_nodes"] = g.PathPlanningSettings().path_nodes.cpu()

            if main_vars.saved_cam_params is not None:
                kwargs["cam_params"] = main_vars.saved_cam_params

            # TODO: make this a checkbox or dropdown
            if main_vars.save_motion_as_loop:
                loop_mode = "WRAP"
            else:
                loop_mode = "CLAMP"

            if main_vars.save_terrain_with_motion:
                terrain = g.g_terrain
            else:
                terrain = None

            os.makedirs("output/_motions/", exist_ok=True)

            medit_lib.save_motion_data("output/_motions/new_motion.pkl",
                                    motion_frames = curr_motion.mlib._motion_frames,
                                    contact_frames = None if not main_vars.use_contact_info else curr_motion.mlib._frame_contacts,
                                    terrain = terrain,
                                    fps = curr_motion.mlib._motion_fps[0].item(),
                                    loop_mode = loop_mode,
                                    **kwargs)
        psim.TreePop()
    

ps.set_user_callback(main_loop)
ps.show()