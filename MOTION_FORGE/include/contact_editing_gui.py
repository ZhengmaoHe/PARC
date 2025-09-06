import torch
import polyscope as ps
import polyscope.imgui as psim
import util.torch_util as torch_util
import anim.motion_lib as motion_lib
import util.geom_util as geom_util
import util.terrain_util as terrain_util
import zmotion_editing_tools.motion_edit_lib as medit_lib
import tools.motion_opt.motion_optimization as moopt
import MOTION_FORGE.include.global_header as g

def contact_editing_gui():
    settings = g.ContactEditingSettings()
    main_vars = g.MainVars()
    curr_motion = g.MotionManager().get_curr_motion()

    selected_body_name = curr_motion.char.char_model.get_body_name(settings.selected_body_id)
    opened = psim.BeginCombo("Contact Body", selected_body_name)
    if opened:
        for body_name in curr_motion.char.char_model._body_names:
            clicked, hovering = psim.Selectable(body_name, selected_body_name==body_name)
            if clicked:
                settings.selected_body_id = curr_motion.char.char_model.get_body_id(body_name)
        psim.EndCombo()
    if psim.Button("Set contact label"):
        fps = int(curr_motion.mlib._motion_fps[0].item())
        curr_frame_idx = int(round(main_vars.motion_time * fps))
        curr_motion.mlib._frame_contacts[curr_frame_idx, settings.selected_body_id] = 1.0
    if psim.Button("Remove contact label"):
        fps = int(curr_motion.mlib._motion_fps[0].item())
        curr_frame_idx = int(round(main_vars.motion_time * fps))
        curr_motion.mlib._frame_contacts[curr_frame_idx, settings.selected_body_id] = 0.0

    changed, settings.start_frame_idx = psim.InputInt("Start frame idx", settings.start_frame_idx)
    changed, settings.end_frame_idx = psim.InputInt("End frame idx", settings.end_frame_idx)
    if settings.start_frame_idx < 0:
        settings.start_frame_idx = 0
    if settings.end_frame_idx >= curr_motion.mlib._motion_num_frames[0].item():
        settings.end_frame_idx = curr_motion.mlib._motion_num_frames[0].item() - 1
    if settings.end_frame_idx < settings.start_frame_idx:
        settings.end_frame_idx = settings.start_frame_idx
    
    if psim.Button("Set contact labels on slice of frames [inclusive]"):
        curr_motion.mlib._frame_contacts[settings.start_frame_idx:settings.end_frame_idx+1, settings.selected_body_id] = 1.0
    if psim.Button("Remove contact labels on slice of frames [inclusive]"):
        curr_motion.mlib._frame_contacts[settings.start_frame_idx:settings.end_frame_idx+1, settings.selected_body_id] = 0.0


    changed, settings.contact_eps = psim.InputFloat("contact eps", settings.contact_eps)
    if psim.Button("Compute Foot and Hand Contacts"):
        corrected_frames, contact_frames = medit_lib.compute_hf_foot_contacts_and_correct_pen(
            curr_motion.mlib._motion_frames,
            g.g_terrain,
            curr_motion.char.char_model,
            settings.contact_eps)
        
        hand_contacts = medit_lib.compute_motion_terrain_hand_contacts(curr_motion.mlib._motion_frames, 
                                                                       g.g_terrain,
                                                                       curr_motion.char.char_model, 
                                                                       settings.contact_eps)
        lh_id = curr_motion.char.char_model.get_body_id("left_hand")
        rh_id = curr_motion.char.char_model.get_body_id("right_hand")
        contact_frames[:, lh_id] = hand_contacts[:, lh_id]
        contact_frames[:, rh_id] = hand_contacts[:, rh_id]

        curr_motion.mlib = motion_lib.MotionLib(curr_motion.mlib._motion_frames.unsqueeze(0), curr_motion.char.char_model, main_vars.device, init_type="motion_frames", 
                                                contact_info=True, fps=curr_motion.mlib._motion_fps[0].item(),
                                                loop_mode=motion_lib.LoopMode.CLAMP, contacts = contact_frames)
        main_vars.use_contact_info = True
        
    if psim.Button("Correct Foot Contacts"):
        corrected_frames, contact_frames = medit_lib.compute_hf_foot_contacts_and_correct_pen(
            curr_motion.mlib._motion_frames,
            g.g_terrain,
            curr_motion.char.char_model,
            settings.contact_eps)
        
        curr_motion.mlib = motion_lib.MotionLib(corrected_frames.unsqueeze(0), curr_motion.char.char_model, main_vars.device, init_type="motion_frames", 
                                                contact_info=True, fps=curr_motion.mlib._motion_fps[0].item(),
                                                loop_mode=motion_lib.LoopMode(curr_motion.mlib.get_motion_loop_mode(0).item()), contacts = contact_frames)
        main_vars.use_contact_info = True

        main_vars.motion_time = 0.0
        curr_motion.char.set_to_time(main_vars.motion_time, main_vars.motion_dt, curr_motion.mlib)
        curr_motion.update_sequence(0.0, curr_motion.mlib._motion_lengths[0].item(), int(round(curr_motion.mlib._motion_lengths[0].item() * 15)))
        curr_motion.update_transforms(shadow_height=curr_motion.char.get_hf_below_root(g.g_terrain))

    if psim.Button("Compute contact point constraints"):
        optimization_settings = g.OptimizationSettings()

        mlib = curr_motion.mlib
        body_constraints = moopt.compute_approx_body_constraints(mlib._frame_root_pos,
                                                                 mlib._frame_root_rot,
                                                                 mlib._frame_joint_rot,
                                                                 mlib._frame_contacts,
                                                                 mlib._kin_char_model,
                                                                 g.g_terrain)
        optimization_settings.body_constraints = body_constraints
        optimization_settings.create_body_constraint_ps_meshes()


        # ps.register_point_cloud("lf points", lf_points.numpy())
        # ps.register_point_cloud("rf points", rf_points.numpy()) 
        # ps.register_point_cloud("lh points", lh_points.numpy()) 
        # ps.register_point_cloud("rh points", rh_points.numpy())
        #print(lf_contact_frame_ids)


        # TODO: optimize points so they are on the surface of the terrain
        # For each point, slice a 5x5 terrain centered around the point and then do sdf optimization.
        # If point is too far from surface at the start, then just mark it as erroneous and remove it?

    return