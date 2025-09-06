import torch
import random
import numpy as np
import copy
import polyscope as ps
import polyscope.imgui as psim
import anim.kin_char_model as kin_char_model
import anim.motion_lib as motion_lib
import util.torch_util as torch_util
import util.terrain_util as terrain_util
import util.geom_util as geom_util
import zmotion_editing_tools.motion_edit_lib as medit_lib
import util.tb_logger as tb_logger
import MOTION_FORGE.include.global_header as g

import tools.motion_opt.motion_optimization as moopt

def motion_optimization_gui():
    settings = g.OptimizationSettings()
    main_vars = g.MainVars()
    curr_motion = g.MotionManager().get_curr_motion()

    changed, settings.num_iters = psim.InputInt("num iters:", settings.num_iters)
    changed, settings.step_size = psim.InputFloat("optimization step size", settings.step_size)
    changed, settings.w_root_pos = psim.InputFloat("w_root_pos", settings.w_root_pos)
    changed, settings.w_root_rot = psim.InputFloat("w_root_rot", settings.w_root_rot)
    changed, settings.w_joint_rot = psim.InputFloat("w_joint_rot", settings.w_joint_rot)
    changed, settings.w_smoothness = psim.InputFloat("w_smoothness", settings.w_smoothness)
    changed, settings.w_penetration = psim.InputFloat("w_penetration", settings.w_penetration)
    changed, settings.w_contact = psim.InputFloat("w_contact", settings.w_contact)
    changed, settings.w_sliding = psim.InputFloat("w_sliding", settings.w_sliding)
    changed, settings.w_body_constraints = psim.InputFloat("w_body_constraints", settings.w_body_constraints)
    changed, settings.w_jerk = psim.InputFloat("w_jerk", settings.w_jerk)
    changed, settings.max_jerk = psim.InputFloat("max_jerk", settings.max_jerk)
    changed, settings.use_wandb = psim.Checkbox("Use wandb", settings.use_wandb)
    #changed, settings.auto_compute_body_constraints = psim.Checkbox("Auto compute body constraints", settings.auto_compute_body_constraints)
    if settings.body_constraints is not None:
        num_bodies = curr_motion.char.char_model.get_num_joints()
        assert len(settings.body_constraints) == num_bodies
        if psim.TreeNode("Body Contact Point Constraints"):
            for b in range(num_bodies):
                curr_body_name = curr_motion.char.char_model.get_body_name(b)
                if psim.TreeNode(curr_body_name):
                    num_constraints = len(settings.body_constraints[b])
                    info_str = "num constraints: " + str(num_constraints)
                    psim.TextUnformatted(info_str)
                    if num_constraints > 0:
                        for c_idx in range(num_constraints):
                            constraint = settings.body_constraints[b][c_idx]
                            if psim.TreeNode("constraint " + str(c_idx)):
                                changed, constraint.start_frame_idx = psim.InputInt("start frame idx", constraint.start_frame_idx)
                                changed, constraint.end_frame_idx = psim.InputInt("end frame idx", constraint.end_frame_idx)

                                np_constraint_pt = constraint.constraint_point.numpy()
                                changed3, np_constraint_pt = psim.InputFloat3("pt", np_constraint_pt)
                                if changed3:
                                    constraint.constraint_point = torch.tensor(np_constraint_pt, dtype=torch.float32, device=g.MainVars().device)
                                    g.OptimizationSettings().create_body_constraint_ps_mesh(
                                        b, constraint.start_frame_idx, constraint.end_frame_idx, constraint.constraint_point,
                                        curr_motion.char.char_model)
                                #psim.TextUnformatted(np.array2string(constraint.constraint_point.numpy()))

                                if psim.Button("Test sdf"):
                                    # NOTE: only works for single sphere bodies right now
                                    curr_body_rot = curr_motion.char.get_body_rot(b)
                                    offset = curr_motion.char.char_model.get_geoms(b)[0]._offset
                                    radius = curr_motion.char.char_model.get_geoms(b)[0]._dims
                                    curr_body_pos = curr_motion.char.get_body_pos(b)
                                    body_center = torch_util.quat_rotate(curr_body_rot, offset) + curr_body_pos
                                    print(constraint.constraint_point)
                                    print(body_center)
                                    print(radius)
                                    sd = geom_util.sdSphere(constraint.constraint_point, body_center, radius)
                                    print("signed distance:", sd.item())

                                psim.TreePop()
                    psim.TreePop()
            psim.TreePop()
    
    if psim.Button("Clear body constraints"):
        settings.clear_body_constraints()

    if psim.Button("Motion optimization"):
        # move everything to GPU before starting optimization
        
        # slice the terrain around the motion so that optimization is more efficient
        terrain = terrain_util.slice_terrain_around_motion(curr_motion.mlib._motion_frames,
                                                           g.g_terrain,
                                                           localize=False)
        

        # move everything to GPU before starting optimization
        if g.MainVars().device == "cpu":
            src_frames = curr_motion.mlib._motion_frames.to(device="cuda:0")
            contact_frames = curr_motion.mlib._frame_contacts.to(device="cuda:0")
            terrain.to_torch(device="cuda:0")
            char_model = kin_char_model.KinCharModel("cuda:0")
            char_model.load_char_file(g.g_char_filepath)
            

            

            if settings.body_constraints is not None:
                body_constraints = []
                for b in range(len(settings.body_constraints)):
                    body_constraints.append([])
                    for c_idx in range(len(settings.body_constraints[b])):
                        curr_body_constraint = settings.body_constraints[b][c_idx]
                        body_constraints[b].append(copy.deepcopy(curr_body_constraint))
                        body_constraints[b][c_idx].constraint_point = body_constraints[b][c_idx].constraint_point.to(device="cuda:0")
            else:
                body_constraints = settings.body_constraints
                
        else:
            src_frames = curr_motion.mlib._motion_frames
            contact_frames = curr_motion.mlib._frame_contacts
            char_model = curr_motion.char.char_model
            body_constraints = settings.body_constraints

        body_points = geom_util.get_char_point_samples(char_model)

        opt_frames = moopt.motion_contact_optimization(
            src_frames=src_frames,
            contacts=contact_frames,
            body_points=body_points,
            terrain=terrain,
            char_model=char_model,
            num_iters=settings.num_iters,
            step_size=settings.step_size,
            w_root_pos=settings.w_root_pos,
            w_root_rot=settings.w_root_rot,
            w_joint_rot=settings.w_joint_rot,
            w_smoothness=settings.w_smoothness,
            w_penetration=settings.w_penetration,
            w_contact=settings.w_contact,
            w_sliding=settings.w_sliding,
            w_body_constraints=settings.w_body_constraints,
            w_jerk=settings.w_jerk,
            body_constraints=body_constraints,
            max_jerk=settings.max_jerk,
            exp_name=None,
            use_wandb=settings.use_wandb,
            log_file="output/opt_log.txt")
        
        if opt_frames.device != main_vars.device:
            opt_frames = opt_frames.to(device=main_vars.device)
        
        new_mlib = motion_lib.MotionLib(opt_frames.unsqueeze(0), curr_motion.char.char_model, main_vars.device, init_type="motion_frames", 
                                        contact_info=True, fps=curr_motion.mlib._motion_fps[0].item(),
                                        loop_mode=motion_lib.LoopMode.CLAMP, contacts = curr_motion.mlib._frame_contacts)
        main_vars.use_contact_info = True

        curr_motion.deselect()
        new_motion = g.MDMMotionPS("optimized_motion", new_mlib, [0.4, 0.8, 0.2])
        g.MotionManager().add_motion(new_motion, "optimized_motion")
        g.MotionManager().set_curr_motion("optimized_motion")
        curr_motion = g.MotionManager().get_curr_motion()
        main_vars.motion_time = 0.0
        curr_motion.char.set_to_time(main_vars.motion_time, main_vars.motion_dt, curr_motion.mlib)
        curr_motion.update_sequence(0.0, curr_motion.mlib._motion_lengths[0].item(), int(round(curr_motion.mlib._motion_lengths[0].item() * 15)))
        curr_motion.update_transforms(shadow_height=curr_motion.char.get_hf_below_root(g.g_terrain))
        curr_motion.char.update_local_hf(g.g_terrain)
        curr_motion.select()

    # if psim.Button("Batch optimize motions"):
    #     import os
    #     input_folder_path = "../Data/simple_0_6_height_motions/v_10/"
    #     assert os.path.isdir(input_folder_path)
    #     files = os.listdir(input_folder_path)
    #     files = [os.path.join(input_folder_path, f) for f in files if os.path.splitext(f)[1] == ".pkl"]

    #     output_folder_path = input_folder_path + "opt/"
    #     os.makedirs(output_folder_path, exist_ok=True)

    #     for file in files:
    #         motion_data = medit_lib.load_motion_file(file)
    #         terrain = motion_data.get_terrain()
    #         src_frames = motion_data.get_frames()
    #         contacts = motion_data.get_contacts()




    #         opt_frames = moopt.motion_contact_optimization(
    #             src_frames=src_frames, 
    #             contacts=contacts,
    #             body_points=curr_motion.char.get_body_point_samples(),
    #             terrain=terrain,
    #             char_model=curr_motion.char.char_model,
    #             num_iters=settings.num_iters,
    #             step_size=settings.step_size,
    #             w_root_pos=settings.w_root_pos,
    #             w_root_rot=settings.w_root_rot,
    #             w_joint_rot=settings.w_joint_rot,
    #             w_smoothness=settings.w_smoothness,
    #             w_penetration=settings.w_penetration,
    #             w_contact=settings.w_contact,
    #             w_sliding=settings.w_sliding,
    #             exp_name=None,
    #             use_wandb=True)
            
    #         output_filepath = output_folder_path + os.path.basename(os.path.splitext(file)[0]) + "_opt.pkl"
            

    #         motion_data = {
    #             'fps': 30,
    #             'loop_mode': "CLAMP",
    #             'frames': opt_frames,
    #             'contacts': contacts,
    #             'terrain': terrain
    #         }
    #         motion_data = medit_lib.MotionData(motion_data)
    #         motion_data.save_to_file(output_filepath)

    return