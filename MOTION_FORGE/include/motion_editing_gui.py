import os
import polyscope.imgui as psim
import polyscope as ps
import numpy as np
import torch
import util.torch_util as torch_util
import anim.motion_lib as motion_lib
import anim.kin_char_model as kin_char_model
import zmotion_editing_tools.motion_edit_lib as medit_lib
import util.geom_util as geom_util
import util.terrain_util as terrain_util
import util.motion_util as motion_util
import MOTION_FORGE.polyscope_util as ps_util
import MOTION_FORGE.include.global_header as g

########## MOTION EDITING GUI ##########
def motion_editor_gui():
    settings = g.MotionEditorSettings()
    main_vars = g.MainVars()

    curr_motion = g.MotionManager().get_curr_motion()
    loaded_motions = g.MotionManager().get_loaded_motions()

    if psim.TreeNode("Character State"):

        if psim.Button("Set to Zero-Pose"):
            curr_motion.char.set_to_zero_pose()
            curr_motion.char.forward_kinematics()

        #root_pos_str = "root pos: " + np.array2string(curr_motion.char.motion_frames[0, -1, 0:3].cpu().numpy())
        #psim.TextUnformatted(root_pos_str)
        char_state = curr_motion.char.motion_frames
        root_pos = char_state.root_pos[-1].cpu().numpy()
        root_pos_changed, new_root_pos = psim.InputFloat3("root pos", root_pos)
        if root_pos_changed:
            new_root_pos = torch.from_numpy(np.array(new_root_pos)).to(dtype=torch.float32, device=main_vars.device)
            curr_motion.char.set_root_pos(new_root_pos)


        #root_rot_str = "root rot: " + np.array2string(curr_motion.char.motion_frames[0, -1, 3:6].cpu().numpy())
        #psim.TextUnformatted(root_rot_str)
        root_rot = char_state.root_rot[-1].cpu().numpy()
        root_rot_changed, new_root_rot = psim.InputFloat4("root rot", root_rot)
        if root_rot_changed:
            new_root_rot = torch.from_numpy(np.array(new_root_rot)).to(dtype=torch.float32, device=main_vars.device)
            curr_motion.char.set_root_rot_quat(new_root_rot)

        # root_vel_str = "root vel: " + np.array2string(curr_motion.char.get_root_vel().cpu().numpy())
        # psim.TextUnformatted(root_vel_str)

        heading_str = "heading :" + str(torch_util.calc_heading(char_state.root_rot[-1]).item())
        psim.TextUnformatted(heading_str)

        if psim.TreeNode("Body Positions"):
            for b in range(curr_motion.char.char_model.get_num_joints()):
                body_pos_str = curr_motion.char.char_model.get_body_name(b) + " pos: "
                body_pos = curr_motion.char.get_body_pos(b).cpu().numpy()
                body_pos_str += np.array2string(body_pos)
                psim.TextUnformatted(body_pos_str)
            psim.TreePop()

        any_dof_changed = False
        if psim.TreeNode("Joint DOFs"):
            joint_dofs = curr_motion.char.char_model.rot_to_dof(char_state.joint_rot[-1])
            #psim.TextUnformatted(np.array2string(joint_dofs.cpu().numpy()))

            for b in range(1, curr_motion.char.char_model.get_num_joints()):
                curr_joint_name = curr_motion.char.char_model.get_body_name(b)
                curr_joint_dof = curr_motion.char.char_model.get_joint(b).get_joint_dof(joint_dofs)

                if curr_joint_dof.shape[0] == 3:
                    changed, new_joint_dof = psim.InputFloat3(curr_joint_name, curr_joint_dof.cpu().numpy(), format="%.6f")
                    if changed:
                        curr_joint_dof[0] = new_joint_dof[0]
                        curr_joint_dof[1] = new_joint_dof[1]
                        curr_joint_dof[2] = new_joint_dof[2]
                        any_dof_changed = True
                elif curr_joint_dof.shape[0] == 1:
                    changed, new_joint_dof = psim.InputFloat(curr_joint_name, curr_joint_dof[0].item(), format="%.6f")
                    if changed:
                        curr_joint_dof[0] = new_joint_dof
                        any_dof_changed = True
                else:
                    psim.TextUnformatted(curr_joint_name + ": no dofs")

            psim.TreePop()

        if psim.TreeNode("Joint Limits"):
            if psim.Button("Rectangular Projection for all dofs"):
                joint_dofs = curr_motion.char.char_model.rot_to_dof(char_state.joint_rot[-1])
                joint_dofs[:] = curr_motion.char.char_model.apply_joint_dof_limits(joint_dofs)
                any_dof_changed = True

            for b in range(1, curr_motion.char.char_model.get_num_joints()):
                curr_joint_name = curr_motion.char.char_model.get_body_name(b)
                curr_joint = curr_motion.char.char_model.get_joint(b)

                if curr_joint.joint_type == kin_char_model.JointType.HINGE: # hinge
                    joint_limit_str = "min: " + str(curr_joint.limits[0].item()) + ", max: " + str(curr_joint.limits[1].item())
                    psim.TextUnformatted(curr_joint_name + " " + joint_limit_str)
                    if psim.Button(curr_joint_name + " Rectangular Projection"):
                        joint_dofs = curr_motion.char.motion_frames[0, -1, 6:]
                        curr_joint_dof = curr_joint.get_joint_dof(joint_dofs)
                        curr_joint_dof[:] = torch.clamp(curr_joint_dof, min=curr_joint.limits[0], max=curr_joint.limits[1])
                        any_dof_changed = True

                elif curr_joint.joint_type == kin_char_model.JointType.SPHERICAL:
                    joint_limit_str_x = "x_min: " + str(curr_joint.limits[0, 0].item()) + ", x_max: " + str(curr_joint.limits[0, 1].item())
                    joint_limit_str_y = "y_min: " + str(curr_joint.limits[1, 0].item()) + ", y_max: " + str(curr_joint.limits[1, 1].item())
                    joint_limit_str_z = "z_min: " + str(curr_joint.limits[2, 0].item()) + ", z_max: " + str(curr_joint.limits[2, 1].item())

                    psim.TextUnformatted(curr_joint_name + " " + joint_limit_str_x)
                    psim.TextUnformatted(curr_joint_name + " " + joint_limit_str_y)
                    psim.TextUnformatted(curr_joint_name + " " + joint_limit_str_z)

                    if psim.Button(curr_joint_name + " Rectangular Projection"):
                        assert False, "TODO"
                        joint_dofs = curr_motion.char.motion_frames[0, -1, 6:]
                        curr_joint_dof = curr_joint.get_joint_dof(joint_dofs)
                        curr_joint_dof[:] = torch.clamp(curr_joint_dof, min=curr_joint.limits[:, 0], max=curr_joint.limits[:, 1])
                        any_dof_changed = True
                else:
                    psim.TextUnformatted(curr_joint_name + ": no joint limits")

            psim.TreePop()

        if root_pos_changed or root_rot_changed or any_dof_changed:
            if any_dof_changed:
                char_state.joint_rot[-1] = curr_motion.char.char_model.dof_to_rot(joint_dofs)

            curr_motion.char.forward_kinematics()


        psim.TreePop()


    if psim.TreeNode("Character Mesh"): # TODO: move somewhere else

        changed, settings.x_scale = psim.InputFloat("x_scale", settings.x_scale)
        changed, settings.y_scale = psim.InputFloat("y_scale", settings.y_scale)
        changed, settings.z_scale = psim.InputFloat("z_scale", settings.z_scale)
        if psim.Button("Scale"):
            curr_motion.char.x_scale = settings.x_scale
            curr_motion.char.y_scale = settings.y_scale
            curr_motion.char.z_scale = settings.z_scale
            curr_motion.char.create_char_mesh()

        psim.TreePop()

    opened = psim.BeginCombo("Motion File", os.path.basename(g.g_motion_filepath))
    if opened:
        for filepath in g.g_motion_filepaths:
            filename = os.path.basename(filepath)
            _, selected = psim.Selectable(filename, g.g_motion_filepath==filepath)
            if selected:
                g.g_motion_filepath = filepath
        psim.EndCombo()

    changed, settings.load_terrain_with_motion = psim.Checkbox("load terrain with motion", settings.load_terrain_with_motion)
    if psim.Button("Load Motion"):
        curr_motion.set_enabled(False, False)
        motion_name = os.path.basename(g.g_motion_filepath)
        if motion_name not in loaded_motions:
            g.MotionManager().load_motion(filepath=g.g_motion_filepath,
                                       char_model_path=g.g_char_filepath,
                                       name=motion_name)
        else:
            g.MotionManager().set_curr_motion(motion_name)
        main_vars.motion_time = 0.0
        curr_motion.char.set_to_time(main_vars.motion_time, main_vars.motion_dt, curr_motion.mlib)
        curr_motion.update_transforms(shadow_height=curr_motion.char.get_hf_below_root(g.g_terrain))
        curr_motion.char.update_local_hf(g.g_terrain)

        if settings.load_terrain_with_motion:
            motion_data = medit_lib.load_motion_file(g.g_motion_filepath)
            assert "terrain" in motion_data._data

            g.g_terrain = motion_data.get_terrain()
            g.TerrainMeshManager().rebuild()

            if "cam_params" in motion_data._data:
                g.MainVars().saved_cam_params = motion_data._data["cam_params"]
            else:
                g.MainVars().saved_cam_params = None

            g.OptimizationSettings().clear_body_constraints()
            if "opt:body_constraints" in motion_data._data:
                g.OptimizationSettings().body_constraints = motion_data._data["opt:body_constraints"]
                g.OptimizationSettings().create_body_constraint_ps_meshes()

        curr_motion.char.update_local_hf(g.g_terrain)

    if psim.Button("Hide all motions"):
        for key in loaded_motions:
            loaded_motions[key].set_disabled()

    if psim.Button("See all motions"):
        for key in loaded_motions:
            loaded_motions[key].set_enabled(main_vars.viewing_motion_sequence, True)

    if psim.TreeNode("Loaded motions"):
        motions_to_delete = set()
        for key in loaded_motions:
            #psim.TextUnformatted(key)
            if psim.TreeNode(key):
                selected = key == curr_motion.name
                changed, selected = psim.Checkbox("selected", selected)
                if changed:
                    curr_motion.deselect()
                    #g.g_motion = loaded_motions[key]
                    g.MotionManager().set_curr_motion(key, sequence_val=main_vars.viewing_motion_sequence)
                    curr_motion.select()

                visible = loaded_motions[key].sequence.mesh.is_enabled()
                changed2, visible = psim.Checkbox("visible", visible)
                if changed2:
                    loaded_motions[key].set_enabled(main_vars.viewing_motion_sequence and visible, visible)

                if psim.Button("delete motion"):
                    if curr_motion.name == key:
                        print("can't delete current motion")
                    else:
                        motions_to_delete.add(key)
                psim.TreePop()

        for motion_name in motions_to_delete:
            loaded_motions[motion_name].remove()
            del loaded_motions[motion_name]
        psim.TreePop()

    if psim.TreeNode("Motion Editor"):

        changed, main_vars.using_sliders = psim.Checkbox("use sliders", main_vars.using_sliders)
        changed, settings.editing_full_sequence = psim.Checkbox("Editing full sequence", settings.editing_full_sequence)
        if main_vars.using_sliders:
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
            curr_motion.char.set_to_time(main_vars.motion_time + curr_motion.time_offset, main_vars.motion_dt, curr_motion.mlib)
            curr_motion.update_transforms(transform_full_sequence=settings.editing_full_sequence)
            curr_motion.char.update_local_hf(g.g_terrain)



        changed, settings.medit_heading_rot_angle = psim.InputFloat("Edit heading rot angle", settings.medit_heading_rot_angle)
        changed, settings.medit_start_frame = psim.InputInt("Edit start frame", settings.medit_start_frame)
        if changed:
            settings.medit_start_frame = max(settings.medit_start_frame, 0)
            settings.medit_start_frame = min(settings.medit_start_frame, curr_motion.mlib._motion_frames.shape[0]-1)

        changed, settings.medit_end_frame = psim.InputInt("Edit end frame", settings.medit_end_frame)
        if changed:
            settings.medit_end_frame = max(settings.medit_end_frame, 0)
            settings.medit_end_frame = min(settings.medit_end_frame, curr_motion.mlib._motion_frames.shape[0]-1)
        if settings.medit_end_frame < settings.medit_start_frame:
            settings.medit_end_frame = settings.medit_start_frame

        changed, settings.medit_scale = psim.InputFloat("Edit scale", settings.medit_scale)

        changed, settings.speed_scale = psim.InputFloat("Speed scale", settings.speed_scale)

        if psim.Button("Scale Motion"):
            # TODO: squish only a certain segment, given a start and end frame
            new_frames = curr_motion.mlib._motion_frames.clone()


            medit_lib.scale_motion_segment(new_frames, settings.medit_scale, settings.medit_start_frame, settings.medit_end_frame)

            if not main_vars.use_contact_info:
                new_mlib = motion_lib.MotionLib(new_frames.unsqueeze(0), curr_motion.char.char_model, main_vars.device, init_type="motion_frames",
                                                    contact_info=main_vars.use_contact_info, fps=curr_motion.mlib._motion_fps[0].item(),
                                                    loop_mode=motion_lib.LoopMode(curr_motion.mlib.get_motion_loop_mode(0).item()))
            else:
                new_mlib = motion_lib.MotionLib(new_frames.unsqueeze(0), curr_motion.char.char_model, main_vars.device, init_type="motion_frames",
                                                    contact_info=main_vars.use_contact_info, fps=curr_motion.mlib._motion_fps[0].item(),
                                                    loop_mode=motion_lib.LoopMode(curr_motion.mlib.get_motion_loop_mode(0).item()), 
                                                    contacts = curr_motion.mlib._frame_contacts)
            g.update_current_motion_mlib(new_mlib)

        if psim.Button("Rotate Motion"):
            new_frames = curr_motion.mlib._motion_frames.clone()
            new_root_rots = torch_util.exp_map_to_quat(new_frames[settings.medit_start_frame:, 3:6])
            rot = torch_util.axis_angle_to_quat(torch.tensor([0., 0., 1.], dtype=torch.float32, device=main_vars.device),
                                                torch.tensor([settings.medit_heading_rot_angle / 180 * torch.pi], dtype=torch.float32, device=main_vars.device))
            new_root_rots = torch_util.quat_multiply(rot, new_root_rots)
            new_frames[settings.medit_start_frame:, 3:6] = torch_util.quat_to_exp_map(new_root_rots)
            canon_pos = new_frames[settings.medit_start_frame, 0:3].clone()
            new_frames[settings.medit_start_frame:, 0:3] -= canon_pos
            new_frames[settings.medit_start_frame:, 0:3] = torch_util.quat_rotate(rot, new_frames[settings.medit_start_frame:, 0:3])
            new_frames[settings.medit_start_frame:, 0:3] += canon_pos

            if not main_vars.use_contact_info:
                curr_motion.mlib = motion_lib.MotionLib(new_frames.unsqueeze(0), curr_motion.char.char_model, main_vars.device, init_type="motion_frames",
                                                    contact_info=main_vars.use_contact_info, fps=curr_motion.mlib._motion_fps[0].item(),
                                                    loop_mode=motion_lib.LoopMode.CLAMP)
            else:
                curr_motion.mlib = motion_lib.MotionLib(new_frames.unsqueeze(0), curr_motion.char.char_model, main_vars.device, init_type="motion_frames",
                                                    contact_info=main_vars.use_contact_info, fps=curr_motion.mlib._motion_fps[0].item(),
                                                    loop_mode=motion_lib.LoopMode.CLAMP, contacts = curr_motion.mlib._frame_contacts)
            curr_motion.mlib = curr_motion.mlib
            main_vars.motion_time = 0.0
            curr_motion.char.set_to_time(main_vars.motion_time, main_vars.motion_dt, curr_motion.mlib)
            curr_motion.update_sequence(0.0, curr_motion.mlib._motion_lengths[0].item(), int(curr_motion.mlib._motion_lengths[0].item()) * 15)
            curr_motion.update_transforms(shadow_height=curr_motion.char.get_hf_below_root(g.g_terrain))

        if psim.Button("Scale speed"):


            dt = 1.0 / curr_motion.mlib._motion_fps[0].item() * settings.speed_scale

            motion_times = torch.arange(0.0, curr_motion.mlib._motion_lengths[0].item(), dt, dtype=torch.float32, device=main_vars.device)
            motion_ids = torch.zeros_like(motion_times, dtype=torch.int64)

            if main_vars.use_contact_info:
                root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, contacts = curr_motion.mlib.calc_motion_frame(motion_ids, motion_times)

                root_rot = torch_util.quat_to_exp_map(root_rot)
                joint_dof = curr_motion.char.char_model.rot_to_dof(joint_rot)
                motion_frames = torch.cat([root_pos, root_rot, joint_dof], dim=-1)
                new_mlib = motion_lib.MotionLib(motion_frames.unsqueeze(0), curr_motion.char.char_model, main_vars.device, init_type="motion_frames",
                                                    contact_info=main_vars.use_contact_info, fps=curr_motion.mlib._motion_fps[0].item(),
                                                    loop_mode=motion_lib.LoopMode.CLAMP, contacts = contacts.unsqueeze(0))
            else:
                root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel = curr_motion.mlib.calc_motion_frame(motion_ids, motion_times)
                root_rot = torch_util.quat_to_exp_map(root_rot)
                joint_dof = curr_motion.char.char_model.rot_to_dof(joint_rot)
                motion_frames = torch.cat([root_pos, root_rot, joint_dof], dim=-1)
                new_mlib = motion_lib.MotionLib(motion_frames.unsqueeze(0), curr_motion.char.char_model, main_vars.device, init_type="motion_frames",
                                                    contact_info=main_vars.use_contact_info, fps=curr_motion.mlib._motion_fps[0].item(),
                                                    loop_mode=motion_lib.LoopMode.CLAMP)

            g.update_current_motion_mlib(new_mlib)


        if(psim.Button("Apply transforms to motion data")):
            curr_motion.apply_transforms_to_motion_data(fps=2)
        if psim.Button("Apply transforms to nearest frame"):
            fps = int(curr_motion.mlib._motion_fps[0].item())
            curr_frame_idx = int(round(main_vars.motion_time * fps))
            curr_motion.apply_transforms_to_motion_data(slice(curr_frame_idx, curr_frame_idx+1), fps=2)
        if psim.Button("Apply transforrms to slice of frames"):
            curr_motion.apply_transforms_to_motion_data(slice(settings.medit_start_frame, settings.medit_end_frame+1), fps=2)

        changed, settings.motion_slice_start_time = psim.InputFloat("motion slice start time", settings.motion_slice_start_time)
        changed, settings.motion_slice_end_time = psim.InputFloat("motion slice end time", settings.motion_slice_end_time)

        if psim.Button("Cut frames [inclusive]"):
            new_frames = medit_lib.cut_motion(curr_motion.mlib._motion_frames, settings.medit_start_frame, settings.medit_end_frame)
            print("new num frames:", new_frames.shape[0])

            if main_vars.use_contact_info:
                new_contact_frames = medit_lib.cut_motion(curr_motion.mlib._frame_contacts, settings.medit_start_frame, settings.medit_end_frame)
            else:
                new_contact_frames = None

            new_mlib = motion_lib.MotionLib(new_frames.unsqueeze(0), curr_motion.char.char_model, main_vars.device, init_type="motion_frames",
                                                contact_info=main_vars.use_contact_info, fps=curr_motion.mlib._motion_fps[0].item(),
                                                loop_mode=motion_lib.LoopMode.CLAMP, contacts = new_contact_frames)

            g.update_current_motion_mlib(new_mlib)

        if psim.Button("Slice motion"):
            if not main_vars.use_contact_info:
                new_frames = medit_lib.slice_motion(curr_motion.mlib._motion_frames, settings.motion_slice_start_time,
                                                settings.motion_slice_end_time, curr_motion.mlib._motion_fps[0].item())

                new_mlib = motion_lib.MotionLib(new_frames.unsqueeze(0), curr_motion.char.char_model, main_vars.device, init_type="motion_frames",
                                                    contact_info=main_vars.use_contact_info, fps=curr_motion.mlib._motion_fps[0].item(),
                                                    loop_mode=motion_lib.LoopMode.CLAMP)
            else:
                new_frames = medit_lib.slice_motion(curr_motion.mlib._motion_frames, settings.motion_slice_start_time,
                                                settings.motion_slice_end_time, curr_motion.mlib._motion_fps[0].item())

                new_contact_frames = medit_lib.slice_motion(curr_motion.mlib._frame_contacts, settings.motion_slice_start_time,
                                                settings.motion_slice_end_time, curr_motion.mlib._motion_fps[0].item())

                new_mlib = motion_lib.MotionLib(new_frames.unsqueeze(0), curr_motion.char.char_model, main_vars.device, init_type="motion_frames",
                                                    contact_info=main_vars.use_contact_info, fps=curr_motion.mlib._motion_fps[0].item(),
                                                    loop_mode=motion_lib.LoopMode.CLAMP, contacts = new_contact_frames)

            g.update_current_motion_mlib(new_mlib)

        if psim.Button("Stride Frames"):
            new_frames = medit_lib.stride_motion(curr_motion.mlib._motion_frames, settings.medit_start_frame, settings.medit_end_frame, 2)
            print("new num frames:", new_frames.shape[0])

            if main_vars.use_contact_info:
                new_contact_frames = medit_lib.stride_motion(curr_motion.mlib._frame_contacts, settings.medit_start_frame, settings.medit_end_frame, 2)
            else:
                new_contact_frames = None

            curr_motion.mlib = motion_lib.MotionLib(new_frames.unsqueeze(0), curr_motion.char.char_model, main_vars.device, init_type="motion_frames",
                                                contact_info=main_vars.use_contact_info, fps=curr_motion.mlib._motion_fps[0].item(),
                                                loop_mode=motion_lib.LoopMode.CLAMP, contacts = new_contact_frames)

            main_vars.motion_time = 0.0
            curr_motion.char.set_to_time(main_vars.motion_time, main_vars.motion_dt, curr_motion.mlib)
            curr_motion.update_sequence(0.0, curr_motion.mlib._motion_lengths[0].item(), int(curr_motion.mlib._motion_lengths[0].item()) * 15)
            curr_motion.update_transforms(shadow_height=curr_motion.char.get_hf_below_root(g.g_terrain))

        if psim.Button("Find and remove hesitation frames"):

            mlib = curr_motion.mlib
            new_motion_frames, new_contact_frames = medit_lib.remove_hesitation_frames(mlib._motion_frames, mlib._frame_contacts,
                                                                                        mlib._kin_char_model,
                                                                                        verbose=True)


            g.MotionManager().make_new_motion(new_motion_frames, new_contact_frames,
                                curr_motion.name, 30, 15)
        psim.TreePop()

    if psim.TreeNode("Motion Stats"):
        if psim.Button("Body derivatives"):
            mlib = curr_motion.mlib
            root_pos = mlib._frame_root_pos
            root_rot = mlib._frame_root_rot
            joint_rot = mlib._frame_joint_rot

            body_pos, body_rot = curr_motion.char.char_model.forward_kinematics(root_pos, root_rot, joint_rot)

            dt = 1.0 / mlib._motion_fps[0].item()

            body_vel = (body_pos[1:] - body_pos[:-1]) / dt
            body_acc = (body_vel[1:] - body_vel[:-1]) / dt
            body_jerk = (body_acc[1:] - body_acc[:-1]) / dt

            vel_mag = body_vel.norm(dim=-1)
            acc_mag = body_acc.norm(dim=-1)
            jerk_mag = body_jerk.norm(dim=-1)

            max_jerk, max_frame_idx = torch.max(jerk_mag, dim=0)

            print("Max speed:", torch.max(vel_mag))
            print("Max acceleration (magnitude):", torch.max(acc_mag))
            print("Max jerk (magnitude):", max_jerk)
            print("Max jerk frame idx:", max_frame_idx)

            print("Mean speed:", torch.mean(vel_mag))
            print("Mean acceleration (magnitude):", torch.mean(acc_mag))
            print("Mean jerk (magnitude):", torch.mean(jerk_mag))

            print("Std speed:", torch.std(vel_mag))
            print("Std acceleration (magnitude):", torch.std(acc_mag))
            print("Std jerk (magnitude):", torch.std(jerk_mag))

        if psim.Button("Compute motion contact loss"):
            from tools.procgen.mdm_path import compute_motion_loss

            mlib = curr_motion.mlib
            root_pos = mlib._frame_root_pos
            root_rot = mlib._frame_root_rot
            joint_rot = mlib._frame_joint_rot
            contacts = mlib._frame_contacts

            body_points = geom_util.get_char_point_samples(mlib._kin_char_model)

            motion_frames = motion_util.MotionFrames(root_pos=root_pos, root_rot = root_rot, joint_rot=joint_rot, contacts=contacts)
            motion_frames = motion_frames.unsqueeze(0)
            losses = compute_motion_loss(motion_frames, None, g.g_terrain, mlib._kin_char_model, body_points, w_contact=1.0, w_pen=1.0, w_path=0.0)
            for key in losses:
                print(key +":", losses[key].item())

        if psim.TreeNode("Plot positions"):
            # Plots for every single body part
            mlib = g.MotionManager().get_curr_motion().mlib
            char_model = mlib._kin_char_model
            device = g.MotionManager().get_curr_motion().mlib._device
            dt = mlib._motion_dt[0].item()
            num_frames = mlib._motion_num_frames[0].item()
            motion_ids = torch.zeros(size=[num_frames], dtype=torch.int64, device=device)

            all_motion_times = torch.arange(start=0, end=num_frames, dtype=torch.float32, device=device) * dt
            curr_motion_time = torch.tensor([g.MainVars().motion_time], dtype=torch.float32, device=device)
            single_motion_id = torch.tensor([0], dtype=torch.int64, device=device)

            if mlib._contact_info:
                root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, contacts = mlib.calc_motion_frame(motion_ids, all_motion_times)
            else:
                root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel = mlib.calc_motion_frame(motion_ids, all_motion_times)

            body_pos, body_rot = char_model.forward_kinematics(root_pos, root_rot, joint_rot)

            frame_idx0, frame_idx1, blend = g.MotionManager().get_curr_motion().mlib._calc_frame_blend(single_motion_id, curr_motion_time)
            offset = frame_idx0.item()

            if psim.TreeNode("Body positions"):
                body_pos_np = body_pos.cpu().numpy()
                for b in range(mlib._kin_char_model.get_num_joints()):
                    body_name = char_model.get_body_name(b)

                    if psim.TreeNode(body_name):

                        xyz_str = ["x", "y", "z"]

                        for i in range(3):
                            body_pos_1d = body_pos_np[:, b, i]

                            min_pos = np.min(body_pos_1d)
                            max_pos = np.max(body_pos_1d)

                            psim.PlotLines(
                                label = body_name + " " + xyz_str[i],
                                values = body_pos_1d.tolist(),
                                values_offset = offset,
                                scale_min = min_pos,
                                scale_max = max_pos
                            )

                        psim.TreePop()
                psim.TreePop()

            if psim.TreeNode("Joint DOFS"):
                joint_rot_np = joint_rot.cpu().numpy()

                for b in range(mlib._kin_char_model.get_num_joints()):
                    body_name = char_model.get_body_name(b)

                    if psim.TreeNode(body_name):
                        xyzw_str = ["x", "y", "z", "w"]
                        for i in range(4):
                            joint_rot_1d = joint_rot_np[:, b, i]
                            psim.PlotLines(
                                label = body_name + " " + xyzw_str[i],
                                values = joint_rot_1d.tolist(),
                                values_offset = offset,
                                scale_min = -1.0,
                                scale_max = 1.0
                            )

                        psim.TreePop()
                psim.TreePop()
            psim.TreePop()
        psim.TreePop()
    return