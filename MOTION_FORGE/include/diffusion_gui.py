import numpy as np
import torch
import polyscope as ps
import polyscope.imgui as psim
import copy

import matplotlib.pyplot as plt
import multiprocessing

import util.torch_util as torch_util
from diffusion.diffusion_util import MDMFrameType
import diffusion.utils.rot_changer as rot_changer
import diffusion.mdm as mdm
import anim.motion_lib as motion_lib
import MOTION_FORGE.polyscope_util as ps_util
import util.terrain_util as terrain_util
import zmotion_editing_tools.motion_edit_lib as medit_lib
import diffusion.gen_util as gen_util
import util.motion_util as motion_util

import MOTION_FORGE.include.global_header as g

########## DIFFUSION GUI ##########
def diffusion_gui():
    main_vars = g.MainVars()
    settings = g.MDMSettings()
    curr_motion = g.MotionManager().get_curr_motion()
    gen_settings = g.MDMSettings().gen_settings

    if settings.prev_sampled_motion_ID is not None:
        changed, settings.resample_prev_motion = psim.Checkbox("resample prev motion", settings.resample_prev_motion)

    if g.g_sampler is not None and (psim.Button("Sample motion")):
        if not settings.resample_prev_motion:
            motion_ids = g.g_sampler._mlib.sample_motions(n=1)
            motion_start_times = g.g_sampler._sample_motion_start_times(motion_ids, g.g_sampler._sample_seq_time)
            settings.prev_sampled_motion_ID = motion_ids
            settings.prev_sampled_motion_start_time = motion_start_times
        
        else:
            motion_ids = settings.prev_sampled_motion_ID
            motion_start_times = settings.prev_sampled_motion_start_time
        
        

        if settings.sample_prev_states_only:
            motion_frames, hfs = g.g_sampler.sample_mismatched_prev_states_and_hfs(1)
        else:
            rot_changer_inst = rot_changer.RotChanger(rot_changer.RotationType.DEFAULT, 
                                                      g.g_sampler._kin_char_model)

            motion_data, hfs, target_info = g.g_sampler.sample_motion_data(
                motion_ids=motion_ids,
                motion_start_times=motion_start_times
            )

            root_pos = motion_data[MDMFrameType.ROOT_POS]
            root_rot = motion_data[MDMFrameType.ROOT_ROT]
            root_rot = rot_changer_inst.convert_quat_to_rot_type(root_rot)

            future_pos = root_pos[0, g.g_sampler._canon_idx].clone()
            future_pos += target_info.future_pos.squeeze(0)
            g.update_selected_pos_flag_mesh(future_pos)
            # TODO
            joint_rot = motion_data[MDMFrameType.JOINT_ROT]
            joint_dof = rot_changer_inst.convert_joint_quats_to_rot_type(joint_rot)
            motion_frames = torch.cat([root_pos, root_rot, joint_dof], dim=-1)
            contacts = motion_data[MDMFrameType.CONTACTS]

        #hfs *= g.g_sampler._max_h
        #motion_frames[..., 0:34] *= g.g_sampler._dof_high

        curr_motion.mlib = motion_lib.MotionLib(
            motion_frames[..., 0:34],
            g.g_sampler._kin_char_model, 
            device=main_vars.device, 
            init_type="motion_frames",
            loop_mode=motion_lib.LoopMode.CLAMP,
            fps=30,
            contact_info=main_vars.use_contact_info,
            contacts=contacts)
        
        curr_motion.update_sequence(0.0, curr_motion.mlib._motion_lengths[0].item(), curr_motion.mlib._motion_num_frames[0].item())

        #g_terrain.hf = hfs.squeeze(0)
        dx = g.g_sampler._dx
        num_x_neg = g.g_sampler._num_x_neg
        num_x_pos = g.g_sampler._num_x_pos
        num_y_neg = g.g_sampler._num_y_neg
        num_y_pos = g.g_sampler._num_y_pos
        grid_dim_x = g.g_sampler._grid_dim_x
        grid_dim_y = g.g_sampler._grid_dim_y
        
        g.g_terrain = terrain_util.SubTerrain(x_dim=grid_dim_x, y_dim=grid_dim_y, dx=dx, dy=dx,
                                            min_x=-dx * num_x_neg, min_y = -dx * num_y_neg, device=main_vars.device)
        g.g_terrain.hf = hfs.squeeze(0)

        g.TerrainMeshManager().rebuild()

        main_vars.motion_time = 0.0
        curr_motion.char.set_to_time(main_vars.motion_time, main_vars.motion_dt, curr_motion.mlib)
        curr_motion.update_transforms(shadow_height=curr_motion.char.get_hf_below_root(g.g_terrain))
        curr_motion.char.update_local_hf(g.g_terrain)

        # visualize floor heights to make sure its working
        floor_heights = motion_frames[..., -1]
        floor_height_pts = torch.cat([motion_frames[..., 0:2], floor_heights.unsqueeze(-1)], dim=-1).squeeze(0)
        floor_height_pts = floor_height_pts.cpu().numpy()
        ps_floor_height_pts = ps.register_point_cloud("floor heights", floor_height_pts, radius=0.0014)
        ps_floor_height_pts.set_color([0.0, 1.0, 1.0])

        main_vars.paused = True

    if g.g_sampler is not None and (psim.Button("Generate MDM OOD terrain for motion")):
        center_h = g.g_terrain.hf[g.g_sampler._num_x_neg, g.g_sampler._num_y_neg]
        hf = g.g_sampler.generate_hfs(1, center_h).squeeze(0)

        min_box_center_x = -g.g_sampler._dx * g.g_sampler._num_x_neg
        min_box_center_y = -g.g_sampler._dx * g.g_sampler._num_y_neg

        g.g_terrain = terrain_util.SubTerrain(x_dim = g.g_sampler._grid_dim_x, y_dim = g.g_sampler._grid_dim_y,
                                              dx = g.g_sampler._dx, dy = g.g_sampler._dx,
                                              min_x = min_box_center_x, min_y = min_box_center_y, device=g.MainVars().device)
        g.g_terrain.hf = hf * g.g_sampler._max_h
        g.TerrainMeshManager().rebuild()

    if g.g_mdm_model is None:
        psim.TextUnformatted("MDM model is not loaded")
        return
    
    # if psim.TreeNode("Loaded MDMS"):
    #     for key in settings.loaded_mdm_models:

    #         psim.TextUnformatted(key)
    #     psim.TreePop()

    opened = psim.BeginCombo("Selected MDM", settings.current_mdm_key)
    if opened:
        for key in settings.loaded_mdm_models:
            _, selected = psim.Selectable(key, key==settings.current_mdm_key)
            if selected:
                settings.select_mdm(key)
        psim.EndCombo()

    changed, settings.gen_settings.starting_diffusion_timestep = psim.InputInt("starting diffusion timestep", settings.gen_settings.starting_diffusion_timestep)
    settings.gen_settings.starting_diffusion_timestep = min(max(0, gen_settings.starting_diffusion_timestep), g.g_mdm_model._diffusion_timesteps)
    changed, settings.attention_debug = psim.Checkbox("attention debug", settings.attention_debug)
    changed, gen_settings.use_ddim = psim.Checkbox("use ddim", gen_settings.use_ddim)
    if gen_settings.use_ddim:
        changed, gen_settings.ddim_stride = psim.InputInt("ddim stride", gen_settings.ddim_stride)

    if psim.TreeNode("Guidance Params"):
        changed, gen_settings.guidance_str = psim.InputFloat("Guidance strength", gen_settings.guidance_str)
        changed, gen_settings.target_guidance = psim.Checkbox("target guidance", gen_settings.target_guidance)
        changed, gen_settings.hf_collision_guidance = psim.Checkbox("HF collision guidance", gen_settings.hf_collision_guidance)

        # if settings.target_guidance or settings.hf_collision_guidance:
        #     changed, settings.strong_hf_guidance = psim.Checkbox("Strong hf guidance", settings.strong_hf_guidance)
        # else:
        #     settings.strong_hf_guidance = False

        changed, gen_settings.cfg_scale = psim.InputFloat("CFG scale", gen_settings.cfg_scale)
        changed, gen_settings.use_cfg = psim.Checkbox("Use CFG", gen_settings.use_cfg)


        changed, gen_settings.guide_speed = psim.Checkbox("Use max speed guidance", gen_settings.guide_speed)
        changed, gen_settings.guide_acc = psim.Checkbox("Use max acc guidance", gen_settings.guide_acc)
        changed, gen_settings.guide_jerk = psim.Checkbox("Use max jerk guidance", gen_settings.guide_jerk)
        changed, gen_settings.w_speed = psim.InputFloat("w_speed", gen_settings.w_speed, format = "%.6g")
        changed, gen_settings.w_acc = psim.InputFloat("w_acc", gen_settings.w_acc, format = "%.6g")
        changed, gen_settings.w_jerk = psim.InputFloat("w_jerk", gen_settings.w_jerk, format = "%.6g")
        changed, gen_settings.max_jerk = psim.InputFloat("max_jerk", gen_settings.max_jerk, format="%.6g")
        psim.TreePop()

    if psim.TreeNode("In-painting params"):
        # TODO: get this working when loading different mdm models of different sequence lengths
        changed, gen_settings.input_root_pos = psim.Checkbox("Inpaint root pos", gen_settings.input_root_pos)
        psim.TreePop()

    changed, gen_settings.prev_state_ind_key = psim.Checkbox("Condition on prev state(s)", gen_settings.prev_state_ind_key)
    changed, gen_settings.target_condition_key = psim.Checkbox("Condition on target", gen_settings.target_condition_key)
    changed, gen_settings.feature_vector_key = psim.Checkbox("Condition on feature vector", gen_settings.feature_vector_key)
    

    changed, settings.append_mdm_motion_to_prev_motion = psim.Checkbox("Append MDM motion to prev motion", settings.append_mdm_motion_to_prev_motion)



    changed, settings.mdm_batch_size = psim.InputInt("mdm batch size", settings.mdm_batch_size)
    settings.mdm_batch_size = max(settings.mdm_batch_size, 1)
    

    changed, settings.sample_prev_states_only = psim.Checkbox("sample prev states only", settings.sample_prev_states_only)

    
    changed, settings.hide_batch_motions = psim.Checkbox("hide motions after batch gen", settings.hide_batch_motions)

    if psim.Button("Regenerate trajectory with In-painted root positions"):

        mlib = curr_motion.mlib
        dt = 1.0 / g.g_mdm_model._sequence_fps
        seq_len = g.g_mdm_model._seq_len
        hist_len = g.g_mdm_model._num_prev_states
        num_new_frames_each_gen = seq_len - hist_len - 5
        curr_motion_time = 1.0 * dt

        g.MDMSettings().gen_settings.use_cfg = False
        g.MDMSettings().gen_settings.input_root_pos = True
        g.MDMSettings().gen_settings.use_ddim = False

        next_motion_time = curr_motion_time + num_new_frames_each_gen * dt

        prev_last_frames = curr_motion.char.compute_motion_frames(curr_motion_time, dt, hist_len, hist_len, mlib).unsqueeze(0)
        prev_last_frames.body_pos = None
        prev_last_frames.body_rot = None
        new_frames = [prev_last_frames]
        while next_motion_time < mlib.get_motion_length(0).item():
            print("Current motion time:", curr_motion_time)

            curr_frames = curr_motion.char.compute_motion_frames(curr_motion_time, dt, seq_len, hist_len, mlib).unsqueeze(0)

            curr_frames = motion_util.cat_motion_frames([prev_last_frames, curr_frames.get_slice(slice(hist_len, seq_len))])

            gen_motion_frames = gen_util.gen_mdm_motion(target_world_pos=curr_frames.root_pos[:, -1],
                                                        prev_frames=curr_frames,
                                                        terrain=g.g_terrain,
                                                        mdm_model=g.g_mdm_model,
                                                        char_model=mlib._kin_char_model,
                                                        mdm_settings=g.MDMSettings().gen_settings)
            gen_motion_frames.set_device(g.MainVars().device)
            curr_motion_time += dt * num_new_frames_each_gen
            next_motion_time = curr_motion_time + num_new_frames_each_gen * dt

            new_frames_slice = slice(hist_len, hist_len + num_new_frames_each_gen)
            new_frames.append(gen_motion_frames.get_slice(new_frames_slice))

            prev_last_frames = gen_motion_frames.get_slice(slice(seq_len-2, seq_len))

        new_frames = motion_util.cat_motion_frames(new_frames)

        mlib_motion_frames, mlib_contact_frames = new_frames.get_mlib_format(mlib._kin_char_model)

        g.MotionManager().make_new_motion(motion_frames=mlib_motion_frames,
                                            contact_frames=mlib_contact_frames,
                                            new_motion_name="MDM_regenerated_motion",
                                            motion_fps=g.g_mdm_model._sequence_fps,
                                            vis_fps=10)
        print("Done regenerating motion with MDM.")

    if psim.Button("Refine motion"):

        mlib = curr_motion.mlib
        dt = 1.0 / g.g_mdm_model._sequence_fps
        seq_len = g.g_mdm_model._seq_len
        hist_len = g.g_mdm_model._num_prev_states

        canon_frame = int(round(g.MainVars().motion_time / dt))
        canon_motion_time = canon_frame * dt
        og_frames = curr_motion.char.compute_motion_frames(canon_motion_time, dt, seq_len, hist_len, mlib).unsqueeze(0)

        g.MDMSettings().gen_settings.use_cfg = False
        g.MDMSettings().gen_settings.input_root_pos = True
        g.MDMSettings().gen_settings.use_ddim = False
        g.MDMSettings().gen_settings.starting_diffusion_timestep = 5

        gen_motion_frames = gen_util.gen_mdm_motion(target_world_pos=og_frames.root_pos[:, -1],
                                                    prev_frames=og_frames,
                                                    terrain=g.g_terrain,
                                                    mdm_model=g.g_mdm_model,
                                                    char_model=mlib._kin_char_model,
                                                    mdm_settings=g.MDMSettings().gen_settings)

        mlib_motion_frames, mlib_contact_frames = gen_motion_frames.get_mlib_format(mlib._kin_char_model)

        g.MotionManager().make_new_motion(motion_frames=mlib_motion_frames,
                                        contact_frames=mlib_contact_frames,
                                        new_motion_name="MDM_refined_motion",
                                        motion_fps=g.g_mdm_model._sequence_fps,
                                        vis_fps=10)
        print("Done refining motion")
        


    if psim.TreeNode("Test functions"):
        if psim.Button("attention test"):
            #for mod in g.g_mdm_model._denoise_model._transformer_encoder.layers:
            #    print(mod)
            first_attn_layer = g.g_mdm_model._denoise_model._transformer_encoder.layers[0].self_attn
            print(first_attn_layer)

        changed, settings.conv_layer_num = psim.InputInt("Conv layer", settings.conv_layer_num)

        if changed:
            num_conv_layers = 0
            for layer in g.g_mdm_model._denoise_model._obs_conv_net._net:
                if isinstance(layer, torch.nn.Conv2d):
                    num_conv_layers +=1

            settings.conv_layer_num = min(num_conv_layers - 1, settings.conv_layer_num)
            settings.conv_layer_num = max(0, settings.conv_layer_num)

        if psim.Button("Conv net filter vis"):
            #print(g.g_mdm_model._denoise_model._obs_conv_net)
            curr_layer_num = 0
            for layer in g.g_mdm_model._denoise_model._obs_conv_net._net:
                if isinstance(layer, torch.nn.Conv2d):
                    if curr_layer_num != settings.conv_layer_num:
                        curr_layer_num += 1
                        continue
                    

                    filters = layer.weight.data

                    # normalize filters
                    min_val = filters.min()
                    max_val = filters.max()
                    filters = (filters - min_val) / (max_val - min_val)


                    # Convert tensor to numpy for visualization
                    filters_np = filters.cpu().numpy()

                    # Number of filters
                    num_filters = filters_np.shape[0]

                    # Plot filters
                    n_cols = 4
                    n_rows = num_filters // n_cols

                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
                    for i in range(num_filters):
                        ax = axes[i // n_cols, i % n_cols]
                        # Assuming filters are in (num_filters, channels, height, width)
                        if filters_np.shape[1] == 3:  # RGB
                            ax.imshow(filters_np[i].transpose(1, 2, 0))
                        else:  # Grayscale
                            ax.imshow(filters_np[i, 0], cmap='gray')
                        ax.axis('off')

                    plt.title("layer " + str(curr_layer_num) + " filters")
                    plt.show()

                    curr_layer_num += 1


        if psim.Button("Conv net feature maps"):
            # Hook to capture feature maps
            def get_feature_maps(module, input, output):
                global feature_maps
                feature_maps = output

            # Register hook
            layer = g.g_mdm_model._denoise_model._obs_conv_net._net[settings.conv_layer_num*2] # *2 to skip the activation layers
            hook = layer.register_forward_hook(get_feature_maps)

            # Perform a forward pass to capture the feature maps
            with torch.no_grad():
                agent = curr_motion.char
                hf_z = agent.get_normalized_local_hf(g.g_terrain, g.g_mdm_model._max_h)
                g.g_mdm_model._denoise_model._obs_conv_net(hf_z.unsqueeze(0).unsqueeze(0).to(device=g.g_mdm_model._device))

            # Remove the hook
            hook.remove()

            # Convert feature maps to numpy for visualization
            feature_maps_np = feature_maps.squeeze(0).cpu().numpy()

            # Plot feature maps
            num_feature_maps = feature_maps_np.shape[0]
            n_cols = 4
            n_rows = num_feature_maps // n_cols


            def plot_in_new_process():
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
                for i in range(num_feature_maps):
                    ax = axes[i // n_cols, i % n_cols]
                    ax.imshow(feature_maps_np[i].transpose(1, 0), cmap='viridis', origin='lower')
                    ax.axis('off')

                plt.show()

            plot_process = multiprocessing.Process(target=plot_in_new_process)
            plot_process.start()

        if psim.Button("test mdm standardization"):
            temp = torch.randn_like(g.g_mdm_model._mdm_features_mean)

            new_temp = g.g_mdm_model.standardize_features(temp)


            new_new_temp = g.g_mdm_model.unstandardize_features(new_temp)

            print("max error:", torch.max(torch.abs(new_new_temp - temp)))
        
        psim.TreePop()
    return
