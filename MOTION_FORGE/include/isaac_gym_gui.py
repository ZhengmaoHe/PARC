import polyscope as ps
import polyscope.imgui as psim
import tools.procgen.mdm_path as mdm_path
import MOTION_FORGE.include.global_header as g


def isaac_gym_gui():

    if not g.IsaacGymManager().is_ready():
        if psim.Button("Start Isaac Gym"):
            # env_file = "../tests/parkour/parkour_dataset_v_22_exp001/dm_env_TEASER_TERRAIN_mxu.yaml"
            # env_file = "../tests/parkour/parkour_dataset_v_22_exp001/dm_env_long_terrain_x_axis.yaml"
            # env_file = "../Data/shortcut_model_demo/simple_parc/dm_env_simple_parc.yaml"

            env_file = "data/terrains/dm_env_civilization.yaml"
            g.IsaacGymManager().start_isaac_gym(env_file, visualize=False)

            # model_path = "../tests/parkour/parkour_dataset_v_22_exp001/output/model.pt"
            # model_path = "../tests/parkour/parkour_dataset_v_20_exp001/output2/checkpoints/model_0000098800.pt"
            # agent_file = "../tests/parkour/parkour_dataset_v_22_exp001/ppo_agent.yaml"

            model_path = "../tests/parc/april272025/iter_3/p3_tracker/model.pt"
            agent_file = "../tests/parc/april272025/iter_3/p3_tracker/agent_config.yaml"
            g.IsaacGymManager().load_agent(agent_file, model_path)

            g.IsaacGymManager().create_MOTION_FORGE_character()

    if g.IsaacGymManager().is_ready():
        if psim.Button("Step"):
            g.IsaacGymManager().step()

        if psim.Button("Reset"):
            g.IsaacGymManager().reset()

        changed, g.IsaacGymManager().start_time_fraction = psim.SliderFloat("Start time fraction", g.IsaacGymManager().start_time_fraction, v_min=0.0, v_max=1.0)

        if psim.Button("Reset to time"):
            g.IsaacGymManager().reset_to_time(g.IsaacGymManager().start_time_fraction)
    
        if psim.Button("Reset to frame 0"):
            g.IsaacGymManager().reset_to_frame_0()

        if psim.Button("Transfer current motion to IG parkour env"):

            mlib = g.MotionManager().get_curr_motion().mlib
            mlib = mlib.clone(g.IsaacGymManager().env._device)

            g.IsaacGymManager().env.get_dm_env()._motion_lib = mlib

        if psim.Button("Compute value at current frame"):
            g.IsaacGymManager().compute_critic_value()

        if psim.TreeNode("Physics-based motion GUI"):
            ## Input:
            ## - Current state history of the simulated character
            ## - Target location
            
            ## Step 1: Path planner finds a path
            ## Step 2: 

            changed, g.IsaacGymManager().is_recording = psim.Checkbox("Record IG char states", g.IsaacGymManager().is_recording)

            changed, g.IsaacGymManager().replan_time = psim.InputFloat("Replan time", g.IsaacGymManager().replan_time)
            changed, g.IsaacGymManager().num_replan_loops = psim.InputInt("Num replan loops", g.IsaacGymManager().num_replan_loops)
            changed, g.MDMSettings().gen_settings.ddim_stride = psim.InputInt("DDIM Stride", g.MDMSettings().gen_settings.ddim_stride)

            if psim.Button("Make motion from recorded frames"):
                g.IsaacGymManager().create_motion_from_recorded_frames()

            if g.IsaacGymManager().is_recording:
                if len(g.IsaacGymManager().recorded_frames) > 2 and psim.Button("Motion Gen with current state"):

                    prev_frames = g.IsaacGymManager().recorded_frames[-2:]
                    prev_frames = g.motion_util.cat_motion_frames(prev_frames)

                    path_nodes = g.PathPlanningSettings().path_nodes.to(device=g.IsaacGymManager().device)

                    mdm_path_settings = g.PathPlanningSettings().mdm_path_settings
                    mdm_gen_settings = g.MDMSettings().gen_settings

                    gen_motion_frames, done = mdm_path.generate_frames_along_path(prev_frames=prev_frames,
                                                             path_nodes_xyz=path_nodes,
                                                             terrain=g.IsaacGymManager().env.get_dm_env()._terrain,
                                                             char_model=g.IsaacGymManager().env._kin_char_model,
                                                             mdm_model=g.g_mdm_model,
                                                             mdm_settings=mdm_gen_settings,
                                                             path_settings=mdm_path_settings,
                                                             verbose=False)
                    
                    mlib_motion_frames, mlib_contact_frames = gen_motion_frames.get_mlib_format(g.IsaacGymManager().env._kin_char_model)

                    mlib_motion_frames = mlib_motion_frames.to(device=g.MainVars().device).squeeze(0)
                    mlib_contact_frames = mlib_contact_frames.to(device=g.MainVars().device).squeeze(0)
                    print(mlib_motion_frames.shape, mlib_contact_frames.shape)
                    g.MotionManager().make_new_motion(mlib_motion_frames, mlib_contact_frames, "closed_loop_gen_motion",
                                                      motion_fps=30, vis_fps=5, new_color = [0.2, 0.8, 0.2])
                    
                changed, g.IsaacGymManager().is_closed_loop_generating = psim.Checkbox("Is closed loop generating", g.IsaacGymManager().is_closed_loop_generating)

            psim.TreePop()

        changed, g.IsaacGymManager().paused = psim.Checkbox("Paused", g.IsaacGymManager().paused)

        if not g.IsaacGymManager().paused:
            g.IsaacGymManager().loop_function()

            
        if psim.TreeNode("Extra Info"):
            motion_time = g.IsaacGymManager().env.get_dm_env()._get_motion_times(0)
            psim.TextUnformatted("Motion time: " + str(motion_time.item()) + " s")

            critic_val = g.IsaacGymManager().compute_critic_value()
            psim.TextUnformatted("Critic val: " + str(critic_val))
            psim.TreePop()
            

    return