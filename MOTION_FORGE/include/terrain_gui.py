import torch
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import matplotlib.pyplot as plt

import util.geom_util as geom_util
import util.terrain_util as terrain_util
import zmotion_editing_tools.motion_edit_lib as medit_lib
import time

import MOTION_FORGE.include.global_header as g

########## TERRAIN EDITING GUI ##########
def terrain_editing_gui():
    main_vars = g.MainVars()
    settings = g.TerrainEditorSettings()
    curr_motion = g.MotionManager().get_curr_motion()

    if psim.TreeNode("Terrain Info"):

        shape_str = "Shape: " + np.array2string(g.g_terrain.dims.cpu().numpy())
        psim.TextUnformatted(shape_str)

        min_xy_str = "Min X,Y: " + np.array2string(g.g_terrain.min_point.cpu().numpy())
        psim.TextUnformatted(min_xy_str)

        selected_grid_ind_str = "Curr grid ind: " + np.array2string(main_vars.selected_grid_ind.cpu().numpy())
        psim.TextUnformatted(selected_grid_ind_str)

        curr_grid_ind_point = g.g_terrain.get_point(main_vars.selected_grid_ind)
        z = g.g_terrain.get_hf_val(main_vars.selected_grid_ind)
        curr_grid_ind_xyz = np.array([curr_grid_ind_point[0].item(), curr_grid_ind_point[1].item(), z])
        curr_grid_ind_point_str = "Curr grid point X,Y, Z: " + np.array2string(curr_grid_ind_xyz, precision=6)
        psim.TextUnformatted(curr_grid_ind_point_str)
        psim.TreePop()

    changed, main_vars.mouse_size = psim.InputInt("mouse size", main_vars.mouse_size)
    if changed:
        main_vars.mouse_size = min(main_vars.mouse_size, 10)
        main_vars.mouse_size = max(main_vars.mouse_size, 0)
        g.update_mouse_ball_ps_meshes(main_vars.mouse_size)
    changed, settings.height = psim.InputFloat("terrain height", settings.height)

    
    opened = psim.BeginCombo("terrain edit mode", settings.edit_modes[settings.curr_edit_mode])
    if opened:
        for i, item in enumerate(settings.edit_modes):
            is_selected = (i == settings.curr_edit_mode)
            if psim.Selectable(item, is_selected)[0]:
                settings.curr_edit_mode = i

            # Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
            if is_selected:
                psim.SetItemDefaultFocus()
        psim.EndCombo()

    changed, settings.viewing_terrain = psim.Checkbox("View terrain", settings.viewing_terrain)
    if changed:
        g.TerrainMeshManager().hf_ps_mesh.set_enabled(settings.viewing_terrain)
    changed, settings.viewing_mask = psim.Checkbox("View heightfield mask", settings.viewing_mask)
    if changed:
        g.TerrainMeshManager().hf_mask_mesh.set_enabled(settings.viewing_mask)
    changed, settings.viewing_max = psim.Checkbox("View max heightfield bounds", settings.viewing_max)
    if changed:
        g.TerrainMeshManager().hf_max_mesh.set_enabled(settings.viewing_max)
    changed, settings.viewing_min = psim.Checkbox("View min heightfield bounds", settings.viewing_min)
    if changed:
        g.TerrainMeshManager().hf_min_mesh.set_enabled(settings.viewing_min)
    

    changed, settings.mask_mode = psim.Checkbox("Place mask (TRUE) / Remove mask (FALSE)", settings.mask_mode)

    if psim.Button("Set all to height"):
        if settings.edit_modes[settings.curr_edit_mode] == "heightfield":
            g.g_terrain.hf[...] = settings.height
            g.TerrainMeshManager().soft_rebuild()
            curr_motion.char.update_local_hf(g.g_terrain)

        if settings.edit_modes[settings.curr_edit_mode] == "max":
            g.g_terrain.hf_maxmin[..., 0] = settings.height
            g.TerrainMeshManager().build_ps_max_mesh()
        if settings.edit_modes[settings.curr_edit_mode] == "min":
            g.g_terrain.hf_maxmin[..., 1] = settings.height
            g.TerrainMeshManager().build_ps_min_mesh()

    if psim.Button("Compute extra vals for terrain"):
        char_point_samples = geom_util.get_char_point_samples(curr_motion.char.char_model)
        hf_mask_inds = terrain_util.compute_hf_extra_vals(
            curr_motion.mlib._motion_frames,
            g.g_terrain,
            curr_motion.char.char_model,
            char_point_samples)
        
        curr_motion.mlib._hf_mask_inds = [hf_mask_inds]

        g.TerrainMeshManager().rebuild()

    if psim.Button("Pad terrain"):
        g.g_terrain.pad(1, 0.0)
        g.TerrainMeshManager().rebuild()

    if psim.Button("Pad with min height"):
        min_h = torch.min(g.g_terrain.hf).item()
        g.g_terrain.pad(1, min_h)
        g.TerrainMeshManager().rebuild()

    if psim.TreeNode("Slice terrain"):
        changed, settings.slice_min_i = psim.InputInt("Slice min i", settings.slice_min_i)
        changed, settings.slice_min_j = psim.InputInt("Slice min j", settings.slice_min_j)
        changed, settings.slice_max_i = psim.InputInt("Slice max i", settings.slice_max_i)
        changed, settings.slice_max_j = psim.InputInt("Slice max j", settings.slice_max_j)

        if psim.Button("Slice terrain"):
            terrain_util.slice_terrain(g.g_terrain, 
                                       settings.slice_min_i, settings.slice_min_j, 
                                       settings.slice_max_i, settings.slice_max_j)
            g.TerrainMeshManager().rebuild()
        psim.TreePop()

    if psim.Button("Slice terrain around motion (and localize motion)"):
        sliced_terrain, localized_motion = terrain_util.slice_terrain_around_motion(curr_motion.mlib._motion_frames, g.g_terrain)

        g.MotionManager().make_new_motion(motion_frames=localized_motion, 
                          contact_frames=curr_motion.mlib._frame_contacts, 
                          new_motion_name=curr_motion.name, 
                          motion_fps=curr_motion.mlib._motion_fps[0].item(), 
                          vis_fps=5)
        
        g.g_terrain = sliced_terrain
        g.TerrainMeshManager().rebuild()

    if psim.Button("Slice terrain around motion"):
        sliced_terrain = terrain_util.slice_terrain_around_motion(curr_motion.mlib._motion_frames, g.g_terrain, localize=False)
        
        g.g_terrain = sliced_terrain
        g.TerrainMeshManager().rebuild()

    if psim.TreeNode("Build new terrain"):
        changed, settings.new_terrain_dim_x = psim.InputInt("New terrain dim x", settings.new_terrain_dim_x)
        changed, settings.new_terrain_dim_y = psim.InputInt("New terrain dim y", settings.new_terrain_dim_y)
        if psim.Button("Build new terrain"):
            g.g_terrain = terrain_util.SubTerrain("terrain", 
                                                  x_dim = settings.new_terrain_dim_x,
                                                  y_dim = settings.new_terrain_dim_y,
                                                  dx = 0.4,
                                                  dy = 0.4,
                                                  min_x = 0.0,
                                                  min_y = 0.0,
                                                  device=main_vars.device)
            g.TerrainMeshManager().rebuild()
        psim.TreePop()
        
        
    if psim.Button("Downsample Terrain"):

        g.g_terrain = terrain_util.downsample_terrain(g.g_terrain)

        # TODO: don't hard code thjs
        g.g_terrain.hf_maxmin[..., 0] = 3.0
        g.g_terrain.hf_maxmin[..., 1] = -3.0
        g.TerrainMeshManager().rebuild()
        curr_motion.char.update_local_hf(g.g_terrain)
    
    changed, settings.terrain_padding = psim.InputFloat("terrain padding", settings.terrain_padding)
    if(psim.Button("Regenerate Terrain")):
        g.g_terrain, hf_mask_inds = medit_lib.create_terrain_for_motion(curr_motion.mlib._motion_frames, 
                                                                        curr_motion.char.char_model,
                                                                        char_points = curr_motion.char.get_body_point_samples(),
                                                                        padding=settings.terrain_padding,
                                                                        dx = g.g_terrain.dxdy[0].item())
        
        # TODO: don't hard code thjs
        g.g_terrain.hf_maxmin[..., 0] = 3.0
        g.g_terrain.hf_maxmin[..., 1] = -3.0
        g.g_terrain.convert_mask_to_maxmin()
        curr_motion.mlib._hf_mask_inds = [hf_mask_inds]
        g.TerrainMeshManager().rebuild()
        curr_motion.char.update_local_hf(g.g_terrain)

    if(psim.Button("Guess Terrain")):
        terrain_util.hf_from_motion_discrete_heights(curr_motion.mlib._motion_frames,
                                                        g.g_terrain,
                                                        curr_motion.char.char_model,
                                                        [0.0, 0.6])
        g.TerrainMeshManager().soft_rebuild()

        curr_motion.char.update_local_hf(g.g_terrain)

    if psim.TreeNode("Procedural Generation"):
        if psim.TreeNode("Generate Boxes Config"):
            changed, settings.num_boxes = psim.InputInt("num boxes", settings.num_boxes)
            changed, settings.box_max_len = psim.InputInt("box max len", settings.box_max_len)
            changed, settings.box_min_len = psim.InputInt("box min len", settings.box_min_len)
            changed, settings.max_box_h = psim.InputFloat("box max height", settings.max_box_h)
            changed, settings.min_box_h = psim.InputFloat("box min height", settings.min_box_h)
            changed, settings.max_box_angle = psim.InputFloat("max box angle", settings.max_box_angle)
            changed, settings.min_box_angle = psim.InputFloat("min box angle", settings.min_box_angle)
            changed, settings.use_maxmin = psim.Checkbox("Use maxmin", settings.use_maxmin)
            psim.TreePop()
        
        if psim.Button("Generate Boxes"):

            terrain_util.add_boxes_to_hf2(g.g_terrain.hf,
                                        box_max_height=settings.max_box_h,
                                        box_min_height=settings.min_box_h, 
                                        hf_maxmin= g.g_terrain.hf_maxmin if settings.use_maxmin else None, 
                                        num_boxes=settings.num_boxes, 
                                        box_max_len=settings.box_max_len, 
                                        box_min_len=settings.box_min_len,
                                        max_angle=settings.max_box_angle,
                                        min_angle=settings.min_box_angle)

            g.TerrainMeshManager().soft_rebuild()
            curr_motion.char.update_local_hf(g.g_terrain)

        if psim.TreeNode("Curvy Paths config"):
            changed, settings.num_terrain_paths = psim.InputInt("num terrain paths", settings.num_terrain_paths)
            changed, settings.path_max_height = psim.InputFloat("path max height", settings.path_max_height)
            changed, settings.path_min_height = psim.InputFloat("path min height", settings.path_min_height)
            changed, settings.floor_height = psim.InputFloat("floor height", settings.floor_height)
            psim.TreePop()
            
        if psim.Button("Generate Curvy Paths"):
            terrain_util.gen_paths_hf(g.g_terrain, num_paths = settings.num_terrain_paths, maxpool_size=settings.maxpool_size,
                                    floor_height=settings.floor_height,
                                    path_min_height=settings.path_min_height, path_max_height=settings.path_max_height)
            g.TerrainMeshManager().rebuild()

        if psim.TreeNode("Staircases config"):
            changed, settings.min_stair_start_height = psim.InputFloat("min_stair_start_height", settings.min_stair_start_height)
            changed, settings.max_stair_start_height = psim.InputFloat("max_stair_start_height", settings.max_stair_start_height)
            changed, settings.min_step_height = psim.InputFloat("min_step_height", settings.min_step_height)
            changed, settings.max_step_height = psim.InputFloat("max_step_height", settings.max_step_height)
            changed, settings.num_stairs = psim.InputInt("num_stairs", settings.num_stairs)
            changed, settings.min_stair_thickness = psim.InputFloat("min_stair_thickness", settings.min_stair_thickness)
            changed, settings.max_stair_thickness = psim.InputFloat("max_stair_thickness", settings.max_stair_thickness)
            psim.TreePop()

        if psim.Button("Generate Staircases"):
            g.g_terrain.hf[...] = 0.0
            terrain_util.add_stairs_to_hf(g.g_terrain,
                                          min_stair_start_height=settings.min_stair_start_height,
                                          max_stair_start_height=settings.max_stair_start_height,
                                          min_step_height=settings.min_step_height,
                                          max_step_height=settings.max_step_height,
                                          num_stairs=settings.num_stairs,
                                          min_stair_thickness=settings.min_stair_thickness,
                                          max_stair_thickness=settings.max_stair_thickness)
            g.TerrainMeshManager().rebuild()

        if psim.Button("Gen Cave"):

            g.g_terrain = terrain_util.generate_cave(25, 25, 5, device=g.MainVars().device)

            g.TerrainMeshManager().rebuild()

        psim.TreePop()

    changed, settings.maxpool_size = psim.InputInt("maxpool size", settings.maxpool_size)
    if(psim.Button("Maxpool terrain")):
        terrain_util.maxpool_hf(g.g_terrain.hf, g.g_terrain.hf_maxmin, settings.maxpool_size)
        g.TerrainMeshManager().soft_rebuild()
        curr_motion.char.update_local_hf(g.g_terrain)

    if psim.Button("Maxpool terrain x"):
        terrain_util.maxpool_hf_1d_x(g.g_terrain.hf, g.g_terrain.hf_maxmin, settings.maxpool_size)
        g.TerrainMeshManager().soft_rebuild()
        curr_motion.char.update_local_hf(g.g_terrain)

    if psim.Button("Maxpool terrain y"):
        terrain_util.maxpool_hf_1d_y(g.g_terrain.hf, g.g_terrain.hf_maxmin, settings.maxpool_size)
        g.TerrainMeshManager().soft_rebuild()
        curr_motion.char.update_local_hf(g.g_terrain)

    if psim.Button("Minpool Terrain"):
        terrain_util.minpool_hf(g.g_terrain.hf, g.g_terrain.hf_maxmin, settings.maxpool_size)
        g.TerrainMeshManager().soft_rebuild()
        curr_motion.char.update_local_hf(g.g_terrain)

    if psim.Button("Detect sharp lines"):
        terrain = g.g_terrain

        def detect_sharp_line(i, j):

            center_h = terrain.hf[i, j]
            
            test1 = center_h > terrain.hf[i-1, j] and center_h > terrain.hf[i+1, j]
            test2 = center_h < terrain.hf[i-1, j] and center_h < terrain.hf[i+1, j]
            test3 = center_h > terrain.hf[i, j-1] and center_h > terrain.hf[i, j+1]
            test4 = center_h < terrain.hf[i, j-1] and center_h < terrain.hf[i, j+1]

            return test1 or test2 or test3 or test4

        sharp_line_points = []
        for i in range(1, terrain.hf.shape[0]-1):
            for j in range(1, terrain.hf.shape[1]-1):
                if detect_sharp_line(i, j):
                    xyz = terrain.get_xyz_point(torch.tensor([i, j], dtype=torch.int64)).numpy()
                    sharp_line_points.append(xyz)

        sharp_line_points = np.stack(sharp_line_points)

        ps.register_point_cloud("sharp line points", sharp_line_points, radius=0.01)
        #test = 0

    if psim.Button("Remove sharp lines"):
        terrain_util.remove_sharp_lines(g.g_terrain)
        g.TerrainMeshManager().rebuild()

    if psim.Button("Flat maxpool 2x2"):

        terrain_util.flat_maxpool_2x2(g.g_terrain)

        g.TerrainMeshManager().rebuild()

    if psim.Button("Flat maxpool 3x3"):

        terrain_util.flat_maxpool_3x3(g.g_terrain)

        g.TerrainMeshManager().rebuild()

    if psim.Button("Flatten 4x4 path start/end nodes"):

        start_node_th = torch.from_numpy(g.PathPlanningSettings().start_node)
        end_node_th = torch.from_numpy(g.PathPlanningSettings().end_node)
        terrain_util.flatten_4x4_near_edge(g.g_terrain, start_node_th,
                                           g.g_terrain.get_hf_val(start_node_th).item())
        terrain_util.flatten_4x4_near_edge(g.g_terrain, end_node_th,
                                           g.g_terrain.get_hf_val(end_node_th).item())
        g.TerrainMeshManager().rebuild()

    if psim.Button("Plot local hf"):
        hf_z = curr_motion.char.get_normalized_local_hf(g.g_terrain, 3.0)
        hf_z = hf_z.transpose(1, 0).rot90(k=3)
        #plt.imshow(hf_z.transpose(1, 0), cmap='viridis', origin='lower')
        plt.imshow(hf_z, cmap='viridis', origin='lower')
        plt.show()

    if psim.Button("Plot augmented local hf using sampler"):
        import yaml
        from diffusion.mdm_heightfield_contact_motion_sampler import MDMHeightfieldContactMotionSampler

        cfg_path = "diffusion/mdm.yaml"
        with open(cfg_path, "r") as stream:
            config = yaml.safe_load(stream)
            config["device"] = g.MainVars().device
        config["motion_lib_file"] = curr_motion.mlib
        sampler = MDMHeightfieldContactMotionSampler(config)

        motion_ids = torch.tensor([0], dtype=torch.int64, device=g.MainVars().device)
        motion_start_times = torch.tensor([0.0], dtype=torch.float32, device=g.MainVars().device)
        motion, hf, target = sampler.sample_motion_data(motion_ids=motion_ids,
                                   motion_start_times=motion_start_times,
                                   ret_hf_obs=True,
                                   ret_target_info=True)

        curr_motion.set_to_time(1.0 / curr_motion.mlib._motion_fps[0].item())

        hf_z = hf.squeeze(0) + curr_motion.char.get_body_pos(0)[2].item()
        curr_motion.char.update_local_hf(g.g_terrain, hf_z=hf_z)


        hf_z = hf_z.transpose(1, 0).rot90(k=3)
        plt.imshow(hf_z, cmap='viridis', origin='lower')
        plt.show()

    if psim.Button("Plot global hf"):
        hf_z = g.g_terrain.hf.numpy()
        plt.imshow(hf_z.transpose(1, 0), cmap='viridis', origin='lower')
        plt.xticks([])
        plt.yticks([])
        filepath = "output/terrain" + str(time.time()) + ".png"
        plt.savefig(fname=filepath, bbox_inches="tight")
        plt.show()

    if(psim.Button("Reload terrain from file")):
        motion_data = medit_lib.load_motion_file(g.g_motion_filepath)
        assert "terrain" in motion_data._data

        g.g_terrain = motion_data.get_terrain()
        g.TerrainMeshManager().rebuild()

        curr_motion.char.update_local_hf(g.g_terrain)
    return