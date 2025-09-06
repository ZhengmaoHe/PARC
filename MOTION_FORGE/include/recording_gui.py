import os
import torch

import shutil
import subprocess
import send2trash
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import anim.motion_lib as motion_lib
import zmotion_editing_tools.motion_edit_lib as medit_lib
import MOTION_FORGE.polyscope_util as ps_util
import MOTION_FORGE.include.global_header as g
import MOTION_FORGE.include.ig_obs_gui as ig_obs_gui

def cleanup_for_images_and_videos():
    main_vars = g.MainVars()
    curr_motion = g.MotionManager().get_curr_motion()
    curr_motion.sequence.mesh.set_enabled(False)
    curr_motion.char.set_prev_state_enabled(False)
    g.OptimizationSettings().clear_body_constraints()
    
    main_vars.viewing_local_hf = False
    curr_motion.char.set_local_hf_enabled(main_vars.viewing_local_hf)

    g.g_dir_mesh.set_enabled(False)

    main_vars.mouse_ball_visible = False
    for mesh in g.g_mouse_ball_meshes:
        mesh.set_enabled(main_vars.mouse_ball_visible)

    curr_motion.char.set_body_points_enabled(False)

    ps.set_background_color([1.0, 1.0, 1.0])
    ps.get_surface_mesh("origin axes").set_enabled(False)
    ps.get_surface_mesh("selected pos flag").set_enabled(False)

    print("View json:")
    print(ps.get_view_as_json())
    return

def record_video(name=None):
    main_vars = g.MainVars()
    curr_motion = g.MotionManager().get_curr_motion()
    fps = int(curr_motion.mlib._motion_fps[0].item())
    g.g_dir_mesh.set_enabled(False)
    main_vars.mouse_ball_visible = False
    for mesh in g.g_mouse_ball_meshes:
        mesh.set_enabled(False)

    curr_motion.sequence.mesh.set_enabled(False)
    curr_motion.char.set_prev_state_enabled(False)

    image_folder = "output/ps_output/"
    video_folder = "output/videos/"
    if os.path.exists(image_folder):
        shutil.rmtree(image_folder)
    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(video_folder, exist_ok=True)

    video_fps = max(60, fps)#120

    num_frames = curr_motion.mlib._motion_num_frames[0].item() * (video_fps // fps)
    for curr_frame_idx in range(0, num_frames):
        main_vars.motion_time = curr_frame_idx * 1.0 / video_fps
        print("screenshotting at time:", main_vars.motion_time)
        curr_motion.char.set_to_time(main_vars.motion_time, main_vars.motion_dt, curr_motion.mlib)
        curr_motion.update_transforms(shadow_height=curr_motion.char.get_hf_below_root(g.g_terrain))
        curr_motion.char.update_local_hf(g.g_terrain)

        if g.IGObsSettings().has_obs and g.IGObsSettings().record_obs:
            ig_obs_gui.animate_obs(main_vars.motion_time, g.IGObsSettings().overlay_obs_on_motion)

        ss_filepath = image_folder + "frame_" + "{:06d}".format(curr_frame_idx) + ".png"
        ps.screenshot(ss_filepath)

    
    if name is None:
        video_filepath = video_folder + os.path.splitext(curr_motion.name)[0] + ".mp4"
    else:
        video_filepath = video_folder + name + ".mp4"
    print("writing video to:", video_filepath)

    if os.path.exists(video_filepath):
        send2trash.send2trash(video_filepath)
    # remove '*conda*' entries
    path_cur = os.environ['PATH']
    path_new = ':'.join(p for p in path_cur.split(':') if 'conda' not in p)
    command = "ffmpeg -framerate " + str(video_fps) + " -i " + image_folder + "frame_%06d.png "
    #command += "-vcodec mpeg4 -qscale 0 "# -vprofile main -preset slow -b:v 400k -maxrate 400k -bufsize 800k "
    #command += "-vf scale=-1:480 "# -threads 0 -acodec libvo_aacenc -ab 128k " + video_filepath
    #command += "-c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p "
    command += "-vcodec mpeg4 -b 9000k "
    command += video_filepath
    command = command.split()

    # Run the ffmpeg command
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env={'PATH': path_new})

    # Check if there were any errors
    if result.returncode == 0:
        print('Video creation successful!')
        # delete the .pngs
        
    else:
        print('Error creating video:')
        print(result.stderr.decode('utf-8'))

    main_vars.paused = True
    return


def recording_gui():
    main_vars = g.MainVars()
    curr_motion = g.MotionManager().get_curr_motion()
    fps = int(curr_motion.mlib._motion_fps[0].item())

    if psim.Button("Clean up for images/videos"):
        cleanup_for_images_and_videos()

    changed, g.RecordingSettings().view_json = psim.InputText("view json", g.RecordingSettings().view_json)

    if psim.Button("Setup from view json"):
        
        ps.set_view_from_json(g.RecordingSettings().view_json)

    if psim.Button("Setup from view json 2"):
        ps.set_view_from_json(g.RecordingSettings().view_json2)

    if psim.Button("Print current camera view json"):
        print(ps.get_view_as_json())

    if psim.Button("Record Video (need ffmpeg)"):
        record_video()

    if psim.Button("Record Rotating Video"):
        # Play the motion 3 times, while completing one full rotation
        # rotate camera around terrain
        radius = 10.0
        target = g.g_terrain.get_xyz_point(grid_inds = g.g_terrain.dims // 2).numpy()

        g.g_dir_mesh.set_enabled(False)
        main_vars.mouse_ball_visible = False
        for mesh in g.g_mouse_ball_meshes:
            mesh.set_enabled(False)

        curr_motion.sequence.mesh.set_enabled(False)
        curr_motion.char.set_prev_state_enabled(False)

        image_folder = "output/ps_output/"
        video_folder = "output/"
        if os.path.exists(image_folder):
            shutil.rmtree(image_folder)
        os.makedirs(image_folder, exist_ok=True)

        video_fps = max(60, fps)#120

        num_frames = curr_motion.mlib._motion_num_frames[0].item() * (video_fps // fps)


        for i in range(3):
            for curr_frame_idx in range(0, num_frames):

                rot_time = (curr_frame_idx + i * num_frames) / (3.0 * num_frames) * 2.0 * np.pi
                camera_location_x = target[0] + np.cos(rot_time) * radius
                camera_location_y = target[1] + np.sin(rot_time) * radius
                camera_location_z = target[2] + 4.0
                camera_location = np.array([camera_location_x, camera_location_y, camera_location_z])
                ps.look_at(camera_location=camera_location,target=target)

                main_vars.motion_time = curr_frame_idx * 1.0 / video_fps
                print("screenshotting (loop: "+ str(i)+") at time:", main_vars.motion_time)
                curr_motion.char.set_to_time(main_vars.motion_time, main_vars.motion_dt, curr_motion.mlib)
                curr_motion.update_transforms(shadow_height=curr_motion.char.get_hf_below_root(g.g_terrain))
                curr_motion.char.update_local_hf(g.g_terrain)

                ss_filepath = image_folder + "frame_" + "{:06d}".format(curr_frame_idx + i * num_frames) + ".png"
                ps.screenshot(ss_filepath)

        
        video_filepath = video_folder + os.path.splitext(curr_motion.name)[0] + ".mp4"
        print("writing video to:", video_filepath)

        if os.path.exists(video_filepath):
            send2trash.send2trash(video_filepath)
        # remove '*conda*' entries
        path_cur = os.environ['PATH']
        path_new = ':'.join(p for p in path_cur.split(':') if 'conda' not in p)
        command = "ffmpeg -framerate " + str(video_fps) + " -i " + image_folder + "frame_%06d.png "
        command += "-vcodec mpeg4 -qscale 0 "# -vprofile main -preset slow -b:v 400k -maxrate 400k -bufsize 800k "
        #command += "-vf scale=-1:480 "# -threads 0 -acodec libvo_aacenc -ab 128k " + video_filepath
        command += video_filepath
        command = command.split()

        # Run the ffmpeg command
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env={'PATH': path_new})

        # Check if there were any errors
        if result.returncode == 0:
            print('Video creation successful!')
            # delete the .pngs
            
        else:
            print('Error creating video:')
            print(result.stderr.decode('utf-8'))

        main_vars.paused = True



    changed, g.RecordingSettings().root_pos_spacing = psim.InputFloat("root_pos_spacing", g.RecordingSettings().root_pos_spacing)

    return