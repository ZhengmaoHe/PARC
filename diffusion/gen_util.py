import torch
import util.torch_util as torch_util
import anim.kin_char_model as kin_char_model
import anim.motion_lib as motion_lib
import diffusion.mdm as mdm
import util.geom_util as geom_util
import util.terrain_util as terrain_util
from diffusion.diffusion_util import MDMFrameType
from diffusion.diffusion_util import MDMKeyType, MDMCustomGuidance, RelativeZStyle, MDMInPaint
from util.motion_util import MotionFrames

class MDMGenSettings:
    use_prev_state = True
    use_cfg = True
    cfg_scale = 0.65
    guide_speed = False
    guide_acc = False
    guide_jerk = False
    w_speed = 0.1
    w_acc = 0.1
    w_jerk = 0.1
    max_jerk = 1000.0
    strong_hf_guidance = False
    target_guidance = False
    hf_collision_guidance = False
    prev_state_ind_key = True
    target_condition_key = True
    feature_vector_key = True
    use_ddim = True
    ddim_stride = 10
    guidance_str = 0.1
    input_root_pos = False
    starting_diffusion_timestep = 0


def gen_mdm_motion(target_world_pos: torch.Tensor,
                   prev_frames: MotionFrames,
                   terrain: terrain_util.SubTerrain,
                   mdm_model: mdm.MDM,
                   char_model: kin_char_model.KinCharModel,
                   mdm_settings: MDMGenSettings,
                   #batch_size: int = 1,
                   verbose: bool = True) -> MotionFrames:

    if mdm_model is None:
        print("MDM not loaded")
        assert False

    canon_idx = mdm_model._num_prev_states - 1

    #main_vars = g.MainVars()
    #mdm_settings = g.MDMSettings()
    conds = dict()
    batch_size = prev_frames.root_pos.shape[0]
    input_device = prev_frames.root_pos.device
    assert terrain.hf.device == input_device
    assert target_world_pos.device == input_device
    assert char_model._device == input_device
    mdm_device = mdm_model._device

    #agent = g.g_motion.char
    #prev_frames = agent.motion_frames
    #canon_idx = agent._history_length - 1

    # using_floor_heights = MDMFrameType.FLOOR_HEIGHTS in g.g_mdm_model._ftype_slices
    # if using_floor_heights:
    #     # append floor height to prev states
    #     global_prev_xy = prev_states[..., 0:2]
    #     grid_inds = terrain.get_grid_index(global_prev_xy.to(device=main_vars.device))
    #     prev_floor_heights = terrain.hf[grid_inds[..., 0], grid_inds[..., 1]].to(device=g.g_mdm_model._device)
    #     z_below_root = agent.get_hf_below_root(terrain)
    #     if verbose:
    #         print("z_below_root:", z_below_root)
    #     prev_floor_heights = prev_floor_heights - z_below_root
        
    #     # assuming history length >= 2
    #     prev_states = torch.cat([prev_states, prev_floor_heights.unsqueeze(-1)], dim=-1)

    #uncanon_prev_states = prev_states.clone()
    ## CANONICALIZE
    canon_slice = slice(canon_idx, canon_idx+1)
    canon_xy = prev_frames.root_pos[:, canon_slice, 0:2]

    canon_heading_quat = torch_util.calc_heading_quat(prev_frames.root_rot[:, canon_slice])
    canon_heading_quat_inv = torch_util.calc_heading_quat_inv(prev_frames.root_rot[:, canon_slice])
    canon_heading = torch_util.calc_heading(prev_frames.root_rot[:, canon_slice])

    root_pos = prev_frames.root_pos.clone()
    root_pos[..., 0:2] -= canon_xy
    root_pos = torch_util.quat_rotate(canon_heading_quat_inv, root_pos)
    root_rot = torch_util.quat_multiply(canon_heading_quat_inv, prev_frames.root_rot)


    hf_z = terrain_util.sample_hf_z_on_terrain(terrain=terrain,
                                               center_xy=canon_xy,
                                               heading=canon_heading,
                                               dx=mdm_model._dx,
                                               dy=mdm_model._dy,
                                               num_x_neg=mdm_model._num_x_neg,
                                               num_x_pos=mdm_model._num_x_pos,
                                               num_y_neg=mdm_model._num_y_neg,
                                               num_y_pos=mdm_model._num_y_pos)

    if mdm_model._relative_z_style == RelativeZStyle.RELATIVE_TO_ROOT_FLOOR: # RELATIVE TO ROOT FLOOR
        ind = terrain.get_grid_index(canon_xy)
        z_below_root = terrain.get_hf_val(ind)#hf[ind[0], ind[1]].item()
        hf_z = hf_z - z_below_root
        
        # need to also make sure the "prev state" has it's height
        # relatively corrected
        root_pos[..., 2] -= z_below_root.unsqueeze(-1)
    elif mdm_model._relative_z_style == RelativeZStyle.RELATIVE_TO_ROOT: # RELATIVE TO ROOT
        canon_z = prev_frames.root_pos[:, canon_slice, 2]
        root_pos[..., 2] -= canon_z

        hf_z = hf_z - canon_z.unsqueeze(-1)
    else:
        assert False
    
    # get canonicalized body pos from canoicalized root pos, root_rot
    body_pos, body_rot = char_model.forward_kinematics(root_pos, root_rot, prev_frames.joint_rot)

    ## GET CANONICALIZED TARGET ##
    target = target_world_pos[:, 0:2].to(dtype=torch.float32).unsqueeze(1) - canon_xy
    target = torch_util.rotate_2d_vec(target, -canon_heading)

    assert mdm_model._target_type == mdm.TargetType.XY_DIR
    # target_len = torch.norm(target, dim=-1)
    # target_dir = target.clone()
    # if target_len < 0.05:
    #     target_dir[...] = 0.0
    # else:
    #     target_dir = target_dir / target_len
    target_dir = torch.nn.functional.normalize(target, dim=-1)
    
    conds[MDMKeyType.OBS_KEY] = hf_z.to(device=mdm_device)
    conds[MDMKeyType.OBS_FLAG_KEY] = torch.tensor([True], dtype=torch.bool, device=mdm_device).expand(batch_size)

    mdm_prev_state_input = {
        MDMFrameType.ROOT_POS: root_pos[:, 0:mdm_model._num_prev_states].to(device=mdm_device),
        MDMFrameType.ROOT_ROT: root_rot[:, 0:mdm_model._num_prev_states].to(device=mdm_device),
        MDMFrameType.JOINT_POS: body_pos[:, 0:mdm_model._num_prev_states, 1:, :].to(device=mdm_device),
        MDMFrameType.JOINT_ROT: prev_frames.joint_rot[:, 0:mdm_model._num_prev_states].to(device=mdm_device),
        MDMFrameType.CONTACTS: prev_frames.contacts[:, 0:mdm_model._num_prev_states].to(device=mdm_device)
    }
    # TODO: ^ the above gets put in depending on g.g_mdm_model._frame_components
    if mdm_settings.input_root_pos:
        mdm_future_state_input = {
            MDMFrameType.ROOT_POS: root_pos.to(device=mdm_device),
            MDMFrameType.ROOT_ROT: root_rot.to(device=mdm_device),
            MDMFrameType.JOINT_POS: body_pos[..., 1:, :].to(device=mdm_device),
            MDMFrameType.JOINT_ROT: prev_frames.joint_rot.to(device=mdm_device),
            MDMFrameType.CONTACTS: prev_frames.contacts.to(device=mdm_device)
        }

        mdm_in_paint_params = MDMInPaint()
        mdm_in_paint_params.frame_idxs = torch.tensor([-2, -1], dtype=torch.int64).unsqueeze(0).expand(batch_size, -1)
        mdm_in_paint_params.frames = mdm_future_state_input # TODO: convert to mdm input representation
        conds[MDMKeyType.IN_PAINT_PARAMS] = mdm_in_paint_params

    conds[MDMKeyType.PREV_STATE_KEY] = mdm_prev_state_input
    if isinstance(mdm_settings.use_prev_state, torch.Tensor):
        conds[MDMKeyType.PREV_STATE_NOISE_IND_KEY] = mdm_settings.use_prev_state
    else:
        conds[MDMKeyType.PREV_STATE_NOISE_IND_KEY] = torch.tensor([mdm_settings.use_prev_state], dtype=torch.bool, device=mdm_device).expand(batch_size)
    conds[MDMKeyType.TARGET_KEY] = target_dir.to(device=mdm_device)
    conds[MDMKeyType.TARGET_FLAG_KEY] = torch.tensor([mdm_settings.target_condition_key], dtype=torch.bool, device=mdm_device).expand(batch_size)
    if isinstance(mdm_settings.prev_state_ind_key, torch.Tensor):
        conds[MDMKeyType.PREV_STATE_FLAG_KEY] = mdm_settings.prev_state_ind_key
    else:
        conds[MDMKeyType.PREV_STATE_FLAG_KEY] = torch.tensor([mdm_settings.prev_state_ind_key], dtype=torch.bool, device=mdm_device).expand(batch_size)
    
    guidance_params = MDMCustomGuidance()
    if mdm_settings.use_cfg:
        guidance_params.obs_cfg_scale = mdm_settings.cfg_scale

    if mdm_settings.target_guidance:
        guidance_params.guidance_str = mdm_settings.guidance_str
        guidance_params.target_xy = target
        #guidance_params.target_floor_height =

    if mdm_settings.hf_collision_guidance:
        guidance_params.guidance_str = mdm_settings.guidance_str
        guidance_params.body_points = geom_util.get_char_point_samples(char_model)

    if mdm_settings.use_cfg or \
        mdm_settings.target_guidance or \
        mdm_settings.hf_collision_guidance or \
        mdm_settings.strong_hf_guidance:
        guidance_params.strong_hf_guidance = mdm_settings.strong_hf_guidance
        conds[MDMKeyType.GUIDANCE_PARAMS] = guidance_params

    guidance_params.guide_speed = mdm_settings.guide_speed
    guidance_params.guide_acc = mdm_settings.guide_acc
    guidance_params.guide_jerk = mdm_settings.guide_jerk
    guidance_params.w_speed = mdm_settings.w_speed
    guidance_params.w_acc = mdm_settings.w_acc
    guidance_params.w_jerk = mdm_settings.w_jerk
    guidance_params.max_jerk = mdm_settings.max_jerk
    mdm_mode = mdm.GenerationMode.MODE_DDIM if mdm_settings.use_ddim else mdm.GenerationMode.MODE_REVERSE_DIFFUSION
    
    mdm_ret = mdm_model.gen_sequence_with_contacts(conds, mdm_settings.ddim_stride, mode=mdm_mode)


    new_root_pos = mdm_ret[MDMFrameType.ROOT_POS].to(device=input_device)
    new_root_rot = mdm_ret[MDMFrameType.ROOT_ROT].to(device=input_device)
    new_joint_rot = mdm_ret[MDMFrameType.JOINT_ROT].to(device=input_device)
    new_contacts = mdm_ret[MDMFrameType.CONTACTS].to(device=input_device)

    ## UNCANONICALIZE ##
    new_root_rot = torch_util.quat_multiply(canon_heading_quat, new_root_rot)
    new_root_pos = torch_util.quat_rotate(canon_heading_quat, new_root_pos)
    new_root_pos[..., 0:2] = new_root_pos[..., 0:2] + canon_xy

    if mdm_model._relative_z_style == RelativeZStyle.RELATIVE_TO_ROOT_FLOOR:
        new_root_pos[..., 2] += z_below_root
    elif mdm_model._relative_z_style == RelativeZStyle.RELATIVE_TO_ROOT: # RELATIVE TO ROOT
        new_root_pos[..., 2] += canon_z

    ret_motion_frames = MotionFrames(root_pos = new_root_pos,
                                     root_rot = new_root_rot,
                                     joint_rot = new_joint_rot,
                                     body_pos = None,
                                     body_rot = None,
                                     contacts = new_contacts)
    
    return ret_motion_frames