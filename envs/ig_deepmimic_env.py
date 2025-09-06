import numpy as np
import torch
import envs.base_env as base_env
import envs.ig_char_env as ig_char_env
import util.torch_util as torch_util

    
@torch.jit.script
def compute_phase_obs(phase, num_phase_encoding):
    # type: (Tensor, int) -> Tensor
    phase_obs = phase.unsqueeze(-1)

    # positional embedding of phase
    if (num_phase_encoding > 0):
        pe_exp = torch.arange(num_phase_encoding, device=phase.device, dtype=phase.dtype)
        pe_scale = 2.0 * np.pi * torch.pow(2.0, pe_exp)
        pe_scale = pe_scale.unsqueeze(0)
        pe_val = phase.unsqueeze(-1) * pe_scale
        pe_sin = torch.sin(pe_val)
        pe_cos = torch.cos(pe_val)

        phase_obs = torch.cat((phase_obs, pe_sin, pe_cos), dim=-1)

    return phase_obs

@torch.jit.script
def convert_to_local(root_rot, root_vel, root_ang_vel, key_pos):
    # type: (Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]

    heading_inv_rot = torch_util.calc_heading_quat_inv(root_rot)

    local_root_rot = torch_util.quat_mul(heading_inv_rot, root_rot)
    local_root_vel = torch_util.quat_rotate(heading_inv_rot, root_vel)
    local_root_ang_vel = torch_util.quat_rotate(heading_inv_rot, root_ang_vel)
    
    if (len(key_pos) > 0):
        heading_rot_expand = heading_inv_rot.unsqueeze(-2)
        heading_rot_expand = heading_rot_expand.repeat((1, key_pos.shape[1], 1))
        flat_heading_rot_expand = heading_rot_expand.reshape(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                                                heading_rot_expand.shape[2])
        flat_key_pos = key_pos.reshape(key_pos.shape[0] * key_pos.shape[1], key_pos.shape[2])
        flat_local_key_pos = torch_util.quat_rotate(flat_heading_rot_expand, flat_key_pos)
        local_key_pos = flat_local_key_pos.reshape(key_pos.shape[0], key_pos.shape[1], key_pos.shape[2])
    else:
        local_key_pos = key_pos

    return local_root_rot, local_root_vel, local_root_ang_vel, local_key_pos

@torch.jit.script
def compute_tar_obs(ref_root_pos, ref_root_rot, root_pos, root_rot, joint_rot, key_pos,
                    global_obs, root_height_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool) -> Tensor
    ref_root_pos = ref_root_pos.unsqueeze(-2)
    root_pos_obs = root_pos - ref_root_pos
    
    if (len(key_pos) > 0):
        key_pos = key_pos - root_pos.unsqueeze(-2)

    if (not global_obs):
        heading_inv_rot = torch_util.calc_heading_quat_inv(ref_root_rot)
        heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2)
        heading_inv_rot_expand = heading_inv_rot_expand.repeat((1, root_pos.shape[1], 1))
        heading_inv_rot_flat = heading_inv_rot_expand.reshape((heading_inv_rot_expand.shape[0] * heading_inv_rot_expand.shape[1], 
                                                               heading_inv_rot_expand.shape[2]))
        root_pos_obs_flat = torch.reshape(root_pos_obs, [root_pos_obs.shape[0] * root_pos_obs.shape[1], root_pos_obs.shape[2]])
        root_pos_obs_flat = torch_util.quat_rotate(heading_inv_rot_flat, root_pos_obs_flat)
        root_pos_obs = torch.reshape(root_pos_obs_flat, root_pos.shape)
        
        root_rot = torch_util.quat_mul(heading_inv_rot_expand, root_rot)

        if (len(key_pos) > 0):
            heading_inv_rot_expand = heading_inv_rot_expand.unsqueeze(-2)
            heading_inv_rot_expand = heading_inv_rot_expand.repeat((1, 1, key_pos.shape[2], 1))
            heading_inv_rot_flat = heading_inv_rot_expand.reshape((heading_inv_rot_expand.shape[0] * heading_inv_rot_expand.shape[1] * heading_inv_rot_expand.shape[2],
                                                                   heading_inv_rot_expand.shape[3]))
            key_pos_flat = key_pos.reshape((key_pos.shape[0] * key_pos.shape[1] * key_pos.shape[2],
                                            key_pos.shape[3]))
            key_pos_flat = torch_util.quat_rotate(heading_inv_rot_flat, key_pos_flat)
            key_pos = key_pos_flat.reshape(key_pos.shape)

    if (root_height_obs):
        root_pos_obs[..., 2] = root_pos[..., 2]
    else:
        root_pos_obs = root_pos_obs[..., :2]

    root_rot_flat = torch.reshape(root_rot, [root_rot.shape[0] * root_rot.shape[1], root_rot.shape[2]])
    root_rot_obs_flat = torch_util.quat_to_tan_norm(root_rot_flat)
    root_rot_obs = torch.reshape(root_rot_obs_flat, [root_rot.shape[0], root_rot.shape[1], root_rot_obs_flat.shape[-1]])

    joint_rot_flat = torch.reshape(joint_rot, [joint_rot.shape[0] * joint_rot.shape[1] * joint_rot.shape[2], joint_rot.shape[3]])
    joint_rot_obs_flat = torch_util.quat_to_tan_norm(joint_rot_flat)
    joint_rot_obs = torch.reshape(joint_rot_obs_flat, [joint_rot.shape[0], joint_rot.shape[1], joint_rot.shape[2] * joint_rot_obs_flat.shape[-1]])
    
    obs = [root_pos_obs, root_rot_obs, joint_rot_obs]
    if (len(key_pos) > 0):
        key_pos = torch.reshape(key_pos, [key_pos.shape[0], key_pos.shape[1], key_pos.shape[2] * key_pos.shape[3]])
        obs.append(key_pos)

    obs = torch.cat(obs, dim=-1)

    return obs

@torch.jit.script
def compute_deepmimic_obs(root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, key_pos, global_obs, root_height_obs, 
                          phase, num_phase_encoding, enable_phase_obs, 
                          enable_tar_obs, tar_root_pos, tar_root_rot, tar_joint_rot, tar_key_pos):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, Tensor, int, bool, bool, Tensor, Tensor, Tensor, Tensor) -> Tensor
    char_obs = ig_char_env.compute_char_obs(root_pos=root_pos,
                                            root_rot=root_rot,
                                            root_vel=root_vel,
                                            root_ang_vel=root_ang_vel,
                                            joint_rot=joint_rot,
                                            dof_vel=dof_vel,
                                            key_pos=key_pos,
                                            global_obs=global_obs,
                                            root_height_obs=root_height_obs)
    obs = [char_obs]

    if (enable_phase_obs):
        phase_obs = compute_phase_obs(phase=phase, num_phase_encoding=num_phase_encoding)
        obs.append(phase_obs)

    if (enable_tar_obs):
        if (global_obs):
            ref_root_pos = root_pos
            ref_root_rot = root_rot
        else:
            ref_root_pos = tar_root_pos[..., 0, :]
            ref_root_rot = tar_root_rot[..., 0, :]

        tar_obs = compute_tar_obs(ref_root_pos=ref_root_pos,
                                  ref_root_rot=ref_root_rot,
                                  root_pos=tar_root_pos, 
                                  root_rot=tar_root_rot, 
                                  joint_rot=tar_joint_rot,
                                  key_pos=tar_key_pos,
                                  global_obs=global_obs,
                                  root_height_obs=root_height_obs)
        
        tar_obs = torch.reshape(tar_obs, [tar_obs.shape[0], tar_obs.shape[1] * tar_obs.shape[2]])
        obs.append(tar_obs)

    obs = torch.cat(obs, dim=-1)
    
    return obs

@torch.jit.script
def compute_done(done_buf, time, ep_len, root_rot, body_pos, tar_root_rot, tar_body_pos, 
                 contact_force, contact_body_ids, termination_heights,
                 pose_termination, pose_termination_dist, 
                 global_obs, enable_early_termination,
                 motion_times, motion_len, motion_len_term,
                 track_root):
    # type: (Tensor, Tensor, float, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, float, bool, bool, Tensor, Tensor, Tensor, bool) -> Tensor
    done = torch.full_like(done_buf, base_env.DoneFlags.NULL.value)
    
    timeout = time >= ep_len
    done[timeout] = base_env.DoneFlags.TIME.value
    
    motion_end = motion_times >= motion_len
    motion_end = torch.logical_and(motion_end, motion_len_term)
    # setting the done flag flat to fail at the end of the motion avoids the
    # local minimal of a character just standing still until the end of the motion
    done[motion_end] = base_env.DoneFlags.FAIL.value

    if (enable_early_termination):
        failed = torch.zeros(done.shape, device=done.device, dtype=torch.bool)

        if (contact_body_ids.shape[0] > 0):
            masked_contact_buf = contact_force.detach().clone()
            masked_contact_buf[:, contact_body_ids, :] = 0
            fall_contact = torch.any(torch.abs(masked_contact_buf) > 0.1, dim=-1)

            body_height = body_pos[..., 2]
            fall_height = body_height < termination_heights
            fall_height[:, contact_body_ids] = False

            fall_contact = torch.logical_and(fall_contact, fall_height)
            has_fallen = torch.any(fall_contact, dim=-1)
            failed = torch.logical_or(failed, has_fallen)

        if (pose_termination):
            root_pos = body_pos[..., 0:1, :]
            tar_root_pos = tar_body_pos[..., 0:1, :]
            body_pos = body_pos[..., 1:, :] - root_pos
            tar_body_pos = tar_body_pos[..., 1:, :] - tar_root_pos

            if (not global_obs):
                body_pos = ig_char_env.convert_to_local_root_body_pos(root_rot, body_pos)
                tar_body_pos = ig_char_env.convert_to_local_root_body_pos(tar_root_rot, tar_body_pos)

            body_pos_diff = tar_body_pos - body_pos
            body_pos_dist = torch.sum(body_pos_diff * body_pos_diff, dim=-1)
            body_pos_dist = torch.max(body_pos_dist, dim=-1)[0]
            pose_fail = body_pos_dist > pose_termination_dist * pose_termination_dist

            if (track_root):
                root_pos_diff = tar_root_pos - root_pos
                root_pos_dist = torch.sum(root_pos_diff * root_pos_diff, dim=-1)
                root_pos_fail = root_pos_dist > pose_termination_dist * pose_termination_dist
                root_pos_fail = root_pos_fail.squeeze(-1)
                pose_fail = torch.logical_or(pose_fail, root_pos_fail)

            failed = torch.logical_or(failed, pose_fail)
            
        # only fail after first timestep
        not_first_step = (time > 0.0)
        failed = torch.logical_and(failed, not_first_step)
        done[failed] = base_env.DoneFlags.FAIL.value
    
    return done

@torch.jit.script
def compute_reward(root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, key_pos,
                   tar_root_pos, tar_root_rot, tar_root_vel, tar_root_ang_vel,
                   tar_joint_rot, tar_dof_vel, tar_key_pos,
                   joint_rot_err_w, dof_err_w, track_root_h,
                   track_root):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool) -> Tensor
    pose_w = 0.5
    vel_w = 0.1
    root_pose_w = 0.15
    root_vel_w = 0.1
    key_pos_w = 0.15

    pose_scale = 0.25
    vel_scale = 0.01
    root_pose_scale = 5.0
    root_vel_scale = 1.0
    key_pos_scale = 10.0

    pose_diff = torch_util.quat_diff_angle(joint_rot, tar_joint_rot)
    pose_err = torch.sum(joint_rot_err_w * pose_diff * pose_diff, dim=-1)

    vel_diff = tar_dof_vel - dof_vel
    vel_err = torch.sum(dof_err_w * vel_diff * vel_diff, dim=-1)

    root_pos_diff = tar_root_pos - root_pos

    if (not track_root):
        root_pos_diff[..., 0:2] = 0

    if (not track_root_h):
        root_pos_diff[..., 2] = 0

    root_pos_err = torch.sum(root_pos_diff * root_pos_diff, dim=-1)
    
    if (len(key_pos) > 0):
        key_pos = key_pos - root_pos.unsqueeze(-2)
        tar_key_pos = tar_key_pos - tar_root_pos.unsqueeze(-2)

    if (not track_root):
        root_rot, root_vel, root_ang_vel, key_pos = convert_to_local(root_rot, root_vel, root_ang_vel, key_pos)
        tar_root_rot, tar_root_vel, tar_root_ang_vel, tar_key_pos = convert_to_local(tar_root_rot, tar_root_vel, tar_root_ang_vel, tar_key_pos)
        
    root_rot_err = torch_util.quat_diff_angle(root_rot, tar_root_rot)
    root_rot_err *= root_rot_err

    root_vel_diff = tar_root_vel - root_vel
    root_vel_err = torch.sum(root_vel_diff * root_vel_diff, dim=-1)

    root_ang_vel_diff = tar_root_ang_vel - root_ang_vel
    root_ang_vel_err = torch.sum(root_ang_vel_diff * root_ang_vel_diff, dim=-1)

    if (len(key_pos) > 0):
        key_pos_diff = tar_key_pos - key_pos
        key_pos_err = torch.sum(key_pos_diff * key_pos_diff, dim=-1)
        key_pos_err = torch.sum(key_pos_err, dim=-1)
    else:
        key_pos_err = torch.zeros([0], device=key_pos.device)

    pose_r = torch.exp(-pose_scale * pose_err)
    vel_r = torch.exp(-vel_scale * vel_err)
    root_pose_r = torch.exp(-root_pose_scale * (root_pos_err + 0.1 * root_rot_err))
    root_vel_r = torch.exp(-root_vel_scale * (root_vel_err + 0.1 * root_ang_vel_err))
    key_pos_r = torch.exp(-key_pos_scale * key_pos_err)

    r = pose_w * pose_r \
        + vel_w * vel_r \
        + root_pose_w * root_pose_r \
        + root_vel_w * root_vel_r \
        + key_pos_w * key_pos_r

    return r

@torch.jit.script
def compute_tracking_error(root_pos, root_rot, body_rot, body_pos,
                            tar_root_pos, tar_root_rot,
                            tar_body_rot, tar_body_pos,
                            root_vel, root_ang_vel, dof_vel,
                            tar_root_vel, tar_root_ang_vel, tar_dof_vel):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    body_pos = body_pos - root_pos.unsqueeze(-2)
    tar_body_pos = tar_body_pos - tar_root_pos.unsqueeze(-2)

    root_pos_diff = tar_root_pos - root_pos
    root_pos_diff_l2 = torch.linalg.vector_norm(root_pos_diff, dim=-1)

    pose_diff = torch_util.quat_diff_angle(body_rot, tar_body_rot)
    pose_err = torch.abs(pose_diff)
    pose_err = torch.mean(pose_err, dim=-1)

    body_pos_diff = tar_body_pos - body_pos
    body_pos_diff_l2 = torch.linalg.vector_norm(body_pos_diff, dim=-1)
    body_pos_err = torch.mean(body_pos_diff_l2, dim=-1)

    root_rot_diff = torch_util.quat_diff_angle(root_rot, tar_root_rot)
    root_rot_err = torch.abs(root_rot_diff)

    dof_vel_diff = tar_dof_vel - dof_vel
    dof_vel_err = torch.mean(torch.abs(dof_vel_diff), dim=-1)

    root_vel_diff = tar_root_vel - root_vel
    root_vel_err = torch.mean(torch.abs(root_vel_diff), dim=-1)

    root_ang_vel_diff = tar_root_ang_vel - root_ang_vel
    root_ang_vel_err = torch.mean(torch.abs(root_ang_vel_diff), dim=-1)

    tracking_error = torch.stack([root_pos_diff_l2, root_rot_err, body_pos_err, pose_err, dof_vel_err, root_vel_err, root_ang_vel_err], dim=-1)
    return tracking_error