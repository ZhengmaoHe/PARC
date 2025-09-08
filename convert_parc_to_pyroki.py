import argparse
import pickle
import torch
import numpy as np
from anim.kin_char_model import KinCharModel
import util.torch_util as torch_util
import sys
sys.path.append('/home/ubuntu/myProject/PARC')
import util.terrain_util as terrain_util

# SMPL joint names for reference (45 keypoints)
SMPL_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine_1", "left_knee", "right_knee", "spine_2",
    "left_ankle", "right_ankle", "spine_3", "left_foot", "right_foot", "neck",
    "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand",
    "nose", "right_eye", "left_eye", "right_ear", "left_ear",
    "left_big_toe", "left_small_toe", "left_heel", "right_big_toe", "right_small_toe", "right_heel",
    "left_thumb", "left_index", "left_middle", "left_ring", "left_pinky",
    "right_thumb", "right_index", "right_middle", "right_ring", "right_pinky"
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl', type=str, default='/home/ubuntu/myProject/PARC/parc_dataset/april272025/iter_3/teaser_2000_2049/teaser_2000_0_opt_dm.pkl', help='Path to input .pkl file')
    parser.add_argument('--xml', type=str, default='data/assets/humanoid.xml', help='Path to humanoid.xml')
    parser.add_argument('--output_dir', type=str, default='/home/ubuntu/myProject/PARC/pyroki_retarget/teaser_2000_2049/', help='Directory to output npy files')
    args = parser.parse_args()

    device = 'cpu'
    model = KinCharModel(device)
    import ipdb; ipdb.set_trace()
    model.load_char_file(args.xml)
    with open(args.pkl, 'rb') as f:
        data = pickle.load(f)

    frames = torch.tensor(data['frames'], dtype=torch.float32, device=device)
    fps = data['fps']
    num_frames = frames.shape[0]

    if 'terrain' in data:
        terrain = data['terrain']
        terrain.to_torch(device)
        padding = 2.0
        sliced_terrain, adjusted_frames = terrain_util.slice_terrain_around_motion(frames, terrain, padding=padding)
        frames = adjusted_frames
        heightmap = sliced_terrain.hf.cpu().numpy()
        dxdy = sliced_terrain.dxdy.cpu().numpy()
    else:
        heightmap = np.zeros((100,100))
        dxdy = np.array([0.4, 0.4])

    root_pos = frames[:, 0:3]
    root_rot_exp = frames[:, 3:6]
    root_rot_quat = torch_util.exp_map_to_quat(root_rot_exp)
    joint_dof = frames[:, 6:]
    joint_rot = model.dof_to_rot(joint_dof)
    body_pos, body_rot = model.forward_kinematics(root_pos, root_rot_quat, joint_rot)

    # Mapping from humanoid body names to SMPL joint indices
    humanoid_to_smpl = {
        'pelvis': 0,
        'left_thigh': 1,
        'right_thigh': 2,
        'torso': 3,  # spine_1
        'left_shin': 4,
        'right_shin': 5,
        'upper_torso': 6,  # assume torso is spine_2
        'left_foot': 7,
        'right_foot': 8,
        'neck': 9,  # spine_3
        'left_foot_toe': 10,
        'right_foot_toe': 11,
        'head': 15,
        'left_upper_arm': 16,
        'right_upper_arm': 17,
        'left_lower_arm': 18,
        'right_lower_arm': 19,
        'left_hand': 20,
        'right_hand': 21,
        # Add more mappings as needed, set others to None or handle zeros
    }

    smpl_keypoints = np.zeros((num_frames, 45, 3))
    for body_name, smpl_idx in humanoid_to_smpl.items():
        if body_name in model._name_body_map:
            model_id = model._name_body_map[body_name]
            smpl_keypoints[:, smpl_idx] = body_pos[:, model_id].numpy()

    # Foot contacts
    contacts = data.get('contacts', np.zeros((num_frames, model.get_num_joints())))
    left_foot_id = model.get_body_id('left_foot')
    right_foot_id = model.get_body_id('right_foot')
    eps = 1e-5
    left_foot_contact = contacts[:, left_foot_id]
    right_foot_contact = contacts[:, right_foot_id]

    # Save to output_dir
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    # import ipdb; ipdb.set_trace()
    np.save(os.path.join(args.output_dir, 'smpl_keypoints.npy'), smpl_keypoints)
    np.save(os.path.join(args.output_dir, 'left_foot_contact.npy'), left_foot_contact)
    np.save(os.path.join(args.output_dir, 'right_foot_contact.npy'), right_foot_contact)
    np.save(os.path.join(args.output_dir, 'heightmap.npy'), heightmap)
    np.save(os.path.join(args.output_dir, 'dxdy.npy'), dxdy)

    print(f'Files saved to {args.output_dir}')

if __name__ == '__main__':
    main()
