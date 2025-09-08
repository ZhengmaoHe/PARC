import argparse
import pickle
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from anim.kin_char_model import KinCharModel
import util.torch_util as torch_util

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl', type=str, default='/home/ubuntu/myProject/PARC/parc_dataset/april272025/iter_3/teaser_2000_2049/teaser_2000_0_opt_dm.pkl', help='Path to input .pkl file')
    parser.add_argument('--xml', type=str, default='data/assets/humanoid.xml', help='Path to humanoid.xml')
    parser.add_argument('--output', type=str, default='/home/ubuntu/myProject/PARC/parc_dataset/april272025/iter_3/teaser_2000_2049/teaser_2000_0_opt_dm.bvh', help='Path to output .bvh file')
    args = parser.parse_args()

    device = 'cpu'
    model = KinCharModel(device)
    model.load_char_file(args.xml)

    with open(args.pkl, 'rb') as f:
        data = pickle.load(f)

    frames = torch.tensor(data['frames'], dtype=torch.float32, device=device)
    fps = data['fps']
    num_frames = frames.shape[0]

    root_pos = frames[:, 0:3]
    root_rot_exp = frames[:, 3:6]
    root_rot_quat = torch_util.exp_map_to_quat(root_rot_exp)
    joint_dof = frames[:, 6:]
    joint_rot = model.dof_to_rot(joint_dof)
    body_pos, body_rot = model.forward_kinematics(root_pos, root_rot_quat, joint_rot)

    # Define bone data
    bone_data = [
        {'name': 'Hips', 'parent': None, 'model_id': model.get_body_id('pelvis'), 'channels': ['Xposition', 'Yposition', 'Zposition', 'Zrotation', 'Xrotation', 'Yrotation']},
        {'name': 'Spine2', 'parent': 'Hips', 'model_id': model.get_body_id('torso'), 'channels': ['Zrotation', 'Xrotation', 'Yrotation']},
        {'name': 'Head', 'parent': 'Spine2', 'model_id': model.get_body_id('head'), 'channels': ['Zrotation', 'Xrotation', 'Yrotation']},
        {'name': 'LeftArm', 'parent': 'Spine2', 'model_id': model.get_body_id('left_upper_arm'), 'channels': ['Zrotation', 'Xrotation', 'Yrotation']},
        {'name': 'LeftForeArm', 'parent': 'LeftArm', 'model_id': model.get_body_id('left_lower_arm'), 'channels': ['Zrotation', 'Xrotation', 'Yrotation']},
        {'name': 'LeftHand', 'parent': 'LeftForeArm', 'model_id': model.get_body_id('left_hand'), 'channels': ['Zrotation', 'Xrotation', 'Yrotation']},
        {'name': 'RightArm', 'parent': 'Spine2', 'model_id': model.get_body_id('right_upper_arm'), 'channels': ['Zrotation', 'Xrotation', 'Yrotation']},
        {'name': 'RightForeArm', 'parent': 'RightArm', 'model_id': model.get_body_id('right_lower_arm'), 'channels': ['Zrotation', 'Xrotation', 'Yrotation']},
        {'name': 'RightHand', 'parent': 'RightForeArm', 'model_id': model.get_body_id('right_hand'), 'channels': ['Zrotation', 'Xrotation', 'Yrotation']},
        {'name': 'LeftUpLeg', 'parent': 'Hips', 'model_id': model.get_body_id('left_thigh'), 'channels': ['Zrotation', 'Xrotation', 'Yrotation']},
        {'name': 'LeftLeg', 'parent': 'LeftUpLeg', 'model_id': model.get_body_id('left_shin'), 'channels': ['Zrotation', 'Xrotation', 'Yrotation']},
        {'name': 'LeftFoot', 'parent': 'LeftLeg', 'model_id': model.get_body_id('left_foot'), 'channels': ['Zrotation', 'Xrotation', 'Yrotation']},
        {'name': 'LeftToe', 'parent': 'LeftFoot', 'model_id': model.get_body_id('left_foot'), 'offset': np.array([8.85, 0, 0]), 'channels': ['Zrotation', 'Xrotation', 'Yrotation']},  # dummy offset in cm
        {'name': 'RightUpLeg', 'parent': 'Hips', 'model_id': model.get_body_id('right_thigh'), 'channels': ['Zrotation', 'Xrotation', 'Yrotation']},
        {'name': 'RightLeg', 'parent': 'RightUpLeg', 'model_id': model.get_body_id('right_shin'), 'channels': ['Zrotation', 'Xrotation', 'Yrotation']},
        {'name': 'RightFoot', 'parent': 'RightLeg', 'model_id': model.get_body_id('right_foot'), 'channels': ['Zrotation', 'Xrotation', 'Yrotation']},
        {'name': 'RightToe', 'parent': 'RightFoot', 'model_id': model.get_body_id('right_foot'), 'offset': np.array([8.85, 0, 0]), 'channels': ['Zrotation', 'Xrotation', 'Yrotation']},  # dummy
    ]

    # Compute offsets
    for bone in bone_data:
        if 'offset' not in bone:
            model_id = bone['model_id']
            bone['offset'] = model._local_translation[model_id].numpy() * 100  # to cm

    # Build parent map
    name_to_idx = {bone['name']: i for i, bone in enumerate(bone_data)}

    # Write HIERARCHY
    def write_hierarchy(f, bone_idx, indent=''):
        bone = bone_data[bone_idx]
        if bone['parent'] is None:
            f.write(f'ROOT {bone["name"]}\n')
        else:
            f.write(f'{indent}JOINT {bone["name"]}\n')
        f.write(f'{indent}{{\n')
        f.write(f'{indent}\tOFFSET {bone["offset"][0]:.6f} {bone["offset"][1]:.6f} {bone["offset"][2]:.6f}\n')
        f.write(f'{indent}\tCHANNELS {len(bone["channels"])} {" ".join(bone["channels"])}\n')

        children = [i for i, b in enumerate(bone_data) if b['parent'] == bone['name']]
        for child_idx in children:
            write_hierarchy(f, child_idx, indent + '\t')

        if not children:
            f.write(f'{indent}\tEnd Site\n{indent}\t{{\n{indent}\t\tOFFSET 0 0 0\n{indent}\t}}\n')

        f.write(f'{indent}}}\n')

    # Collect motion data
    motion_lines = []
    for frame in range(num_frames):
        data = []
        for bone_idx, bone in enumerate(bone_data):
            model_id = bone['model_id']
            g_pos = body_pos[frame, model_id].numpy() * 100  # to cm
            g_rot = body_rot[frame, model_id].numpy()
            g_rot_scipy = g_rot[[1,2,3,0]]  # to xyzw

            if bone['parent'] is None:
                # Root position
                data.extend(g_pos.tolist())
                # Root euler ZXY
                rot = R.from_quat(g_rot_scipy)
                euler = rot.as_euler('ZXY', degrees=True)
                data.extend(euler.tolist())
            else:
                parent_idx = name_to_idx[bone['parent']]
                parent_model_id = bone_data[parent_idx]['model_id']
                p_rot = body_rot[frame, parent_model_id].numpy()
                local_rot = torch_util.quat_mul(torch_util.quat_conjugate(torch.tensor(p_rot)), torch.tensor(g_rot)).numpy()
                local_rot_scipy = local_rot[[1,2,3,0]]
                rot = R.from_quat(local_rot_scipy)
                euler = rot.as_euler('ZXY', degrees=True)
                data.extend(euler.tolist())

        motion_lines.append(' '.join(f'{v:.6f}' for v in data))
        import ipdb; ipdb.set_trace()

    # Write BVH file
    with open(args.output, 'w') as f:
        f.write('HIERARCHY\n')
        write_hierarchy(f, 0)
        f.write('MOTION\n')
        f.write(f'Frames: {num_frames}\n')
        f.write(f'Frame Time: {1.0 / fps}\n')
        for line in motion_lines:
            f.write(line + '\n')

    print(f'BVH file written to {args.output}')

if __name__ == '__main__':
    main()
