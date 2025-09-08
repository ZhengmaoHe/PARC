"""Humanoid Retargeting

Simpler motion retargeting to the G1 humanoid.
"""

import time
from pathlib import Path
from typing import Tuple, TypedDict

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp
import pyroki as pk
import viser
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf

from _utils import (
    SMPL_JOINT_NAMES,
    create_conn_tree,
    get_humanoid_retarget_indices,
)

import argparse
import pickle
import os

class RetargetingWeights(TypedDict):
    local_alignment: float
    """Local alignment weight, by matching the relative joint/keypoint positions and angles."""
    global_alignment: float
    """Global alignment weight, by matching the keypoint positions to the robot."""


def main():
    """Main function for humanoid retargeting."""

    urdf = load_robot_description("g1_description")
    robot = pk.Robot.from_urdf(urdf)

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="pyroki_retarget/teaser_2450_2499/teaser_2450_0_opt_dm")
    parser.add_argument('--output', type=str, default="pyroki_retarget/teaser_2450_2499/teaser_2400_0teaser_2450_0_opt_dm_opt_dm.pkl")

    args = parser.parse_args()
    args.keypoints = os.path.join(args.input_dir, 'smpl_keypoints.npy')
    args.left_contact = os.path.join(args.input_dir, 'left_foot_contact.npy')
    args.right_contact = os.path.join(args.input_dir, 'right_foot_contact.npy')
    args.heightmap = os.path.join(args.input_dir, 'heightmap.npy')

    # Load data from args
    smpl_keypoints = onp.load(args.keypoints)
    is_left_foot_contact = onp.load(args.left_contact)
    is_right_foot_contact = onp.load(args.right_contact)
    heightmap = onp.load(args.heightmap).transpose()
    grid_shape = onp.load(os.path.join(os.path.dirname(args.heightmap), 'grid_shape.npy'))

    num_timesteps = smpl_keypoints.shape[0]
    assert smpl_keypoints.shape == (num_timesteps, 45, 3)
    # assert is_left_foot_contact.shape == (num_timesteps,)
    # assert is_right_foot_contact.shape == (num_timesteps,)

    offset = onp.load(os.path.join(os.path.dirname(args.heightmap), 'offset.npy'))
    dxdy = onp.load(os.path.join(os.path.dirname(args.heightmap), 'dxdy.npy'))
    import ipdb; ipdb.set_trace()
    # import ipdb; ipdb.set_trace()
    center_offset = offset + ((grid_shape-1) * dxdy / 2)
    heightmap = pk.collision.Heightmap(
        # pose=jaxlie.SE3.identity(),
        pose = jaxlie.SE3.from_translation(jnp.array([center_offset[0], center_offset[1], 0.0])),
        size=jnp.array([dxdy[1], dxdy[0], 1.0]),
        height_data=heightmap,
    )

    # Get the left and right foot keypoints, projected on the heightmap.
    left_foot_keypoint_idx = SMPL_JOINT_NAMES.index("left_foot")
    right_foot_keypoint_idx = SMPL_JOINT_NAMES.index("right_foot")
    left_foot_keypoints = smpl_keypoints[..., left_foot_keypoint_idx, :].reshape(-1, 3)
    right_foot_keypoints = smpl_keypoints[..., right_foot_keypoint_idx, :].reshape(
        -1, 3
    )
    left_foot_keypoints = heightmap.project_points(left_foot_keypoints)
    right_foot_keypoints = heightmap.project_points(right_foot_keypoints)

    smpl_joint_retarget_indices, g1_joint_retarget_indices = (
        get_humanoid_retarget_indices()
    )
    smpl_mask = create_conn_tree(robot, g1_joint_retarget_indices)

    server = viser.ViserServer()
    base_frame = server.scene.add_frame("/base", show_axes=False)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")
    playing = server.gui.add_checkbox("playing", True)
    timestep_slider = server.gui.add_slider("timestep", 0, num_timesteps - 1, 1, 0)
    server.scene.add_mesh_trimesh("/heightmap", heightmap.to_trimesh())

    weights = pk.viewer.WeightTuner(
        server,
        RetargetingWeights(  # type: ignore
            local_alignment=2.0,
            global_alignment=1.0,
        ),
    )


    while True:
        with server.atomic():
            if playing.value:
                timestep_slider.value = (timestep_slider.value + 1) % num_timesteps
            tstep = timestep_slider.value
            server.scene.add_point_cloud(
                "/target_keypoints",
                onp.array(smpl_keypoints[tstep]),
                onp.array((0, 0, 255))[None].repeat(45, axis=0),
                point_size=0.01,
            )

        time.sleep(3/30)

if __name__ == "__main__":
    main()
