# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# TODO: remove numpy, make this all torch
import numpy as np
import copy

import torch
import random
import util.torch_util as torch_util
import util.geom_util as geom_util
import anim.kin_char_model as kin_char_model
import math
from typing import List
from collections import OrderedDict

class SubTerrain:
    def __init__(self, terrain_name="terrain", x_dim=256, y_dim=256, dx=1.0, dy =1.0, min_x=-1.0, min_y=-1.0,
                 device="cuda:0"):
        self.terrain_name = terrain_name
        # self.dims = np.array([x_dim, y_dim], dtype=np.int64)
        # self.hf = np.zeros((x_dim, y_dim), dtype=np.float32)
        # self.min_point = np.array([min_x, min_y], dtype=np.float32)
        # self.dxdy = np.array([dx, dy], dtype=np.float32)

    # def to_torch(self, device):
        self.hf = torch.zeros(size=(x_dim, y_dim), dtype=torch.float32, device=device)
        self.dims = torch.tensor([x_dim, y_dim], dtype=torch.int64, device=device)
        self.min_point = torch.tensor([min_x, min_y], dtype=torch.float32, device=device)
        self.dxdy = torch.tensor([dx, dy], dtype=torch.float32, device=device)
        self.hf_mask = torch.zeros(size=(x_dim, y_dim), dtype=torch.bool, device=device)
        self.hf_maxmin = torch.zeros(size=(x_dim, y_dim, 2), dtype=torch.float32, device=device) # max/min heights for randomization
        self.hf_maxmin[..., 0] = 1.0
        self.hf_maxmin[..., 1] = -1.0
        return
    
    def set_device(self, device):
        # only works with torch tensors
        self.hf = self.hf.to(device=device)
        self.dims = self.dims.to(device=device)
        self.min_point = self.min_point.to(device=device)
        self.dxdy = self.dxdy.to(device=device)
        self.hf_mask = self.hf_mask.to(device=device)
        self.hf_maxmin = self.hf_maxmin.to(device=device)
        return

    def to_torch(self, device):
        if not isinstance(self.hf, torch.Tensor):
            self.hf = torch.tensor(self.hf, dtype=torch.float32, device=device)
        else:
            self.hf = self.hf.to(device=device)

        if not isinstance(self.dims, torch.Tensor):
            self.dims = torch.tensor(self.dims, dtype=torch.int64, device=device)
        else:
            self.dims = self.dims.to(device=device)

        if not isinstance(self.min_point, torch.Tensor):
            self.min_point = torch.tensor(self.min_point, dtype=torch.float32, device=device)
        else:
            self.min_point = self.min_point.to(device=device)

        if not isinstance(self.dxdy, torch.Tensor):
            self.dxdy = torch.tensor(self.dxdy, dtype=torch.float32, device=device)
        else:
            self.dxdy = self.dxdy.to(device=device)

        if not isinstance(self.hf_mask, torch.Tensor):
            self.hf_mask = torch.tensor(self.hf_mask, dtype=torch.bool, device=device)
        else:
            self.hf_mask = self.hf_mask.to(device=device)

        if not isinstance(self.hf_maxmin, torch.Tensor):
            self.hf_maxmin = torch.tensor(self.hf_maxmin, dtype=torch.float32, device=device)
        else:
            self.hf_maxmin = self.hf_maxmin.to(device=device)
        return
    
    def to_numpy(self):
        self.hf = self.hf.detach().cpu().numpy()
        self.dims = self.dims.detach().cpu().numpy()
        self.min_point = self.min_point.detach().cpu().numpy()
        self.dxdy = self.dxdy.detach().cpu().numpy()
        self.hf_mask = self.hf_mask.detach().cpu().numpy()
        self.hf_maxmin = self.hf_maxmin.detach().cpu().numpy()
        return
    
    def numpy_copy(self):
        new_terrain = copy.deepcopy(self)
        new_terrain.to_numpy()
        return new_terrain
    
    def torch_copy(self):
        new_terrain = copy.deepcopy(self)
        new_terrain.hf = self.hf.clone()
        new_terrain.dims = self.dims.clone()
        new_terrain.min_point = self.min_point.clone()
        new_terrain.dxdy = self.dxdy.clone()
        new_terrain.hf_mask = self.hf_mask.clone()
        new_terrain.hf_maxmin = self.hf_maxmin.clone()
        return new_terrain
    
    def get_real_size(self):
        return self.dims * self.dxdy
    
    def get_max_point(self):
        return self.min_point + self.get_real_size() - self.dxdy # max point is in the cell center
    
    def get_inbounds_grid_index(self, grid_ind: torch.Tensor):
        grid_ind = torch.clamp(grid_ind, torch.zeros_like(self.dims), self.dims-1)
        return grid_ind
    
    def round_point_to_grid_point(self, point: torch.Tensor):
        return torch.round((point - self.min_point) / self.dxdy) * self.dxdy + self.min_point

    def round_point_to_grid_index(self, point: torch.Tensor):
        return torch.round((point - self.min_point) / self.dxdy).to(dtype=torch.int64)

    def get_grid_index(self, point: torch.Tensor):
        inds = self.round_point_to_grid_index(point)
        inds = self.get_inbounds_grid_index(grid_ind=inds)
        return inds
    
    def get_hf_val_from_points(self, xy_points):
        grid_ind = self.get_grid_index(xy_points)
        return self.hf[grid_ind[..., 0], grid_ind[..., 1]]
    
    def get_hf_val(self, grid_ind: torch.Tensor):
        grid_ind = self.get_inbounds_grid_index(grid_ind)
        return self.hf[grid_ind[0], grid_ind[1]]
    
    def set_hf_val(self, grid_ind, val: float):
        grid_ind = self.get_inbounds_grid_index(grid_ind)
        self.hf[grid_ind[0], grid_ind[1]] = val
        return
    
    def set_hf_mask_val(self, grid_ind, val: bool):
        grid_ind = self.get_inbounds_grid_index(grid_ind)
        self.hf_mask[grid_ind[0], grid_ind[1]] = val
        return
    
    def set_hf_max_val(self, grid_ind, val: float):
        grid_ind = self.get_inbounds_grid_index(grid_ind)
        self.hf_maxmin[grid_ind[0], grid_ind[1], 0] = val
        return
    
    def set_hf_min_val(self, grid_ind, val: float):
        grid_ind = self.get_inbounds_grid_index(grid_ind)
        self.hf_maxmin[grid_ind[0], grid_ind[1], 1] = val
        return
    
    # def flip_by_XZ_axis(self, y_ind, y_offset=0.0):
    #     # assumes torch tensors
    #     # flip the heightfield
    #     self.hf = torch.flip(self.hf, dims=[1])
    #     self.hf_mask = torch.flip(self.hf_mask, dims=[1])
    #     self.hf_maxmin[..., 0] = torch.flip(self.hf_maxmin[..., 0], dims=[1])
    #     self.hf_maxmin[..., 1] = torch.flip(self.hf_maxmin[..., 1], dims=[1])

    #     # then compute the min point based on the flipping y index
    #     num_pos_y = (self.dims[1]-1) - y_ind
    #     self.min_point[1] = -num_pos_y * self.dxdy[1] + y_offset
    #     return

    def flip_by_XZ_axis(self):
        self.hf = torch.flip(self.hf, dims=[1])
        self.hf_mask = torch.flip(self.hf_mask, dims=[1])
        self.hf_maxmin[..., 0] = torch.flip(self.hf_maxmin[..., 0], dims=[1])
        self.hf_maxmin[..., 1] = torch.flip(self.hf_maxmin[..., 1], dims=[1])
        
        # new minpoint is discovered by the old max_point[1]
        max_point = self.get_max_point()
        self.min_point[1] = -1.0 * max_point[1]
        return
    
    def get_origin_index(self):
        return self.get_grid_index(torch.zeros_like(self.min_point))
    
    def get_point(self, ij):
        return self.min_point + ij * self.dxdy
    
    def get_xyz_point(self, grid_inds):
        xy_point = self.get_point(grid_inds)
        z_point = self.hf[grid_inds[..., 0], grid_inds[..., 1]]
        xyz_point = torch.cat([xy_point, z_point.unsqueeze(-1)], dim=-1)
        return xyz_point
    
    def fill_in_hf_mask(self, num_neighbors=4):
        # num_neighbors: the min number of neighbors a cell must have to be filled
        hf_mask = copy.deepcopy(self.hf_mask)
        # TODO use conv kernel to count number of neighbors so this can be done a lot faster
        for i in range(1, self.hf_mask.shape[0]-1):
            for j in range(1, self.hf_mask.shape[1]-1):
                
                if self.hf_mask[i, j] == False:
                    num_true = 0
                    for ii in range(3):
                        for jj in range(3):
                            if self.hf_mask[i - 1 + ii, j - 1 + jj]:
                                num_true += 1
                    if num_true >= num_neighbors:
                        hf_mask[i, j] = True
        self.hf_mask = hf_mask
        return
    
    def update_old(self):
        if not hasattr(self, "hf_maxmin"):
            if isinstance(self.hf, torch.Tensor):
                self.hf_maxmin = torch.zeros(size=(self.hf.shape[0], self.hf.shape[1], 2), dtype=torch.float32, device=self.hf.device) # max/min heights for randomization
                self.hf_maxmin[..., 0] = 1.0
                self.hf_maxmin[..., 1] = -1.0
            else:
                self.hf_maxmin = np.zeros(shape=(self.hf.shape[0], self.hf.shape[1], 2), dtype=np.float32) # max/min heights for randomization
                self.hf_maxmin[..., 0] = 1.0
                self.hf_maxmin[..., 1] = -1.0

        return
    
    def pad(self, padding_size, height=0.0):
        # only works if terrain is torch
        self.hf = torch.nn.functional.pad(self.hf, [padding_size, padding_size, padding_size, padding_size], value=height)
        device=self.hf.device

        new_hf_mask = torch.zeros(size=[self.hf_mask.shape[0] + padding_size*2, self.hf_mask.shape[1] + padding_size*2],
                                    dtype=torch.bool, device=device)
        new_hf_mask[padding_size:-padding_size, padding_size:-padding_size] = self.hf_mask[:, :]
        self.hf_mask = new_hf_mask
        
        max_val = torch.max(self.hf_maxmin[..., 0]).item()
        min_val = torch.min(self.hf_maxmin[..., 1]).item()
        new_hf_max = torch.nn.functional.pad(self.hf_maxmin[..., 0], [padding_size, padding_size, padding_size, padding_size], value=max_val)
        new_hf_min = torch.nn.functional.pad(self.hf_maxmin[..., 1], [padding_size, padding_size, padding_size, padding_size], value=min_val)
        self.hf_maxmin = torch.cat([new_hf_max.unsqueeze(-1), new_hf_min.unsqueeze(-1)], dim=-1)
        self.min_point -= self.dxdy * padding_size
        self.dims += padding_size * 2
        return
    
    def convert_mask_to_maxmin(self):
        hf_vals = self.hf[self.hf_mask]
        self.hf_maxmin[..., 0][self.hf_mask] = hf_vals
        self.hf_maxmin[..., 1][self.hf_mask] = hf_vals
        return
    
    def get_grid_node_xy_points(self):
        dim_x = self.hf.shape[0]
        dim_y = self.hf.shape[1]
        x_points = torch.linspace(0.0, (dim_x - 1.0) * self.dxdy[0].item(), dim_x, device=self.hf.device)
        y_points = torch.linspace(0.0, (dim_y - 1.0) * self.dxdy[1].item(), dim_y, device=self.hf.device)

        x, y = torch.meshgrid(x_points, y_points, indexing='ij') # shape: [dim_x, dim_y]
        x = x + self.min_point[0]
        y = y + self.min_point[1]
        xy_points = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], dim=-1)
        return xy_points
    
def remove_sharp_lines(terrain:SubTerrain):
    def detect_sharp_line_high(i, j, eps=0.1):

        center_h = terrain.hf[i, j]
        
        test1 = center_h > terrain.hf[i-1, j] + eps and center_h > terrain.hf[i+1, j] + eps
        test2 = center_h > terrain.hf[i, j-1] + eps and center_h > terrain.hf[i, j+1] + eps

        return test1 or test2
    
    def detect_sharp_line_low(i, j, eps=0.1):

        center_h = terrain.hf[i, j]
        test1 = center_h < terrain.hf[i-1, j] - eps and center_h < terrain.hf[i+1, j] - eps
        test2 = center_h < terrain.hf[i, j-1] - eps and center_h < terrain.hf[i, j+1] - eps
        return test1 or test2


    for i in range(1, terrain.hf.shape[0]-1):
        for j in range(1, terrain.hf.shape[1]-1):
            if detect_sharp_line_high(i, j):
                min_h = min(terrain.hf[i-1, j].item(), terrain.hf[i+1, j].item(),
                    terrain.hf[i, j-1].item(), terrain.hf[i, j+1].item())
                terrain.hf[i, j] = min_h

            elif detect_sharp_line_low(i, j):
                max_h = max(terrain.hf[i-1, j].item(), terrain.hf[i+1, j].item(),
                    terrain.hf[i, j-1].item(), terrain.hf[i, j+1].item())
                terrain.hf[i, j] = max_h
    return

# from https://gist.github.com/vadimkantorov/ac1b097753f217c5c11bc2ff396e0a57
def rand_perlin_2d(shape, res, fade = lambda t: 6*t**5 - 15*t**4 + 10*t**3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    
    grid = torch.stack(torch.meshgrid(torch.arange(0, res[0], delta[0]), torch.arange(0, res[1], delta[1]), indexing='ij'), dim = -1) % 1
    angles = 2*math.pi*torch.rand(res[0]+1, res[1]+1)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim = -1)
    
    tile_grads = lambda slice1, slice2: gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]].repeat_interleave(d[0], 0).repeat_interleave(d[1], 1)
    dot = lambda grad, shift: (torch.stack((grid[:shape[0],:shape[1],0] + shift[0], grid[:shape[0],:shape[1], 1] + shift[1]  ), dim = -1) * grad[:shape[0], :shape[1]]).sum(dim = -1)
    
    n00 = dot(tile_grads([0, -1], [0, -1]), [0,  0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1],[1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1,-1])
    t = fade(grid[:shape[0], :shape[1]])
    return math.sqrt(2) * torch.lerp(torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1])

def rand_perlin_2d_octaves(shape, res, octaves=1, persistence=0.5):
    noise = torch.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * rand_perlin_2d(shape, (frequency*res[0], frequency*res[1]))
        frequency *= 2
        amplitude *= persistence
    return noise

def linear_parkour_course(terrain: SubTerrain, block_centers, block_heights, block_dims):
    ## Creates a parkour course stretched out by the x-length of the terrain
    num_blocks = len(block_centers)
    assert num_blocks == len(block_heights)
    assert num_blocks == len(block_dims)

    # Note: each vertex in the height map gets a dx by dy rect centered around it
    dx = terrain.dxdy[0]
    dy = terrain.dxdy[1]
    triangles = []
    vertices = []


    
    x_len = terrain.dxdy[0] * terrain.dims[0] #terrain.min_point[0] + terrain.dxdy[0]
    y_len = terrain.dxdy[1] * terrain.dims[1]
    min_x = terrain.min_point[0]
    min_y = terrain.min_point[1]
    max_x = min_x + x_len
    # 2 extra verts for front running plane, 2 verts for back running plane
    vertices = np.zeros(shape=(num_blocks*8 + 2 + 2, 3), dtype=np.float32)
    # 2 extra tris for back running plane
    triangles = np.zeros(shape=(num_blocks*8 + 2, 3), dtype=np.uint32)

    # initial 2 vertices
    vertices[0, :] = [min_x - dx/2, min_y - dy/2, 0.0]
    vertices[1, :] = [max_x + dx/2, min_y - dy/2, 0.0]

    


    for i in range(num_blocks):

        ## Fill in height field data along y-axis
        y = block_centers[i]
        h = block_heights[i]
        w = block_dims[i]

        # pick the heightmap indices
        start_y = y - w//2
        end_y = y + w//2
        #block_front = y - w//2
        #block_back = y + w//2

        #start_y = int(np.round((block_front - min_x) / terrain.dxdy[1]))
        #end_y = int(np.round((block_back - min_y) / terrain.dxdy[1]))

        #print((start_y-end_y)*terrain.dxdy[1])

        terrain.hf[:, start_y:end_y+1] = h

        ## Fill in triangle mesh data along y-axis
        # Each "block" of the parkour course has 4 quad / 8 triangles
        # 1 for the running area, 3 for the vault/gap
        # To construct these quads, assuming we can borrow the furthest 2 vertices from the previous block,
        # we need 8 vertices per block

        #vertices = [[]]

        v_ind = i * 8 + 2

        block_front = min_y + start_y * dy - dy/2
        block_back = min_y + end_y * dy + dy/2
        if h == 0.0:
            # make sure the triangles aren't degenerate
            vertices[v_ind + 0, :] = [min_x - dx/2, block_front - dy, 0.0]
            vertices[v_ind + 1, :] = [max_x + dx/2, block_front - dy, 0.0]
            vertices[v_ind + 2, :] = [min_x - dx/2, block_front, 0.0]
            vertices[v_ind + 3, :] = [max_x + dx/2, block_front, 0.0]
            vertices[v_ind + 4, :] = [min_x - dx/2, block_back - dy, 0.0]
            vertices[v_ind + 5, :] = [max_x + dx/2, block_back - dy, 0.0]
            vertices[v_ind + 6, :] = [min_x - dx/2, block_back, 0.0]
            vertices[v_ind + 7, :] = [max_x + dx/2, block_back, 0.0]
        else:
            #print("MESH GAP TRUE WIDTH", block_back - block_front)
            #print("w:", w)
            #print("y:", y)

            vertices[v_ind + 0, :] = [min_x - dx/2, block_front, 0.0]
            vertices[v_ind + 1, :] = [max_x + dx/2, block_front, 0.0]
            vertices[v_ind + 2, :] = [min_x - dx/2, block_front, h]
            vertices[v_ind + 3, :] = [max_x + dx/2, block_front, h]
            vertices[v_ind + 4, :] = [min_x - dx/2, block_back, h]
            vertices[v_ind + 5, :] = [max_x + dx/2, block_back, h]
            vertices[v_ind + 6, :] = [min_x - dx/2, block_back, 0.0]
            vertices[v_ind + 7, :] = [max_x + dx/2, block_back, 0.0]

        t_ind = i * 8
        # counter clock-wise = normals facing outward
        # quad running
        for _ in range(4):
            triangles[t_ind + 0, :] = [v_ind - 2, v_ind - 1, v_ind + 0]
            triangles[t_ind + 1, :] = [v_ind - 1, v_ind + 1, v_ind + 0]
            t_ind += 2
            v_ind += 2
        # quad block front
            
    # final vertices and final quad
    v_ind = num_blocks * 8 + 2
    vertices[v_ind + 0, :] = [min_x - dx/2, y_len + dy/2, 0.0]
    vertices[v_ind + 1, :] = [max_x + dx/2, y_len + dy/2, 0.0]
    t_ind = num_blocks * 8
    triangles[t_ind + 0, :] = [v_ind - 2, v_ind - 1, v_ind + 0]
    triangles[t_ind + 1, :] = [v_ind - 1, v_ind + 1, v_ind + 0]

    return terrain, vertices, triangles

def random_linear_parkour_course(terrain: SubTerrain,
                                 gap_width, gap_height,
                                 vault_width, vault_height,
                                 num_padding_cells # padding offset at edges of terrain,
                                 ):
    # TODO: MAKE MORE RANDOM, AND KEEP TRACK OF THE VALID RESET LOCATIONS?
    # real_block_centers = (np.arange(1, num_blocks+1)) * block_spacing# - 1.0
    # block_centers = np.round(real_block_centers / terrain.dxdy[1]).astype(dtype=int)
    

    

    
    #block_centers = segment_start_locs + possible_block_distances[block_ids] + num_padding_cells

    # randomly decide block centers
    min_block_spacing = 6.5
    max_block_spacing = 8.0
    block_centers = []
    y = 0.0
    max_y_len = terrain.get_real_size()[1]
    dy = terrain.dxdy[1]
    while y < max_y_len:
        val = random.random()
        block_spacing = min_block_spacing if val < 0.5 else max_block_spacing
        y += block_spacing
        y_ind = int(round(y/dy)) + num_padding_cells
        block_centers.append(y_ind)

    block_centers = np.array(block_centers, dtype=np.int64)

    num_blocks = block_centers.shape[0]

    # decide whether it is a vault or gap
    # 0: vault, 1: gap, 2: flat
    possible_block_heights = np.array([vault_height, gap_height])#, 0.0])
    possible_block_widths = np.array([vault_width, gap_width])#, gap_width])

    block_ids = np.random.randint(0, possible_block_heights.shape[0], size=(num_blocks,))
    #block_ids = np.zeros(shape=(num_blocks,), dtype=np.int64) # only vault block

    block_heights = possible_block_heights[block_ids]
    block_widths = possible_block_widths[block_ids]


    return linear_parkour_course(terrain, block_centers, block_heights, block_widths)


def convert_heightfield_to_trimesh(height_field_raw, horizontal_scale, vertical_scale, slope_threshold=None):
    """
    Convert a heightfield array to a triangle mesh represented by vertices and triangles.
    Optionally, corrects vertical surfaces above the provide slope threshold:

        If (y2-y1)/(x2-x1) > slope_threshold -> Move A to A' (set x1 = x2). Do this for all directions.
                   B(x2,y2)
                  /|
                 / |
                /  |
        (x1,y1)A---A'(x2',y1)

    Parameters:
        height_field_raw (np.array): input heightfield
        horizontal_scale (float): horizontal scale of the heightfield [meters]
        vertical_scale (float): vertical scale of the heightfield [meters]
        slope_threshold (float): the slope threshold above which surfaces are made vertical. If None no correction is applied (default: None)
    Returns:
        vertices (np.array(float)): array of shape (num_vertices, 3). Each row represents the location of each vertex [meters]
        triangles (np.array(int)): array of shape (num_triangles, 3). Each row represents the indices of the 3 vertices connected by this triangle.
    """
    hf = height_field_raw
    num_rows = hf.shape[0]
    num_cols = hf.shape[1]

    y = np.linspace(0, (num_cols-1)*horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows-1)*horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)

    if slope_threshold is not None:

        slope_threshold *= horizontal_scale # / vertical_scale # mxu: hf is now floats, and vertical scale will always be 1.0
        move_x = np.zeros((num_rows, num_cols))
        move_y = np.zeros((num_rows, num_cols))
        move_corners = np.zeros((num_rows, num_cols))
        move_x[:num_rows-1, :] += (hf[1:num_rows, :] - hf[:num_rows-1, :] > slope_threshold)
        move_x[1:num_rows, :] -= (hf[:num_rows-1, :] - hf[1:num_rows, :] > slope_threshold)
        move_y[:, :num_cols-1] += (hf[:, 1:num_cols] - hf[:, :num_cols-1] > slope_threshold)
        move_y[:, 1:num_cols] -= (hf[:, :num_cols-1] - hf[:, 1:num_cols] > slope_threshold)
        move_corners[:num_rows-1, :num_cols-1] += (hf[1:num_rows, 1:num_cols] - hf[:num_rows-1, :num_cols-1] > slope_threshold)
        move_corners[1:num_rows, 1:num_cols] -= (hf[:num_rows-1, :num_cols-1] - hf[1:num_rows, 1:num_cols] > slope_threshold)
        xx += (move_x + move_corners*(move_x == 0)) * horizontal_scale
        yy += (move_y + move_corners*(move_y == 0)) * horizontal_scale

    # create triangle mesh vertices and triangles from the heightfield grid
    vertices = np.zeros((num_rows*num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = hf.flatten()# * vertical_scale# mxu: don't need this if heightfield is already floats
    triangles = -np.ones((2*(num_rows-1)*(num_cols-1), 3), dtype=np.uint32)
    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols-1) + i*num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1
        start = 2*i*(num_cols-1)
        stop = start + 2*(num_cols-1)
        triangles[start:stop:2, 0] = ind0
        triangles[start:stop:2, 1] = ind3
        triangles[start:stop:2, 2] = ind1
        triangles[start+1:stop:2, 0] = ind0
        triangles[start+1:stop:2, 1] = ind2
        triangles[start+1:stop:2, 2] = ind3

        if i % 10000 == 0:
            print(i ," /", num_rows, " quads completed")

    return vertices, triangles

def gen_paths_hf(terrain: SubTerrain, num_paths=25, maxpool_size=3, floor_height = -1.0, path_min_height = -0.5, path_max_height=3.0):
    device = terrain.hf.device

    terrain.hf[...] = floor_height

    # Create a bunch of twirling paths
    def generate_curvy_path(start_pos, start_vel, num_points=180, curviness=7):
        v = start_vel.clone()
        v[0] = 1.0
        angle = torch.rand(size=(1,), dtype=torch.float32, device=device) * 2.0 * torch.pi
        v = torch_util.rotate_2d_vec(v, angle).squeeze(dim=0)
        dt = 1.0 / 30.0

        pos = start_pos.clone()

        xy = torch.zeros(size=(num_points, 2), dtype=torch.float32, device=device)
        for i in range(num_points):
            xy[i] = pos.clone()
            
            pos += v * dt

            angle = torch.randn(size=(1,), dtype=torch.float32, device=device)
            v = torch_util.rotate_2d_vec(v, angle*dt*curviness).squeeze(dim=0)
            #print(pos)

        return xy

    max_point = terrain.dims * terrain.dxdy + terrain.min_point

    for path in range(num_paths):
        print("generating path", path)
        start_pos = torch.rand(size=(2,), dtype=torch.float32, device=device) 
        start_pos = start_pos * (max_point - terrain.min_point) + terrain.min_point.to(device="cpu")
        start_vel = torch.randn(size=(2,), dtype=torch.float32, device=device)
        path_xy = generate_curvy_path(start_pos, start_vel, num_points=1000)

        xy_inds = torch.round((path_xy - terrain.min_point) / terrain.dxdy).to(dtype=torch.int64)

        xy_inds = torch.clamp(xy_inds, torch.zeros_like(terrain.dims), terrain.dims - 1)

        path_height = random.random() * (path_max_height - path_min_height) + path_min_height

        terrain.hf[xy_inds[:, 0], xy_inds[:, 1]] = path_height


    # 3x3 maxpool filter
    maxpool = torch.nn.MaxPool2d(kernel_size=maxpool_size*2 + 1, stride=1, padding=maxpool_size)

    new_hf = maxpool(terrain.hf.unsqueeze(dim=0)).squeeze(dim=0)

    terrain.hf = new_hf
    return

def hf_from_motion(motion_frames: torch.Tensor,
                         min_height: float,
                         ground_height: float,
                         dx: float,
                         canon_idx = 0,
                         num_neg_x = 15,
                         num_pos_x = 15,
                         num_neg_y = 15,
                         num_pos_y = 15,
                         floor_heights = None,
                         char_model: kin_char_model.KinCharModel = None):
    # N: motion frame length
    # d: dofs of motion frame

    # motion_frames: (N, d)
    # floor_heights: (N,)

    # Creates a heightfield with a path underneath the character's root position
    # We will use this to sample heightfields from lots of motions

    # Assume motion is canonicalized

    #M = motion_frames.shape[0]
    N = motion_frames.shape[0]
    device = motion_frames.device   
    root_pos = motion_frames[..., 0:3]

    # Assume frame with canon_idx has root pos = 0
    min_x = root_pos[canon_idx, 0] - dx * num_neg_x
    min_y = root_pos[canon_idx, 1] - dx * num_neg_y
    min_point = torch.stack([min_x, min_y])#, dim=1)

    grid_dim_x = num_neg_x + num_pos_x + 1
    grid_dim_y = num_neg_y + num_pos_y + 1
    low_ind_bound = torch.tensor([0, 0], dtype=torch.int64, device=device)
    high_ind_bound = torch.tensor([grid_dim_x, grid_dim_y], dtype=torch.int64, device=device) - 1

    hf = torch.zeros(size=(grid_dim_x, grid_dim_y), dtype=torch.float32, device=device)
    hf[...] = min_height

    root_rot_quat = torch_util.exp_map_to_quat(motion_frames[..., 3:6])
    joint_rot_quat = char_model.dof_to_rot(motion_frames[..., 6:])
    body_pos, body_rot = char_model.forward_kinematics(root_pos, root_rot_quat, joint_rot_quat)

    num_bodies = char_model.get_num_joints()
    xy_inds = []
    for body_id in range(num_bodies):

        # ALTERNATIVE:
        # 1. Get all the box points
        # 2. Get all the grid indices containing the points
        # 3. store the min point height to be contained in that index
        # geom = char_model._geoms[body_id][0]
        # dims = geom._dims
        # offset = geom._offset
        # if geom._shape_type == "sphere":
        #     continue
        # box_points = geom_util.get_box_points_batch(body_pos[:, body_id], body_rot[:, body_id], dims.unsqueeze(0), offset.unsqueeze(0))

        
        #curr_xy_inds = torch.round((box_points[..., 0:2] - min_point)/dx).to(dtype=torch.int64)
        curr_xy_inds = torch.round((body_pos[..., body_id, 0:2] - min_point) / dx).to(dtype=torch.int64)
        curr_xy_inds = torch.clamp(curr_xy_inds, low_ind_bound, high_ind_bound)
        xy_inds.append(curr_xy_inds)

        if floor_heights is not None:
            hf[curr_xy_inds[:, 0], curr_xy_inds[:, 1]] = floor_heights
        else:
            hf[curr_xy_inds[:, 0], curr_xy_inds[:, 1]] = ground_height

    xy_inds = torch.cat(xy_inds, dim=0)

    # # Create a mask of shape (M, grim_dim_x, grim_dim_y) where the elements corresponding to xy_inds are True
    # mask = torch.zeros(hf.shape, dtype=torch.bool, device=hf.device)
    # mask[torch.arange(hf.shape[0]).unsqueeze(-1), xy_inds[:, :, 0], xy_inds[:, :, 1]] = True


    # # Get the floor heights

    # # Use the mask to set the desired elements in hf to their corresponding floor heights
    # hf[mask] = ground_height
    #floor_heights = None

    # Note: there is an indexing race condition. If multiple indices in a list are the same,
    # the value the tensor will be set to at that index will be the earliest value (on GPU),
    # or latest value (on CPU)
    # e.g. a = [1, 2, 3], b = [1, 1], c = [-1, -2]
    # a[b] => a = [1, -1, 3] on GPU, a = [1, -2, 3] on CPU
    # This isn't a huge deal with a smaller dx
    

    mask = torch.zeros(hf.shape, dtype=torch.bool, device=hf.device)
    mask[xy_inds[:, 0], xy_inds[:, 1]] = True
    return hf, mask

def hf_from_motion_discrete_heights(motion_frames: torch.Tensor, terrain: SubTerrain,
                                    char_model: kin_char_model.KinCharModel,
                                    heights,
                                    pen_eps=0.01):
    # Assume heights is sorted in ascending order

    # First, get the minimum height over all cells
    # Round the min height down to the nearest height

    num_bodies = char_model.get_num_joints()
    root_pos = motion_frames[..., 0:3]
    root_rot_quat = torch_util.exp_map_to_quat(motion_frames[..., 3:6])
    joint_rot_quat = char_model.dof_to_rot(motion_frames[..., 6:])
    body_pos, body_rot = char_model.forward_kinematics(root_pos, root_rot_quat, joint_rot_quat)

    min_heights = torch.ones_like(terrain.hf) * heights[-1]


    for body_id in range(num_bodies):
        grid_inds = terrain.get_grid_index(body_pos[:, body_id, 0:2])

        for i in range(grid_inds.shape[0]):
            ind_i = grid_inds[i, 0]
            ind_j = grid_inds[i, 1]
            curr_val = min_heights[ind_i, ind_j]
            min_heights[ind_i, ind_j] = min(curr_val, body_pos[i, body_id, 2])

    
    maxpool = torch.nn.MaxPool2d(kernel_size=7, stride=1, padding=3)

    min_heights = -maxpool(-min_heights.unsqueeze(0)).squeeze(0)
    print(min_heights.shape)

    # now set the terrains height
    for i in range(min_heights.shape[0]):
        for j in range(min_heights.shape[1]):
            if terrain.hf_mask[i, j]:
                terrain.hf[i, j] = heights[0]
                for k in range(len(heights)):
                    if min_heights[i, j] - 0.5 < heights[k]:
                        terrain.hf[i, j] = heights[k]
                        break
            else:
                terrain.hf[i, j] = heights[0]
    return

# we don't expect to backprop through this, so no need to make it a module
class HeightfieldAugmenter:
    def __init__(self, grid_dim_x=16, grid_dim_y=16, 
                 num_point_pairs = 3, p_scale = 1.0,
                 num_points_line = 10, dx = 0.25,
                 max_noise_chance = 0.1,
                 device="cuda:0"):

        self._num_point_pairs = num_point_pairs
        self._p_scale = p_scale
        self._num_points_line = num_points_line
        self._dx = dx
        self._max_noise_chance = max_noise_chance

        self._device = device
        self._line_points = torch.linspace(0.0, 1.0, self._num_points_line, device=device)

        self._low_ind_bound = torch.tensor([0, 0], dtype=torch.int64, device=device)
        self._high_ind_bound = torch.tensor([grid_dim_x, grid_dim_y], dtype=torch.int64, device=device) - 1

        self._maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        return
    
    def augment_hfs(self, hfs, min_point, max_point):
        # hfs: (num_hfs, grid_dim_x, grid_dim_y)
        num_hfs = hfs.shape[0]
        device = self._device

        # 1. Augment with lines between random points
        p1 = torch.rand(size=(num_hfs, self._num_point_pairs, 2), device=device)
        p1 = (p1 * (max_point - min_point).unsqueeze(dim=1) + min_point.unsqueeze(dim=1)) * self._p_scale
        p2 = torch.rand(size=(num_hfs, self._num_point_pairs, 2), device=device)
        p2 = (p2 * (max_point - min_point).unsqueeze(dim=1) + min_point.unsqueeze(dim=1)) * self._p_scale

        # draw lines between each p1 and p2 for each motion
        dp = p2 - p1
        points = torch.linspace(0.0, 1.0, self._num_points_line, device=device)
        points = points.unsqueeze(0).unsqueeze(0).unsqueeze(-1) # (M, P, num_points_line)
        points = points + dp.unsqueeze(-2)
        p_inds = torch.round((points - min_point.unsqueeze(1).unsqueeze(1)) / self._dx).to(dtype=torch.int64)
        p_inds = torch.clamp(p_inds, self._low_ind_bound, self._high_ind_bound)
        mask = torch.zeros(hfs.shape, dtype=torch.bool, device=hfs.device)
        mask[torch.arange(hfs.shape[0]).unsqueeze(-1).unsqueeze(-1), p_inds[:, :, :, 0], p_inds[:, :, :, 1]] = True
        
        hfs = hfs.clone()
        hfs[mask] = 1.0

        # Augment with maxpool (to make paths thicker)
        hfs = self._maxpool(hfs)

        # Some extra noise:
        noise_chance = torch.rand(size=(num_hfs, 1, 1), dtype=torch.float32, device=device) * self._max_noise_chance
        noise = (torch.rand_like(hfs) < noise_chance).float()

        hfs = torch.clamp_max(hfs + noise, 1.0)

        return hfs
    




def add_boxes_to_hf(hf, hf_mask, box_max_height=3.0, box_min_height=-3.0, hf_maxmin = None, num_boxes=32, box_max_len=None, box_min_len=None,
                    max_angle = 2.0 * torch.pi, min_angle = 0.0):
    # hf: height field
    # hf_mask: indices to not touch
    # box_heights: list of possible heights of boxes

    device=hf.device
    not_hf_mask = torch.logical_not(hf_mask)


    box_hf = hf.clone()

    grid_dim_x = hf.shape[0]
    grid_dim_y = hf.shape[1]
    if box_max_len is None:
        box_max_len = min(grid_dim_x // 4, grid_dim_y //4)
    if box_min_len is None:
        box_min_len = 1

    # Good method
    x_points = torch.linspace(0, hf.shape[0] - 1, hf.shape[0], dtype=torch.int64, device=device)
    y_points = torch.linspace(0, hf.shape[1] - 1, hf.shape[1], dtype=torch.int64, device=device)
    x, y = torch.meshgrid(x_points, y_points, indexing='ij')
    xy = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], dim=-1).to(dtype=torch.float32)
    for b in range(num_boxes):

        box_center = torch.rand(size=(2,), dtype=torch.float32, device=device) * torch.tensor(hf.shape, dtype=torch.float32, device=device)
        box_len = torch.rand(size=(2,), dtype=torch.float32, device=device) * (box_max_len - box_min_len) + box_min_len
        angle = random.random() * (max_angle - min_angle) + min_angle
        angle = angle * torch.ones_like(xy[:, :, 0])

        #grid_center_xy = (torch.tensor(hf.shape, dtype=torch.float32, device=device) - 1.0) / 2.0
        rot_xy = torch_util.rotate_2d_vec(xy - box_center, angle) + box_center
        # translate grid points so that box's start xy are at the origin
        trans_xy = rot_xy# + box_center

        in_box_x_1 = trans_xy[..., 0] < box_center[0] + box_len[0] / 2
        in_box_x_2 = trans_xy[..., 0] > box_center[0] - box_len[0] / 2

        in_box_y_1 = trans_xy[..., 1] < box_center[1] + box_len[1] / 2
        in_box_y_2 = trans_xy[..., 1] > box_center[1] - box_len[1] / 2

        in_box_x = torch.logical_and(in_box_x_1, in_box_x_2)
        in_box_y = torch.logical_and(in_box_y_1, in_box_y_2)
        in_box = torch.logical_and(in_box_x, in_box_y)
        # if grid point is inside box

        #h = random.random() * (max_height - min_height) + min_height

        #h_ind = random.randint(0, len(box_heights)-1)
        #h = box_heights[h_ind]
        h = random.random() * (box_max_height - box_min_height) + box_min_height
        h_th = torch.ones_like(x, dtype=torch.float32, device=device) * h

        #print(h_th.shape, box_hf[x, y].shape, in_box.shape)
        box_hf[x, y] = torch.where(in_box, h_th, box_hf[x, y])

    if hf_maxmin is not None:
        box_hf[not_hf_mask] = torch.clamp(box_hf[not_hf_mask], hf_maxmin[..., 1][not_hf_mask], hf_maxmin[..., 0][not_hf_mask])
    
    hf[not_hf_mask] = box_hf[not_hf_mask]

    return

def add_boxes_to_hf2(hf, box_max_height=3.0, box_min_height=-3.0, hf_maxmin = None, 
                     num_boxes=32, box_max_len=None, box_min_len=None,
                     max_angle = 2.0 * torch.pi, min_angle = 0.0):
    
    device=hf.device
    grid_dim_x = hf.shape[0]
    grid_dim_y = hf.shape[1]
    if box_max_len is None:
        box_max_len = min(grid_dim_x // 4, grid_dim_y //4)
    if box_min_len is None:
        box_min_len = 1

    # Good method
    x_points = torch.linspace(0, hf.shape[0] - 1, hf.shape[0], dtype=torch.int64, device=device)
    y_points = torch.linspace(0, hf.shape[1] - 1, hf.shape[1], dtype=torch.int64, device=device)
    x, y = torch.meshgrid(x_points, y_points, indexing='ij')
    xy = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], dim=-1).to(dtype=torch.float32)
    for b in range(num_boxes):

        box_center = torch.rand(size=(2,), dtype=torch.float32, device=device) * torch.tensor(hf.shape, dtype=torch.float32, device=device)
        box_len = torch.rand(size=(2,), dtype=torch.float32, device=device) * (box_max_len - box_min_len) + box_min_len
        angle = random.random() * (max_angle - min_angle) + min_angle
        angle = angle * torch.ones_like(xy[:, :, 0])

        #grid_center_xy = (torch.tensor(hf.shape, dtype=torch.float32, device=device) - 1.0) / 2.0
        rot_xy = torch_util.rotate_2d_vec(xy - box_center, angle) + box_center
        # translate grid points so that box's start xy are at the origin
        trans_xy = rot_xy# + box_center

        in_box_x_1 = trans_xy[..., 0] < box_center[0] + box_len[0] / 2
        in_box_x_2 = trans_xy[..., 0] > box_center[0] - box_len[0] / 2

        in_box_y_1 = trans_xy[..., 1] < box_center[1] + box_len[1] / 2
        in_box_y_2 = trans_xy[..., 1] > box_center[1] - box_len[1] / 2

        in_box_x = torch.logical_and(in_box_x_1, in_box_x_2)
        in_box_y = torch.logical_and(in_box_y_1, in_box_y_2)
        in_box = torch.logical_and(in_box_x, in_box_y)
        # if grid point is inside box

        #h = random.random() * (max_height - min_height) + min_height

        #h_ind = random.randint(0, len(box_heights)-1)
        #h = box_heights[h_ind]
        h = random.random() * (box_max_height - box_min_height) + box_min_height
        h_th = torch.ones_like(x, dtype=torch.float32, device=device) * h

        #print(h_th.shape, box_hf[x, y].shape, in_box.shape)
        hf[x, y] = torch.where(in_box, h_th, hf[x, y])

    if hf_maxmin is not None:
        hf[...] = torch.clamp(hf[...], hf_maxmin[..., 1], hf_maxmin[..., 0])

    return

def add_aabox_at_index(terrain: SubTerrain, input_grid_ind: torch.Tensor, height, box_size):
    for i in range(box_size*2 + 1):
        for j in range(box_size*2 + 1):
            node = input_grid_ind.clone()
            node = node.to(dtype=torch.int64)
            node[0] += i - box_size
            node[1] += j - box_size
            grid_ind = terrain.get_inbounds_grid_index(node)
            terrain.hf[grid_ind[0], grid_ind[1]] = height
    return

def add_boxes_to_hf_at_xy_points(box_centers, hf, min_h, max_h,
                                 min_len, max_len,
                                 min_angle, max_angle):
    device=hf.device

    # Good method
    x_points = torch.linspace(0, hf.shape[0] - 1, hf.shape[0], dtype=torch.int64, device=device)
    y_points = torch.linspace(0, hf.shape[1] - 1, hf.shape[1], dtype=torch.int64, device=device)
    x, y = torch.meshgrid(x_points, y_points, indexing='ij')
    xy = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], dim=-1).to(dtype=torch.float32)
    num_boxes = box_centers.shape[0]
    for b in range(num_boxes):

        box_center = box_centers[b]
        box_len = torch.rand(size=(2,), dtype=torch.float32, device=device) * (max_len - min_len) + min_len
        angle = random.random() * (max_angle - min_angle) + min_angle
        angle = angle * torch.ones_like(xy[:, :, 0])

        #grid_center_xy = (torch.tensor(hf.shape, dtype=torch.float32, device=device) - 1.0) / 2.0
        # translate grid points so that box's start xy are at the origin
        trans_xy = torch_util.rotate_2d_vec(xy - box_center, angle) + box_center
        
        in_box_x_1 = trans_xy[..., 0] < box_center[0] + box_len[0] / 2
        in_box_x_2 = trans_xy[..., 0] > box_center[0] - box_len[0] / 2

        in_box_y_1 = trans_xy[..., 1] < box_center[1] + box_len[1] / 2
        in_box_y_2 = trans_xy[..., 1] > box_center[1] - box_len[1] / 2

        in_box_x = torch.logical_and(in_box_x_1, in_box_x_2)
        in_box_y = torch.logical_and(in_box_y_1, in_box_y_2)
        in_box = torch.logical_and(in_box_x, in_box_y)
        # if grid point is inside box

        h = random.random() * (max_h - min_h) + min_h
        h_th = torch.ones_like(x, dtype=torch.float32, device=device) * h

        #print(h_th.shape, box_hf[x, y].shape, in_box.shape)
        hf[x, y] = torch.where(in_box, h_th, hf[x, y])

    return

def draw_box(hf: torch.Tensor, min_point: torch.Tensor, dxdy: torch.Tensor,
             box_center: torch.Tensor, box_w, box_l, angle: torch.Tensor,
             height: float):
    # box_center, box_len in real space, not grid space
    device = hf.device
    grid_dim_x = hf.shape[0]
    grid_dim_y = hf.shape[1]

    # Good method
    x_points = torch.linspace(0, grid_dim_x - 1, grid_dim_x, dtype=torch.int64, device=device) * dxdy[0] + min_point[0]
    y_points = torch.linspace(0, grid_dim_y - 1, grid_dim_y, dtype=torch.int64, device=device) * dxdy[1] + min_point[1]
    x, y = torch.meshgrid(x_points, y_points, indexing='ij')
    xy = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], dim=-1).to(dtype=torch.float32)

    rot_xy = torch_util.rotate_2d_vec(xy - box_center, angle) + box_center
    # translate grid points so that box's start xy are at the origin
    trans_xy = rot_xy# + box_center

    in_box_x_1 = trans_xy[..., 0] < box_center[0] + box_w / 2
    in_box_x_2 = trans_xy[..., 0] > box_center[0] - box_w / 2

    in_box_y_1 = trans_xy[..., 1] < box_center[1] + box_l / 2
    in_box_y_2 = trans_xy[..., 1] > box_center[1] - box_l / 2

    in_box_x = torch.logical_and(in_box_x_1, in_box_x_2)
    in_box_y = torch.logical_and(in_box_y_1, in_box_y_2)
    in_box = torch.logical_and(in_box_x, in_box_y)
    
    hf[in_box] = height
    return

def add_stairs_to_hf(terrain: SubTerrain, min_stair_start_height=-3.0, max_stair_start_height=1.0, 
                     min_step_height = 0.15,
                     max_step_height = 0.25, num_stairs=1,
                     min_stair_thickness = 2.5,
                     max_stair_thickness = 8.0):

    for i in range(num_stairs):

        stair_start = torch.rand(size=[2], dtype=torch.float32, device=terrain.hf.device)
        stair_start = stair_start * (terrain.get_max_point() - terrain.min_point) + terrain.min_point
        stair_end = torch.rand(size=[2], dtype=torch.float32, device=terrain.hf.device)
        stair_end = stair_end * (terrain.get_max_point() - terrain.min_point) + terrain.min_point

        box_center = (stair_start + stair_end) / 2.0
        stair_diff = stair_end - stair_start
        stair_width = torch.linalg.norm(stair_diff).item()
        angle = -torch_util.heading_angle_from_xy(stair_diff[0], stair_diff[1])

        step_w = terrain.dxdy[0].item()
        num_steps = int(np.ceil(stair_width / step_w))
        stair_dxdy = stair_diff / num_steps

        start_height = np.random.random() * (max_stair_start_height - min_stair_start_height) + min_stair_start_height

        step_height = np.random.random() * (max_step_height - min_step_height) + min_step_height
        stair_thickness = np.random.random() * (max_stair_thickness - min_stair_thickness) + min_stair_thickness

        for j in range(num_steps):
            box_center = stair_start + j * stair_dxdy
            box_w = step_w
            box_l = stair_thickness
            height = start_height + j * step_height
            draw_box(hf = terrain.hf,
                    min_point = terrain.min_point,
                    dxdy = terrain.dxdy, 
                    box_center = box_center,
                    box_w = box_w,
                    box_l = box_l,
                    angle = angle,
                    height = height)
        
    return

def get_line_indices(x0, y0, x1, y1):
    indices = []

    # Calculate the differences and step direction
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1

    # Initial error value
    err = dx - dy

    while True:
        # Append the current cell to the indices list
        indices.append([x0, y0])
        
        # Check if we've reached the end point
        if x0 == x1 and y0 == y1:
            break

        # Bresenham's algorithm step
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return indices

def draw_line(terrain: SubTerrain, start_node: torch.Tensor, end_node: torch.Tensor, height: float):
    # TODO use a line drawing algorithm: https://en.wikipedia.org/wiki/Line_drawing_algorithm
    # optionally return indices of the 

    start_node = terrain.get_inbounds_grid_index(start_node)
    end_node = terrain.get_inbounds_grid_index(end_node)

    line_indices = get_line_indices(start_node[0].item(), start_node[1].item(), end_node[0].item(), end_node[1].item())

    line_indices = torch.tensor(line_indices, dtype=torch.int64, device=terrain.hf.device)

    line_indices = terrain.get_inbounds_grid_index(line_indices)

    terrain.hf[line_indices[:, 0], line_indices[:, 1]] = height
    return

def random_heightfield(terrain: SubTerrain):

    max_height = 1.0
    
    terrain.hf = np.random.rand(terrain.dims[0], terrain.dims[1]) * max_height
    return

def convert_heightfield_to_voxelized_trimesh(hf, min_x, min_y, dx=0.1, padding=None):

    if isinstance(hf, torch.Tensor):
        hf = hf.cpu()

    # make a quad for each index of the hf

    total_num_verts = hf.shape[0] * hf.shape[1] * 4
    total_num_flat_tris = hf.shape[0] * hf.shape[1] * 2
    total_num_vert_x_tris = (hf.shape[0]-1) * hf.shape[1] * 2
    total_num_vert_y_tris = hf.shape[0] * (hf.shape[1]-1) * 2
    total_num_tris = total_num_flat_tris + total_num_vert_x_tris + total_num_vert_y_tris

    if padding is not None and padding > 0.0:
        total_num_verts += 8
        total_num_tris += 8

    vertices = np.zeros(shape=(total_num_verts, 3), dtype=np.float32)
    tris = np.zeros(shape=(total_num_tris, 3), dtype=np.uint32)

    # Idea: make a graph
    for i in range(hf.shape[0]):
        for j in range(hf.shape[1]):
            curr_ind = i*hf.shape[1] + j
            x = dx * i + min_x
            y = dx * j + min_y

            z = hf[i, j]#.cpu()
            # create two quads
            p1 = np.array([x - dx/2, y - dx/2, z])
            p2 = np.array([x - dx/2, y + dx/2, z])
            p3 = np.array([x + dx/2, y + dx/2, z])
            p4 = np.array([x + dx/2, y - dx/2, z])

            vertices[curr_ind*4+0] = p1
            vertices[curr_ind*4+1] = p2
            vertices[curr_ind*4+2] = p3
            vertices[curr_ind*4+3] = p4

            tris[curr_ind*2 + 0] = [curr_ind*4 + 0, curr_ind*4 + 2, curr_ind*4 + 1]
            tris[curr_ind*2 + 1] = [curr_ind*4 + 0, curr_ind*4 + 3, curr_ind*4 + 2]

    # now connect the sides
    for i in range(hf.shape[0] - 1):
        for j in range(hf.shape[1]):
            curr_ind = i*(hf.shape[1]) + j

            vert_ind = i*hf.shape[1] + j
            next_vert_ind = (i+1)*hf.shape[1] + j

            x = dx * i + min_x
            y = dx * j + min_y
            z = hf[i, j]#.cpu()
            
            #next_z = hf[i+1, j]

            v1 = vert_ind*4+3
            v2 = vert_ind*4+2
            v3 = next_vert_ind*4+1
            v4 = next_vert_ind*4+0

            #if z
            tris[curr_ind*2 + total_num_flat_tris + 0] = [v1, v3, v2]
            tris[curr_ind*2 + total_num_flat_tris + 1] = [v1, v4, v3]

    # now connect the sides
    for i in range(hf.shape[0]):
        for j in range(hf.shape[1] - 1):
            curr_ind = i*(hf.shape[1]-1) + j

            vert_ind = i*hf.shape[1] + j
            next_vert_ind = (i+1)*hf.shape[1] + j

            x = dx * i + min_x
            y = dx * j + min_y
            z = hf[i, j]#.cpu()


            next_vert_ind = i*hf.shape[1] + j+1
            v1 = vert_ind*4 + 1
            v2 = vert_ind*4 + 2
            v3 = next_vert_ind*4 + 3
            v4 = next_vert_ind*4 + 0

            tris[curr_ind*2 + total_num_flat_tris + total_num_vert_x_tris + 0] = [v1, v2, v3]
            tris[curr_ind*2 + total_num_flat_tris + total_num_vert_x_tris + 1] = [v1, v3, v4]
    
    if padding is not None and padding > 0.0:
        assert isinstance(padding, float)
        # Padding vertices
        # v6 ------------------ v7
        # |                     |
        # |  v2 ----------- v3  |
        # |   |             |   |
        # |   |             |   |
        # |   |             |   |
        # |   |             |   |
        # |  v0 ----------- v1  |
        # |                     |
        # v4 ------------------ v5

        max_x = min_x + dx * (hf.shape[0]-1)
        max_y = min_y + dx * (hf.shape[1]-1)

        if isinstance(hf, torch.Tensor):
            min_z = torch.min(hf).item()
        else:
            min_z = np.min(hf)

        z0 = min_z#hf[0, 0]
        p0 = np.array([min_x - dx/2, min_y - dx/2, z0])
        p4 = p0 + np.array([-padding, -padding, 0.0])

        z1 = min_z#hf[-1, 0]
        p1 = np.array([max_x + dx/2, min_y - dx/2, z1])
        p5 = p1 + np.array([+padding, -padding, 0.0])

        z2 = min_z#hf[0, -1]
        p2 = np.array([min_x - dx/2, max_y + dx/2, z2])
        p6 = p2 + np.array([-padding, +padding, 0.0])

        z3 = min_z#hf[-1, -1]
        p3 = np.array([max_x + dx/2, max_y + dx/2, z3])
        p7 = p3 + np.array([+padding, +padding, 0.0])

        vertices[-8] = p0
        vertices[-7] = p1
        vertices[-6] = p2
        vertices[-5] = p3
        vertices[-4] = p4
        vertices[-3] = p5
        vertices[-2] = p6
        vertices[-1] = p7

        v0 = vertices.shape[0] - 8
        v1 = vertices.shape[0] - 7
        v2 = vertices.shape[0] - 6
        v3 = vertices.shape[0] - 5
        v4 = vertices.shape[0] - 4
        v5 = vertices.shape[0] - 3
        v6 = vertices.shape[0] - 2
        v7 = vertices.shape[0] - 1

        tris[-8] = [v0, v4, v5]
        tris[-7] = [v0, v5, v1]
        tris[-6] = [v1, v5, v7]
        tris[-5] = [v1, v7, v3]
        tris[-4] = [v3, v7, v6]
        tris[-3] = [v3, v6, v2]
        tris[-2] = [v2, v6, v4]
        tris[-1] = [v2, v4, v0]

    return vertices, tris

def convert_hf_mask_to_flat_voxels(hf_mask, hf, min_x, min_y, dx, voxel_w_scale=0.9):
    # A function mainly for visualization
    # If hf_mask[i, j] = True, the create a voxel at (x_i, y_j, hf[i, j])

    voxel_w = voxel_w_scale * dx
    voxel_h = 0.01

    assert hf_mask.shape == hf.shape

    if isinstance(hf, torch.Tensor):
        hf = hf.cpu()

    if isinstance(hf_mask, torch.Tensor):
        hf_mask = hf_mask.cpu()
        num_voxels = torch.sum(hf_mask.to(dtype=torch.int64)).item()
    else:
        num_voxels = np.sum(hf_mask.astype(np.int64))

    num_verts = num_voxels * 8
    num_tris = num_voxels * 10 # roof and sides of box
    vertices = np.zeros(shape=(num_verts, 3), dtype=np.float32)
    tris = np.zeros(shape=(num_tris, 3), dtype=np.uint32)

    curr_vert_ind = 0
    curr_tri_ind = 0
    for i in range(hf.shape[0]):
        for j in range(hf.shape[1]):
            if hf_mask[i, j] == True:
                x = i * dx + min_x
                y = j * dx + min_y
                z = hf[i, j]#.cpu()

                p1 = [x - voxel_w / 2.0, y - voxel_w / 2.0, z]
                p2 = [x - voxel_w / 2.0, y + voxel_w / 2.0, z]
                p3 = [x + voxel_w / 2.0, y + voxel_w / 2.0, z]
                p4 = [x + voxel_w / 2.0, y - voxel_w / 2.0, z]
                p5 = [x - voxel_w / 2.0, y - voxel_w / 2.0, z + voxel_h]
                p6 = [x - voxel_w / 2.0, y + voxel_w / 2.0, z + voxel_h]
                p7 = [x + voxel_w / 2.0, y + voxel_w / 2.0, z + voxel_h]
                p8 = [x + voxel_w / 2.0, y - voxel_w / 2.0, z + voxel_h]

                v1 = curr_vert_ind + 0
                v2 = curr_vert_ind + 1
                v3 = curr_vert_ind + 2
                v4 = curr_vert_ind + 3
                v5 = curr_vert_ind + 4
                v6 = curr_vert_ind + 5
                v7 = curr_vert_ind + 6
                v8 = curr_vert_ind + 7

                vertices[v1] = p1
                vertices[v2] = p2
                vertices[v3] = p3
                vertices[v4] = p4
                vertices[v5] = p5
                vertices[v6] = p6
                vertices[v7] = p7
                vertices[v8] = p8

                tris[curr_tri_ind + 0] = [v1, v5, v6] # -x side
                tris[curr_tri_ind + 1] = [v1, v6, v2] # 
                tris[curr_tri_ind + 2] = [v2, v6, v7] # +y side
                tris[curr_tri_ind + 3] = [v2, v7, v3] #
                tris[curr_tri_ind + 4] = [v3, v7, v8] # +x side
                tris[curr_tri_ind + 5] = [v3, v8, v4] #
                tris[curr_tri_ind + 6] = [v4, v8, v5] # -y side
                tris[curr_tri_ind + 7] = [v4, v5, v1] # 
                tris[curr_tri_ind + 8] = [v5, v7, v6] # roof
                tris[curr_tri_ind + 9] = [v5, v8, v7] #
 

                curr_vert_ind += 8
                curr_tri_ind += 10

    return vertices, tris

def get_local_hf_from_terrain(xy_points, terrain: SubTerrain):
    # this works for terrain with only vertical slopes

    # min_point = terrain.min_point

    # ij_indices = (xy_points - min_point.unsqueeze(dim=0)) / terrain.dxdy.unsqueeze(dim=0)

    # ij_indices = torch.round(ij_indices).to(dtype=torch.int64)

    # ij_indices[:, 0] = torch.clamp(ij_indices[:, 0], 0, terrain.dims[0]-1)
    # ij_indices[:, 1] = torch.clamp(ij_indices[:, 1], 0, terrain.dims[1]-1)

    grid_inds = terrain.get_grid_index(xy_points)

    # using info from terrain, get the indices that match
    hf = terrain.hf[grid_inds[..., 0], grid_inds[..., 1]]

    return hf


def merge_hfs(hf1, hf_mask, hf_maxmin, hf2):

    not_hf_mask = torch.logical_not(hf_mask)

    ret_hf = hf1.clone()#hf2.clone()    

    ret_hf[not_hf_mask] = hf2[not_hf_mask]

    if hf_maxmin is not None:
        ret_hf[not_hf_mask] = torch.clamp(ret_hf[not_hf_mask], hf_maxmin[..., 1][not_hf_mask], hf_maxmin[..., 0][not_hf_mask])
    

    #
    # ret_hf[not_hf_mask] = hf1[not_hf_mask]


    return ret_hf

def char_terrain_intersection(body_pos, body_rot, char_model, terrain: SubTerrain):
    """
    This function computes the intersection (and potential vector to separate intersection)
    of the character and the terrain
    """

    # offset -> rotation -> body pos translation
    # First get tensors of OBBs for each body geometry

    char_obb_pos = []
    char_obb_dims = []
    char_obb_rot = []

    # Get the obbs for every geometry
    for b in range(char_model.get_num_joints()):
        geoms = char_model.get_geoms(b)
        for geom in geoms:
            if geom._shape_type == kin_char_model.GeomType.SPHERE: # skip spheres for now
                continue
            if geom._shape_type == kin_char_model.GeomType.CAPSULE: # skip capsules for now
                continue
            offset = geom._offset
            dims = geom._dims

            pos = body_pos[b] + torch_util.quat_rotate(body_rot[b], offset)
            rot = body_rot[b]

            char_obb_pos.append(pos)
            char_obb_dims.append(dims)
            char_obb_rot.append(rot)

            # TODO: visualize the OBBs as transparent OBBs

    char_obb_pos = torch.stack(char_obb_pos)
    char_obb_dims = torch.stack(char_obb_dims)
    char_obb_rot = torch.stack(char_obb_rot)

    #print(char_obb_pos.shape)

    # Get the indices of the surrounding box of the character
    bounding_box_size = 1.0
    dx = terrain.dxdy[0].item()
    num_pos_x = int(round(bounding_box_size / dx))
    num_neg_x = num_pos_x
    num_pos_y = num_pos_x
    num_neg_y = num_pos_x

    #g_terrain.to_torch('cpu')

    lower_bound_eps = 0.5
    terrain_min_height = torch.min(terrain.hf) - lower_bound_eps
    #print("terrain min height:", terrain_min_height)

    hf_obb_centers = geom_util.get_xy_grid_points(body_pos[0, 0:2], dx, dx, num_neg_x, num_pos_x, num_neg_y, num_pos_y).view(-1, 2)
    hf_obb_heights = get_local_hf_from_terrain(hf_obb_centers, terrain)

    
    hf_obb_z_dims = (hf_obb_heights - terrain_min_height)/2.0

    hf_obb_pos = torch.cat([hf_obb_centers, (hf_obb_heights - hf_obb_z_dims).unsqueeze(-1)], dim=-1)
    num_hf_points = hf_obb_pos.shape[0]

    hf_obb_dims = torch.cat([(terrain.dxdy/2).unsqueeze(0).expand(num_hf_points, -1), hf_obb_z_dims.unsqueeze(-1)], dim=-1)
    hf_obb_rot = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32).unsqueeze(0).expand(num_hf_points, -1)

    info = geom_util.obb_SAT(char_obb_pos,
                            char_obb_rot,
                        char_obb_dims,
                        hf_obb_pos,
                        hf_obb_rot,
                        hf_obb_dims,
                        ret_debug_info=True)

    sep_val = info['sep_val']
    separating = torch.all(torch.any(sep_val > 0, dim=-1)).item()
    info["intersecting"] = not separating
    return info

# TODO
def extract_potential_intersecting_hf_boxes(root_pos: torch.Tensor, padding: float, terrain: SubTerrain):
    # root_pos: (N, 3) tensor, where N is the number of motion frames

    device = root_pos.device

    # first create a large bounding box around the root_pos trajectory
    max_x = torch.max(root_pos[..., 0])
    min_x = torch.min(root_pos[..., 0])
    max_y = torch.max(root_pos[..., 1])
    min_y = torch.min(root_pos[..., 1])

    max_point = torch.tensor([max_x, max_y], device=device) + padding
    min_point = torch.tensor([min_x, min_y], device=device) - padding

    # snapping points to grid positions
    min_point = torch.round((min_point - terrain.min_point) / terrain.dxdy) * terrain.dxdy + terrain.min_point
    max_point = torch.round((max_point - terrain.min_point) / terrain.dxdy) * terrain.dxdy + terrain.min_point

    xy_points = geom_util.get_xy_grid_points_2(min_point, max_point, terrain.dxdy)
    
    z_points = get_local_hf_from_terrain(xy_points, terrain)

    # TODO
    # create a 2D bounding box around the character's root position,
    # and extract the box positions and dimensions from the terrain



    lower_bound_eps = 0.5
    terrain_min_height = torch.min(z_points) - lower_bound_eps

    hf_obb_centers = xy_points.view(-1, 2)
    hf_obb_heights = z_points.view(-1)
    hf_obb_z_dims = (hf_obb_heights - terrain_min_height)/2.0

    hf_obb_pos = torch.cat([hf_obb_centers, (hf_obb_heights - hf_obb_z_dims).unsqueeze(-1)], dim=-1)
    num_hf_points = hf_obb_pos.shape[0]

    hf_obb_dims = torch.cat([(terrain.dxdy/2).unsqueeze(0).expand(num_hf_points, -1), hf_obb_z_dims.unsqueeze(-1)], dim=-1)
    hf_obb_rot = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32, device=device).unsqueeze(0).expand(num_hf_points, -1)

    

    return hf_obb_pos, hf_obb_rot, hf_obb_dims

def obb_terrain_intersection(obb_pos: torch.Tensor, 
                             obb_rot: torch.Tensor, 
                             obb_dims: torch.Tensor,
                             root_pos: torch.Tensor,
                             terrain: SubTerrain):
    
    hf_obb_pos, hf_obb_rot, hf_obb_dims = extract_potential_intersecting_hf_boxes(root_pos, 2.0, terrain)

    # sanity checks
    assert obb_pos.shape == obb_dims.shape
    assert obb_rot.shape[:-1] == obb_pos.shape[:-1]

    info = geom_util.obb_SAT(obb_pos,
                            obb_rot,
                        obb_dims,
                        hf_obb_pos,
                        hf_obb_rot,
                        hf_obb_dims,
                        ret_debug_info=True)

    sep_val = info['sep_val'] # (num frames * num bodies, num grid cells, 15)

    num_frames = root_pos.shape[0]
    sep_val = sep_val.view(2, num_frames, hf_obb_pos.shape[0], 15)
    
    pen_dist, _ = torch.max(sep_val, dim=-1)
    pen_dist, _ = torch.min(pen_dist, dim=-1)

    #separating = torch.all(torch.any(sep_val > 0, dim=-1)).item()
    info["pen_dist"] = pen_dist

    return info

def motion_terrain_intersection(root_pos: torch.Tensor, 
                                root_rot: torch.Tensor, 
                                joint_rot: torch.Tensor, 
                                char_model: kin_char_model.KinCharModel, 
                                terrain: SubTerrain):

    # root_pos = motion_frames[..., 0:3]
    # root_rot = torch_util.exp_map_to_quat(motion_frames[..., 3:6])
    # joint_rot = char_model.dof_to_rot(motion_frames[..., 6:34])

    char_obb_pos, char_obb_rot, char_obb_dims, body_ids = geom_util.get_obbs_from_motion(root_pos, root_rot, joint_rot, char_model)

    info = obb_terrain_intersection(char_obb_pos, char_obb_rot, char_obb_dims, root_pos, terrain)
    info["body_ids"] = body_ids
    return info

def point_hf_penetration_distance(points: torch.Tensor, hf: torch.Tensor, min_xy: torch.Tensor, dxdy: torch.Tensor):
    # return the distance of each point and the hf
    # M: number of points
    # points: (M, 3)
    # hf: (hf dim x, hf dim y)
    # min_xy: (2)
    # dxdy: (2)

    # first get index the point is in
    inds = (points[..., 0:2] - min_xy) / dxdy
    inds = torch.round(inds).to(dtype=torch.int64)
    max_inds = torch.tensor(hf.shape, dtype=torch.int64, device=points.device).unsqueeze(0) - 1
    inds = torch.clamp(inds, min = torch.zeros_like(inds), max = max_inds)

    hf_z_at_point = hf[inds[..., 0], inds[..., 1]]

    
    return points[..., 2] - hf_z_at_point # negative means point is below heightfield

def point_hf_penetration_distance_batch(points: torch.Tensor, hf: torch.Tensor, min_xy: torch.Tensor, dxdy: torch.Tensor):
    # return the distance of each point and the hf
    # M: number of points
    # points: (N, M, 3)
    # hf: (N, hf dim x, hf dim y)
    # min_xy: (2)
    # dxdy: (2)

    # first get index the point is in
    inds = (points[..., 0:2] - min_xy) / dxdy
    inds = torch.round(inds).to(dtype=torch.int64) # expected shape: (N, M, 2)
    max_inds = torch.tensor(hf.shape[1:], dtype=torch.int64, device=points.device).unsqueeze(0).unsqueeze(0) - 1
    inds = torch.clamp(inds, min = torch.zeros_like(inds), max = max_inds)

    # n = hf.shape[0]
    # hf_z_at_point2 = []
    # for i in range(n):
    #     curr_hf_z_at_point = hf[i, inds[i, :, 0], inds[i, :, 1]]
    #     hf_z_at_point2.append(curr_hf_z_at_point)
    # hf_z_at_point2 = torch.stack(hf_z_at_point2)

    # Use advanced indexing to gather the desired values from `hf`
    # inds has shape (N, M, 2), we can split it into separate components for indexing
    x_indices = inds[:, :, 0]  # indices along the first dimension
    y_indices = inds[:, :, 1]  # indices along the second dimension

    # Use the indices to gather the corresponding heights from `hf`
    # This is equivalent to what was done in the loop, but with advanced indexing
    hf_z_at_point = hf.gather(1, x_indices.unsqueeze(-1).expand(-1, -1, hf.shape[2]))  # Expand to have the same dimensionality
    hf_z_at_point = hf_z_at_point.gather(2, y_indices.unsqueeze(-1))  # Gather the correct y values

    # Resulting `hf_z_at_point` has shape (N, M, 1), so we squeeze the last dimension
    hf_z_at_point = hf_z_at_point.squeeze(-1)
    
    return points[..., 2] - hf_z_at_point # negative means point is below heightfield

def maxpool_hf(hf, hf_maxmin, maxpool_size):
    # assumes 2D hf, no batch_size dim, no channel dim
    maxpool_nn = torch.nn.MaxPool2d(kernel_size=maxpool_size*2 + 1, stride=1, padding=maxpool_size)
    new_hf = maxpool_nn(hf.unsqueeze(dim=0)).squeeze(dim=0)
    if hf_maxmin is not None:
        new_hf = torch.clamp(new_hf, hf_maxmin[..., 1], hf_maxmin[..., 0])
    hf[...] = new_hf[...]
    return

def maxpool_hf_1d_x(hf, hf_maxmin, maxpool_size):
    # assumes 2D hf, no batch_size dim, no channel dim
    maxpool_nn = torch.nn.MaxPool1d(kernel_size=maxpool_size*2 + 1, stride=1, padding=maxpool_size)
    new_hf = maxpool_nn(hf.unsqueeze(dim=0).permute(0, 2, 1))
    new_hf = new_hf.permute(0, 2, 1).squeeze(dim=0)
    if hf_maxmin is not None:
        new_hf = torch.clamp(new_hf, hf_maxmin[..., 1], hf_maxmin[..., 0])
    hf[...] = new_hf[...]
    return

def maxpool_hf_1d_y(hf, hf_maxmin, maxpool_size):
    # assumes 2D hf, no batch_size dim, no channel dim
    maxpool_nn = torch.nn.MaxPool1d(kernel_size=maxpool_size*2 + 1, stride=1, padding=maxpool_size)
    new_hf = maxpool_nn(hf.unsqueeze(dim=0)).squeeze(dim=0)
    if hf_maxmin is not None:
        new_hf = torch.clamp(new_hf, hf_maxmin[..., 1], hf_maxmin[..., 0])
    hf[...] = new_hf[...]
    return

def minpool_hf(hf, hf_maxmin, maxpool_size):
    maxpool_nn = torch.nn.MaxPool2d(kernel_size=maxpool_size*2 + 1, stride=1, padding=maxpool_size)
    new_hf = maxpool_nn(-hf.unsqueeze(dim=0)).squeeze(dim=0)
    if hf_maxmin is not None:
        new_hf = torch.clamp(new_hf, -hf_maxmin[..., 0], -hf_maxmin[..., 1])
    hf[...] = -new_hf[...]
    return

def downsample_terrain(terrain: SubTerrain):

    old_x_dim = terrain.dims[0]
    old_y_dim = terrain.dims[1]
    x_dim = old_x_dim // 2
    y_dim = old_y_dim // 2
    dx = terrain.dxdy[0] * 2
    dy = terrain.dxdy[1] * 2
    min_x = terrain.min_point[0]
    min_y = terrain.min_point[1]


    ret_terrain = SubTerrain(terrain.terrain_name,
                             x_dim,
                             y_dim,
                             dx,
                             dy,
                             min_x,
                             min_y,
                             terrain.hf.device)

    # for each 2x2 window in the old terrain, fill the 1x1 window in the new terrain.
    # Keep track of all heights in the 2x2 window, and set the new height to the
    # height with most counts, defaulting to highest height

    # or just maxpool
    for i in range(x_dim):
        for j in range(y_dim):
            max_height = -999.0
            curr_mask = False
            for ii in range(2):
                for jj in range(2):
                    curr_height = terrain.hf[i*2 + ii, j*2 + jj]
                    if curr_height > max_height:
                        max_height = curr_height

                    curr_mask = curr_mask or terrain.hf_mask[i*2 + ii, j*2 + jj]
            ret_terrain.hf[i, j] = max_height
            ret_terrain.hf_mask[i, j] = curr_mask



    return ret_terrain

def slice_terrain_around_motion(motion_frames: torch.Tensor, terrain: SubTerrain, padding=1.0, localize=True):
    # motion_frames just needs to contain xy positions
    # motion_frames can be batched, not terrain

    device = motion_frames.device
    motion_min_x = torch.min(motion_frames[..., 0]).item()
    motion_min_y = torch.min(motion_frames[..., 1]).item()
    motion_max_x = torch.max(motion_frames[..., 0]).item()
    motion_max_y = torch.max(motion_frames[..., 1]).item()

    motion_min_point = torch.tensor([motion_min_x, motion_min_y], dtype=torch.float32, device=device) - padding
    motion_max_point = torch.tensor([motion_max_x, motion_max_y], dtype=torch.float32, device=device) + padding
    # min_grid_ind = terrain.get_grid_index(motion_min_point)
    # max_grid_ind = terrain.get_grid_index(motion_max_point)
    # min_grid_point = terrain.get_point(min_grid_ind)

    min_grid_point = terrain.round_point_to_grid_point(motion_min_point)
    max_grid_point = terrain.round_point_to_grid_point(motion_max_point)

   
    

    # TODO: instead of using literal slices, which won't work well when the motion goes out of bounds,
    # use xy_points

    x_points = torch.arange(min_grid_point[0], max_grid_point[0] + terrain.dxdy[0], step=terrain.dxdy[0], device=device)
    y_points = torch.arange(min_grid_point[1], max_grid_point[1] + terrain.dxdy[1], step=terrain.dxdy[1], device=device)

    x, y = torch.meshgrid(x_points, y_points, indexing='ij')

    xy_points = torch.stack([x, y], dim=-1)

    # geom_util.get_xy_grid_points()
    # x_slice = slice(min_grid_ind[0], max_grid_ind[0] + 1)
    # y_slice = slice(min_grid_ind[1], max_grid_ind[1] + 1)

    sliced_grid_inds = terrain.get_grid_index(xy_points)
    sliced_hf = terrain.hf[sliced_grid_inds[..., 0], sliced_grid_inds[..., 1]].clone()
    sliced_hf_mask = terrain.hf_mask[sliced_grid_inds[..., 0], sliced_grid_inds[..., 1]].clone()
    sliced_hf_maxmin = terrain.hf_maxmin[sliced_grid_inds[..., 0], sliced_grid_inds[..., 1]].clone()

    if localize: # NOTE: won't work with batched motion frames right?
        # move terrain and motion so that motion start frame is at xy = (0, 0)
        canon_xy = motion_frames[..., 0, 0:2].clone()
        localized_motion = motion_frames.clone()
        localized_motion[..., 0:2] -= canon_xy

        sliced_terrain = SubTerrain(
            x_dim = sliced_hf.shape[0],
            y_dim = sliced_hf.shape[1],
            dx = terrain.dxdy[0].item(),
            dy = terrain.dxdy[1].item(),
            min_x = min_grid_point[0].item() - canon_xy[0].item(),
            min_y = min_grid_point[1].item() - canon_xy[1].item(),
            device=device)
        
        assert sliced_hf.shape == sliced_terrain.hf.shape

        sliced_terrain.hf = sliced_hf
        sliced_terrain.hf_mask = sliced_hf_mask
        sliced_terrain.hf_maxmin = sliced_hf_maxmin

        canon_z = sliced_terrain.get_hf_val_from_points(localized_motion[0, 0:2])
        localized_motion[:, 2] -= canon_z
        sliced_terrain.hf = sliced_hf - canon_z
        return sliced_terrain, localized_motion

    else:
        sliced_terrain = SubTerrain(
            x_dim = sliced_hf.shape[0],
            y_dim = sliced_hf.shape[1],
            dx = terrain.dxdy[0].item(),
            dy = terrain.dxdy[1].item(),
            min_x = min_grid_point[0].item(),
            min_y = min_grid_point[1].item(),
            device=device)
        
        assert sliced_hf.shape == sliced_terrain.hf.shape

        sliced_terrain.hf = sliced_hf
        sliced_terrain.hf_mask = sliced_hf_mask
        sliced_terrain.hf_maxmin = sliced_hf_maxmin
        return sliced_terrain



def slice_terrain(terrain: SubTerrain, min_i, min_j, max_i, max_j):
    terrain.dims[0] = max_i - min_i + 1
    terrain.dims[1] = max_j - min_j + 1

    terrain.hf = terrain.hf[min_i:max_i+1, min_j:max_j+1]
    terrain.hf_mask = terrain.hf_mask[min_i:max_i+1, min_j:max_j+1]
    terrain.hf_maxmin = terrain.hf_maxmin[min_i:max_i+1, min_j:max_j+1]

    terrain.min_point = terrain.get_point(torch.tensor([min_i, min_j], dtype=torch.float32, device=terrain.hf.device))
    return

def get_terrain_slice(terrain: SubTerrain, min_i, min_j, max_i, max_j):
    copy_terrain = terrain.torch_copy()
    slice_terrain(copy_terrain, min_i, min_j, max_i, max_j)
    return copy_terrain

def points_boxes_sdf(points, box_centers, box_halfdims):
    # points: (batch_size, N, 3)
    # box_centers: (batch_size, M, 3)
    # box_halfdims: (batch_size, M, 3)
    # ^ this one could be changed if we want since all xy halfdims would be the same in our setting

    assert len(points.shape) == len(box_centers.shape) == len(box_halfdims.shape)

    if len(points.shape) == 2: # (N, 3)
        points = points.unsqueeze(0)
        box_centers = box_centers.unsqueeze(0)
        box_halfdims = box_halfdims.unsqueeze(0)

    batch_size = points.shape[0]
    assert batch_size == box_centers.shape[0]

    N = points.shape[1]
    M = box_centers.shape[1]

    points_exp = points.unsqueeze(2).expand(batch_size, N, M, 3)  # Shape (B, N, M, 3)
    box_centers_exp = box_centers.unsqueeze(1).expand(batch_size, N, M, 3)  # Shape (B, N, M, 3)
    half_dims_exp = box_halfdims.unsqueeze(1).expand(batch_size, N, M, 3)  # Shape (B, N, M, 3)

    # translate points to be in coord frame of the box
    relative_points = points_exp - box_centers_exp

    sdfs = geom_util.sdBox(relative_points, half_dims_exp)
    return sdfs

def points_round_boxes_sdf(points, box_centers, box_halfdims, radius):
    # points: (batch_size, N, 3)
    # box_centers: (batch_size, M, 3)
    # box_halfdims: (batch_size, M, 3)
    # ^ this one could be changed if we want since all xy halfdims would be the same in our setting

    assert len(points.shape) == len(box_centers.shape) == len(box_halfdims.shape)

    if len(points.shape) == 2: # (N, 3)
        points = points.unsqueeze(0)
        box_centers = box_centers.unsqueeze(0)
        box_halfdims = box_halfdims.unsqueeze(0)

    batch_size = points.shape[0]
    assert batch_size == box_centers.shape[0]

    N = points.shape[1]
    M = box_centers.shape[1]

    points_exp = points.unsqueeze(2).expand(batch_size, N, M, 3)  # Shape (B, N, M, 3)
    box_centers_exp = box_centers.unsqueeze(1).expand(batch_size, N, M, 3)  # Shape (B, N, M, 3)
    half_dims_exp = box_halfdims.unsqueeze(1).expand(batch_size, N, M, 3)  # Shape (B, N, M, 3)

    # translate points to be in coord frame of the box
    relative_points = points_exp - box_centers_exp

    sdfs = geom_util.sdRoundBox(relative_points, half_dims_exp, radius)
    return sdfs

def points_hf_sdf(points: torch.Tensor, # [B, N, 3]
                  hf: torch.Tensor, # [B, dim_x, dim_y]
                  hf_min_box_center: torch.Tensor, # [B, 2]
                  hf_dxdy: torch.Tensor, # [2], all same dxdy
                  base_z = -10.0,
                  inverted = True, # to get proper interior distances,
                  radius = None
):
    # assume hf is axis aligned, and points are in the hf's coordinate frame

    # Things will get really messy if we don't assert these shapes
    assert len(points.shape) == 3
    assert len(hf.shape) == 3
    assert len(hf_min_box_center.shape) == 2
    assert len(hf_dxdy.shape) == 1

    batch_size = points.shape[0]
    device = points.device
    dim_x = hf.shape[1]
    dim_y = hf.shape[2]
    num_boxes = dim_x * dim_y

    x_points = torch.linspace(0.0, (dim_x - 1.0) * hf_dxdy[0].item(), dim_x, device=device)
    y_points = torch.linspace(0.0, (dim_y - 1.0) * hf_dxdy[1].item(), dim_y, device=device)

    x, y = torch.meshgrid(x_points, y_points, indexing='ij') # shape: [dim_x, dim_y]
    x = x.unsqueeze(0) + hf_min_box_center[..., 0].unsqueeze(1).unsqueeze(1) # shape: [B, dim_x, dim_y]
    y = y.unsqueeze(0) + hf_min_box_center[..., 1].unsqueeze(1).unsqueeze(1) # shape: [B, dim_x, dim_y]

    if inverted:
        top_z = -base_z
        z = (hf + top_z) / 2.0
        z_halfdims = (top_z - hf) / 2.0



    else:
        z = (hf + base_z) / 2.0 # shape: [B, dim_x, dim_y]
        z_halfdims = (hf - base_z) / 2.0 # shape: [B, dim_x, dim_y]

    box_centers = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1)], dim=-1).view(batch_size, num_boxes, 3)
    # collapsing dim_x, dim_y to dim_x * dim_y

    hf_dxdy_exp = (hf_dxdy / 2.0).unsqueeze(0).unsqueeze(0).expand(batch_size, num_boxes, 2)
    box_halfdims = torch.cat([hf_dxdy_exp, z_halfdims.view(batch_size, num_boxes).unsqueeze(-1)], dim=-1)

    if radius is None:
        sdfs = points_boxes_sdf(points, box_centers, box_halfdims)
    else:
        assert isinstance(radius, float)
        assert radius > 0.0
        sdfs = points_round_boxes_sdf(points, box_centers, box_halfdims, radius)

    point_to_hf_sdfs = torch.min(sdfs, dim=-1)[0]

    if inverted:
        point_to_hf_sdfs *= -1.0

    return point_to_hf_sdfs

def motion_frames_hf_sdf_loss(motion_frames, char_point_samples, hf, hf_min_box_center, hf_dxdy, 
                              char_model: kin_char_model.KinCharModel,
                              ret_vis_info = False, interior_distance = True,
):
    # TODO
    # hf assumed to be axis aligned, motion_frames assumed to be in hf coord frame
    # motion_frames: [batch_size, seq_len, 34]
    # char_point_samples: [num_bodies, num_points(b), 3]
    # hf: [batch_size, dim_x, dim_y]
    # hf_min_box_center: [batch_size, 2]
    # hf_dxdy: [2]

    batch_size = motion_frames.shape[0]
    device = motion_frames.device

    root_pos = motion_frames[..., 0:3]
    root_rot = motion_frames[..., 3:6]
    joint_dof = motion_frames[..., 6:34]

    root_rot_quat = torch_util.exp_map_to_quat(root_rot)
    joint_rot = char_model.dof_to_rot(joint_dof)
    joint_rot = torch_util.quat_pos(joint_rot)
    body_pos, body_rot = char_model.forward_kinematics(root_pos, root_rot_quat, joint_rot)


    transformed_points = []
    for b in range(char_model.get_num_joints()):
        # unsqueeze sequence dimension and batch dimension
        curr_body_points = char_point_samples[b].unsqueeze(0).unsqueeze(0) # shape: (1, 1, num_points(b), 3)

        body_rot_unsq = body_rot[..., b, :].unsqueeze(2) # shape: (batch_size, seq_len, 1, 4)
        body_pos_unsq = body_pos[..., b, :].unsqueeze(2) # shape: (batch_size, seq_len, 1, 3)

        curr_body_points = torch_util.quat_rotate(body_rot_unsq, curr_body_points) + body_pos_unsq
        # shape: (batch_size, seq_len, num_points(b), 3)

        # flatten points across seq_len and num_points dims
        curr_body_points = curr_body_points.view(batch_size, -1, 3)

        transformed_points.append(curr_body_points)
    transformed_points = torch.cat(transformed_points, dim = 1) # shape: (batch_size, num points, 3)

    # TODO: ensure base_z input is much lower than lowest point
    sdf = points_hf_sdf(transformed_points, hf, hf_min_box_center, hf_dxdy, base_z = -10.0, inverted=interior_distance)
    # shape: [batch_size, num_points]

    if interior_distance:
        loss = 0.5 * torch.sum(torch.square(torch.clamp(sdf, max=0.0)), dim=-1)
    else:
        loss = 0.5 * torch.sum(torch.square(torch.clamp(sdf, min=0.0)), dim=-1)
    # TODO: visually test with mdm tester
    if ret_vis_info:
        return loss, transformed_points, sdf
    else:
        return loss
    
def compute_hf_mask_inds(motion_frames: torch.Tensor,
                         terrain: SubTerrain,
                         char_model: kin_char_model.KinCharModel,
                         char_body_points: List[torch.Tensor]):

    num_frames = motion_frames.shape[0]

    # For each frame of motion, for each RB, determine the lowest point.
    root_pos = motion_frames[..., 0:3]
    root_rot = torch_util.exp_map_to_quat(motion_frames[..., 3:6])
    joint_rot = char_model.dof_to_rot(motion_frames[..., 6:])

    body_pos, body_rot = char_model.forward_kinematics(root_pos, root_rot, joint_rot)
    # For each RB, get its AABB

    terrain.hf_mask[...] = False

    mask_grid_inds = []

    min_body_heights = torch.zeros_like(terrain.hf)
    min_body_heights[...] = 99999.9999

    for t in range(num_frames):
        curr_body_grid_inds = []
        for b in range(char_model.get_num_joints()):
            curr_body_points = torch_util.quat_rotate(body_rot[t, b].unsqueeze(0), char_body_points[b]) + body_pos[t, b]

            curr_grid_inds = terrain.get_grid_index(curr_body_points[..., 0:2])

            if len(curr_grid_inds.shape) == 1:
                curr_grid_inds = curr_grid_inds.unsqueeze(0)

            curr_body_grid_inds.append(curr_grid_inds)

            # also compute height diffs
            #curr_height_diffs = curr_body_points[..., 2] - terrain.hf[curr_grid_inds[..., 0], curr_grid_inds[..., 1]]

            for ind in range(curr_body_points.shape[0]):
                ind_i = curr_grid_inds[ind, 0]
                ind_j = curr_grid_inds[ind, 1]
                min_body_heights[ind_i, ind_j] = min(curr_body_points[ind, 2], min_body_heights[ind_i, ind_j])

        curr_body_grid_inds = torch.cat(curr_body_grid_inds, dim=0)
        curr_body_grid_inds = torch.unique(curr_body_grid_inds, dim=0)

        mask_grid_inds.append(curr_body_grid_inds)
    return mask_grid_inds, min_body_heights

def compute_hf_mask_from_inds(terrain: SubTerrain,
                              mask_grid_inds: List[torch.Tensor]):
    hf_mask = torch.zeros_like(terrain.hf_mask)
    num_frames = len(mask_grid_inds)
    for t in range(num_frames):
        curr_mask_grid_inds = mask_grid_inds[t]
        hf_mask[curr_mask_grid_inds[..., 0], curr_mask_grid_inds[..., 1]] = True
    return hf_mask

def compute_hf_mask(motion_frames: torch.Tensor,
                    terrain: SubTerrain, 
                    char_model: kin_char_model.KinCharModel,
                    char_body_points: List[torch.Tensor]):
    
    mask_grid_inds, _ = compute_hf_mask_inds(motion_frames, terrain, char_model, char_body_points)
    return compute_hf_mask_from_inds(terrain, mask_grid_inds)


def compute_hf_extra_vals(motion_frames: torch.Tensor,
                          terrain: SubTerrain, 
                          char_model: kin_char_model.KinCharModel,
                          char_body_points: List[torch.Tensor],
                          z_buf = 3.0,
                          jump_buf = 0.8):
    

    mask_grid_inds, min_body_heights = compute_hf_mask_inds(motion_frames, terrain, char_model, char_body_points)
    hf_mask = compute_hf_mask_from_inds(terrain, mask_grid_inds)
    terrain.hf_mask = hf_mask

    max_h = torch.max(motion_frames[:, 2]).item()
    min_h = torch.min(terrain.hf).item()

    terrain.hf_maxmin[..., 0] = max_h + z_buf
    terrain.hf_maxmin[..., 1] = min_h - z_buf

    hf_vals = terrain.hf[terrain.hf_mask]
    terrain.hf_maxmin[..., 0][terrain.hf_mask] = hf_vals
    terrain.hf_maxmin[..., 1][terrain.hf_mask] = hf_vals


    height_diffs = min_body_heights - terrain.hf

    jump_inds = height_diffs >= jump_buf
    jump_inds = torch.logical_and(jump_inds, terrain.hf_mask)
    terrain.hf_maxmin[..., 0][jump_inds] = min_body_heights[jump_inds] - jump_buf
    terrain.hf_maxmin[..., 1][jump_inds] = min_h - z_buf

    return mask_grid_inds

def sample_hf_z_on_terrain(terrain: SubTerrain,
                           center_xy: torch.Tensor,
                           heading: torch.Tensor,
                           dx: float, dy: float,
                           num_x_neg: int, num_x_pos: int, num_y_neg: int, num_y_pos: int):
    
    local_hf_xy_points = geom_util.get_xy_grid_points(
        center=torch.zeros(size=(2,), dtype=torch.float32, device=center_xy.device),
        dx=dx,
        dy=dy,
        num_x_neg=num_x_neg,
        num_x_pos=num_x_pos,
        num_y_neg=num_y_neg,
        num_y_pos=num_y_pos)
    
    if len(center_xy.shape) == 2:
        center_xy = center_xy.unsqueeze(1).unsqueeze(1)
    elif len(center_xy.shape) == 3:
        assert center_xy.shape[1] == 1
        center_xy = center_xy.unsqueeze(1)
    # shape: [batch_size, 1, 1, 2]

    if len(heading.shape) == 2:
        assert heading.shape[1] == 1
        heading = heading.unsqueeze(1)
    elif len(heading.shape) == 1:
        heading = heading.unsqueeze(1).unsqueeze(1)
    # shape: [batch_size, 1, 1]

    global_hf_xy_points = torch_util.rotate_2d_vec(local_hf_xy_points, heading) + center_xy
    hf_z = terrain.get_hf_val_from_points(global_hf_xy_points)

    # shape: [batch_size, grid_dim_x, grid_dim_y]
    return hf_z

def flat_maxpool_2x2(terrain: SubTerrain):
    dim_x = terrain.hf.shape[0]
    dim_y = terrain.hf.shape[1]
    for i in range(0, dim_x - 1, 2):
        for j in range(0, dim_y - 1, 2):

            max_h = torch.max(terrain.hf[i:i+2, j:j+2]).item()

            terrain.hf[i:i+2, j:j+2] = max_h
            # for ii in range(2):
            #     for jj in range(2):
            #         grid_ind = torch.tensor([i + ii, j + jj], dtype=torch.int64, device=terrain.hf.device)
            #         terrain.set_hf_val(grid_ind, max_h)
    return

def flat_maxpool_3x3(terrain: SubTerrain):

    dim_x = terrain.hf.shape[0]
    dim_y = terrain.hf.shape[1]
    for i in range(0, dim_x - 2, 3):
        for j in range(0, dim_y - 2, 3):

            max_h = torch.max(terrain.hf[i:i+3, j:j+3]).item()

            terrain.hf[i:i+3, j:j+3] = max_h
            # for ii in range(2):
            #     for jj in range(2):
            #         grid_ind = torch.tensor([i + ii, j + jj], dtype=torch.int64, device=terrain.hf.device)
            #         terrain.set_hf_val(grid_ind, max_h)
    return

def flatten_4x4_near_edge(terrain: SubTerrain, grid_ind: torch.Tensor, height: float):
    # Creates a 4x4 flat region around the grid ind.
    # Ensures that the start of the region is on an even index,
    # and that the grid ind is in the middle 2x2 region of the 4x4 region
    if grid_ind[0].item() % 2 == 0:
        x_start = grid_ind[0].item() - 2
    else:
        x_start = grid_ind[0].item() - 1
    x_end = x_start + 4
    x_slice = slice(x_start, x_end)

    if grid_ind[1].item() % 2 == 0:
        y_start = grid_ind[1].item() - 2
    else:
        y_start = grid_ind[1].item() - 1
    y_end = y_start + 4
    y_slice = slice(y_start, y_end)

    terrain.hf[x_slice, y_slice] = height
    return


def generate_cave(height, width, num_iterations, device, dxy=0.4):
    def initialize_map(height, width, p=0.5):
        return torch.rand(height, width) < p

    def apply_cellular_automata(map, num_iterations, birth_threshold=6, death_threshold=4):
        height, width = map.shape
        for _ in range(num_iterations):
            # Count alive neighbors
            neighbor_count = torch.nn.functional.conv2d(map.float().unsqueeze(0).unsqueeze(0),
                                                        torch.ones(1, 1, 3, 3, device=map.device),
                                                        padding=1).squeeze(0).squeeze(0)
            # Apply rules
            birth_mask = (neighbor_count >= birth_threshold) & (~map)
            death_mask = (neighbor_count <= death_threshold) & map
            map = map | birth_mask
            map = map & ~death_mask




        return map
    map = initialize_map(height, width)
    cave_map = apply_cellular_automata(map, num_iterations)

    cave_map = cave_map.to(dtype=torch.float32, device=device)

    terrain = SubTerrain(x_dim=width, y_dim=height, dx=dxy, dy=dxy, device=device)
    terrain.hf = cave_map

    #SubTerrain("cave")
    return terrain