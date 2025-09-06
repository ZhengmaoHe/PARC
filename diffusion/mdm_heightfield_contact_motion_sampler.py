import anim.motion_lib as motion_lib
from anim.motion_lib import LoopMode
import anim.kin_char_model as kin_char_model
import torch
import pickle

import util.torch_util as torch_util

from diffusion.motion_sampler import MotionSampler#, canonicalize_samples

import util.geom_util as geom_util
import util.terrain_util as terrain_util
import random
import enum

import diffusion.utils.rot_changer as rot_changer
from diffusion.diffusion_util import MDMFrameType, RelativeZStyle, TargetInfo
"""
MDMHeightfieldContactMotionSampler Overview:
This class samples motions from files that come attached with a special terrain class.
The local heightfield of a sampled motion field is extracted from the clip's global heightfield,
and augmented using motion-aware randomization.
"""

class HFAugmentationMode(enum.Enum):
    NOISE = 0
    MAXPOOL_AND_BOXES = 1
    NONE = 2

class MDMHeightfieldContactMotionSampler(MotionSampler):
    def __init__(self, cfg):
        super().__init__(cfg)

        self._canonicalize_samples = cfg['features']['canonicalize_samples']
        frame_components = cfg['features']['frame_components']
        self._frame_components = []
        for comp in frame_components:
            comp = MDMFrameType[comp]
            self._frame_components.append(comp)
        
        # why specify here, maybe we should load it from config class/_kin_char_model of our skeleton?
        #self._num_dof = 34
        #self._normalize_dofs = cfg['normalize_dofs']
        self._num_rb = len(self._kin_char_model._body_names)

        if self._canonicalize_samples:
            print("samples will be canonicalized")

        self._fps = cfg['sequence_fps']
        self._timestep = 1.0 / self._fps

        # If we wish to train an autoregressive model, then we need to sample 
        # from different starting frames
        self._random_start_times = cfg['autoregressive']
        if self._random_start_times:
            print("sampled start times will be random")
        
        self._sample_seq_time = cfg['sequence_duration']
        self._motion_times = torch.arange(start=0.0, end=self._sample_seq_time, step=self._timestep,
                                          dtype=torch.float32, device=self._device)
        self._seq_len = self._motion_times.shape[0]
        print("sample sequence length:", self._seq_len)
        
        self._num_prev_states = cfg['num_prev_states']
        self._canon_idx = cfg['num_prev_states'] - 1 # this being disconnected from the MDM trainer might lead to bugs in the future
        
        
        self._future_pos_noise_scale = cfg["future_pos_noise_scale"]

        # use this to generate heightmap observations from the box obs
        self._use_saved_heightmaps = cfg["use_saved_heightmaps"]
        
        self._relative_z_style = RelativeZStyle[cfg["relative_z_style"]]

        self._hf_augmentation_mode = HFAugmentationMode[cfg["hf_augmentation_mode"]]
        hmap_cfg = cfg["heightmap"]
        dx = hmap_cfg["horizontal_scale"]
        dy = dx
        self._dx = dx
        num_x_neg = hmap_cfg["local_grid"]["num_x_neg"]
        num_x_pos = hmap_cfg["local_grid"]["num_x_pos"]
        num_y_neg = hmap_cfg["local_grid"]["num_y_neg"]
        num_y_pos = hmap_cfg["local_grid"]["num_y_pos"]
        self._num_x_neg = num_x_neg
        self._num_x_pos = num_x_pos
        self._num_y_neg = num_y_neg
        self._num_y_pos = num_y_pos
        self._grid_min_point = torch.tensor([-num_x_neg, -num_y_neg], dtype=torch.float32, device=self._device) * dx
        grid_dim_x = num_x_neg + 1 + num_x_pos
        grid_dim_y = num_y_neg + 1 + num_y_pos
        self._grid_dim_x = grid_dim_x
        self._grid_dim_y = grid_dim_y
        self._grid_dims = torch.tensor([grid_dim_x, grid_dim_y], dtype=torch.int64, device=self._device)
        self._num_hf_points = grid_dim_x * grid_dim_y
        zero = torch.zeros(size=(2,), dtype=torch.float32, device=self._device)
        self._generic_heightmap = geom_util.get_xy_grid_points(zero, dx, dy, num_x_neg, num_x_pos, num_y_neg, num_y_pos)


        self._max_h = cfg["heightmap"]["max_h"]
        self._min_h = -self._max_h
        
        self._use_hf_augmentation = cfg["use_hf_augmentation"]
        if self._use_hf_augmentation:
            self._max_num_boxes = cfg["max_num_boxes"]
            self._box_min_len = cfg["box_min_len"]
            self._box_max_len = cfg["box_max_len"]
            self._hf_maxpool_chance = cfg["hf_maxpool_chance"]
            self._hf_max_maxpool_size = cfg["hf_max_maxpool_size"]
            self._hf_change_height_chance = cfg["hf_change_height_chance"]

        num_prev_states = cfg['num_prev_states']
        assert num_prev_states >= 1
        self._ref_frame_idx = num_prev_states - 1 # the state that the motions are observations are canonicalized wrt

        # for randomly perturbing motions
        self._angle_noise_scale = cfg["angle_noise_scale"]
        self._pos_noise_scale = cfg["pos_noise_scale"]

        # future window times
        self._future_window_min = cfg["future_window_min"]
        self._future_window_max = cfg["future_window_max"]

        self.check_init()
        return

    # def _compute_normalizing_tensor(self, motion_lib_file, canon_idx):
    #     print("Computing normalizing tensor")
    #     self._dof_high = torch.zeros(size=(self._num_dof,), dtype=torch.float32)
    #     self._dof_high[0] = 1.0 # the min normalizing value for position will be 1.0
    #     self._dof_high[1] = 1.0
    #     self._dof_high[2] = 1.0
    #     if not self._normalize_dofs:
    #         # pre-computed values
    #         self._dof_high[0] = 1.4191
    #         self._dof_high[1] = 1.2948
    #         self._dof_high[2] = 1.1406
    #         self._dof_high[3:] = torch.pi
    #     else:
    #         self._dof_high[3:] = torch.pi # don't bother computer a normalizing value for joints

    #         # TODO: a function/script to compute motion stats

    #         motion_files, motion_weights = self._mlib._fetch_motion_files(motion_lib_file)
    #         num_motion_files = len(motion_files)
    #         for f in range(num_motion_files):
    #             curr_file = motion_files[f]
    #             print("Loading {:d}/{:d} motion files for normalization: {:s}".format(f + 1, num_motion_files, curr_file))
                
    #             with open(curr_file, "rb") as filestream:
    #                 curr_motion = pickle.load(filestream)
    #             motion_frames = torch.tensor(curr_motion["frames"])
    #             num_frames = motion_frames.shape[0]
    #             assert num_frames > self._seq_len

    #             # Since all the motions will be canonicalized, we need to make sure the normalizing
    #             # tensor is using the canonicalized values
    #             for i in range(num_frames-self._seq_len):
    #                 motion_slice = motion_frames[i:i+self._seq_len].clone().unsqueeze(dim=0)
    #                 motion_slice = canonicalize_samples(motion_slice, canon_idx)
    #                 motion_slice.squeeze(dim=0)
    #                 self._dof_high[0] = torch.max(torch.abs(motion_slice[:, 0]))
    #                 self._dof_high[1] = torch.max(torch.abs(motion_slice[:, 1]))
    #                 self._dof_high[2] = torch.max(torch.abs(motion_slice[:, 2]))
        
    #     self._dof_high = self._dof_high.unsqueeze(0)
    #     self._dof_high = self._dof_high.to(device=self._device)
    #     self._dof_high_np = self._dof_high.cpu().numpy()

    #     print("Finished computing normalizing tensor")
    #     print(self._dof_high)
    #     return
    
    def update_old_sampler(self):
        self._relative_z_style = RelativeZStyle.RELATIVE_TO_ROOT
        return
    
    def get_motion_sequences_for_id(self, motion_id: int):
        motion_full_length = self._mlib.get_motion_length(motion_id)

        dt = 1.0 / self._fps
        seq_duration = self._seq_len * dt
        motion_start_times = torch.arange(start=0, end=motion_full_length - seq_duration, step=dt, dtype=torch.float32, device=self._device)
        #motion_times = motion_start_times.unsqueeze(-1) + self._motion_times
        motion_ids = torch.ones_like(motion_start_times, dtype=torch.int64)
        motion_data = self.sample_motion_data(motion_ids, motion_start_times, ret_hf_obs=False, ret_target_info=False)
        return motion_data
    
    def _sample_motion_future_times(self, motion_ids, start_times, 
                                    future_window_min, future_window_max):
        # Sample a random future time after the given start time.
        # Future window timeframe: [start_time + future_window_min, start_time + future_window_max]
        
        motion_full_lengths = self._mlib.get_motion_length(motion_ids)

        remaining_time = motion_full_lengths - start_times
        max_window_dur = future_window_max - future_window_min
        remaining_time = torch.clamp_max(remaining_time, max_window_dur)

        future_times = torch.rand_like(start_times) * remaining_time + start_times + future_window_min
        return future_times

    def _sample_target_info(self, motion_ids, motion_start_times, canon_pos, canon_heading_quat_inv):
        motion_future_times = self._sample_motion_future_times(motion_ids, 
                                                               motion_start_times,
                                                               self._future_window_min, 
                                                               self._future_window_max)
        
        future_pos, future_rot = self._extract_root_pos_and_rot(motion_ids, motion_future_times)

        future_pos += self._future_pos_noise_scale * torch.randn_like(future_pos)

        future_pos = future_pos - canon_pos
        future_pos = torch_util.quat_rotate(canon_heading_quat_inv, future_pos)
        future_rot = torch_util.quat_multiply(canon_heading_quat_inv, future_rot)
        target_info = TargetInfo(future_pos, future_rot)
        return target_info

    def _extract_motion_data(self, motion_ids, motion_start_times, motion_times, seq_len):
        # This function returns all motion data in global coordinates.

        num_samples = motion_ids.shape[0]

        # 2) Put all frames we want to sample into flat tensors of size (num_samples x num_frames)
        # (this is a motion_lib requirement)
        motion_times = motion_start_times.unsqueeze(-1) + motion_times
        motion_times = motion_times.flatten()
        repeated_motion_ids = motion_ids.unsqueeze(-1).expand(-1, seq_len).flatten()

        # 3) Get motion data from motion_lib
        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, contacts = self._mlib.calc_motion_frame(repeated_motion_ids, motion_times=motion_times)

        # 4) Reshape motion data
        # motion data is currently in the form (num_samples x num_frames) x (num_dof)
        # We reshape it into (num_samples) x (num_frames) x (num_dof)

        root_pos = torch.reshape(root_pos, shape=[num_samples, seq_len, 3])
        root_rot = torch.reshape(root_rot, shape=[num_samples, seq_len, 4])
        root_vel = torch.reshape(root_vel, shape=[num_samples, seq_len, 3])
        root_ang_vel = torch.reshape(root_ang_vel, shape=[num_samples, seq_len, 3])
        joint_rot = torch.reshape(joint_rot, shape=[num_samples, seq_len, -1, 4])
        # TODO
        #dof_vel = torch.reshape(dof_vel, shape=[num_samples, seq_len, -1])
        num_rb = self._kin_char_model.get_num_joints()
        contacts = torch.reshape(contacts, shape=[num_samples, seq_len, num_rb])

        return root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, contacts


        # # 4) Reshape motion data
        # # motion data is currently in the form (num_samples x num_frames) x (num_dof)
        # # We reshape it into (num_samples) x (num_frames) x (num_dof)
        # root_rot = quat_to_exp_map(root_rot)
        # joint_rot = self._motion_lib.joint_rot_to_dof(joint_rot)
        # motion_samples = torch.cat((root_pos, root_rot, joint_rot), dim=1)
        # motion_samples = torch.reshape(motion_samples, shape=(num_samples, seq_len, self._num_dof))

        # # same thing for contact data
        # # it is initially (num_samples x num_frames) x (num_rb)
        # contacts = torch.reshape(contacts, shape=(num_samples, seq_len, self._num_rb))
        

        # return motion_samples, contacts
    
    def _extract_root_pos_and_rot(self, motion_ids, times):
        # TODO: make more efficient
        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, contacts = self._mlib.calc_motion_frame(motion_ids, motion_times=times)

        return root_pos, root_rot
    
    def sample_motion_data(self, 
                           motion_ids=None, 
                           num_samples = None, 
                           motion_start_times=None,
                           ret_hf_obs=True,
                           ret_target_info=True):
        if motion_ids == None:
            assert(num_samples != None)
            motion_ids = self._mlib.sample_motions(num_samples)
        else:
            num_samples = motion_ids.shape[0]

        if motion_start_times is None:
            motion_start_times = self._sample_motion_start_times(motion_ids, self._sample_seq_time)


        motion_start_time_indices = torch.round((motion_start_times / self._timestep)).to(dtype=torch.int64)
        future_time_indices = torch.arange(start=0, end=self._seq_len, dtype=torch.int64, device=self._device).unsqueeze(0)
        motion_time_indices = motion_start_time_indices.unsqueeze(-1) + future_time_indices
        # this should never go out of bounds based on the way motion start times is sampled right...


        ##### GET THE MOTION FRAME DATA FROM MOTION LIB ######
        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, contacts = self._extract_motion_data(motion_ids, 
                                                                                                             motion_start_times, 
                                                                                                             self._motion_times,
                                                                                                             self._seq_len)
        
        # get the reference values for canonicalization
        canon_root_pos = root_pos[:, self._ref_frame_idx, :].clone()
        canon_root_rot = root_rot[:, self._ref_frame_idx, :].clone()
        canon_root_z = canon_root_pos[:, 2].clone()

        

        # get hfs from canonicalized but unnormalized motion samples
        if ret_hf_obs:
            hfs, center_h = self.get_hfs_from_data(motion_ids, canon_root_pos, canon_root_rot, canon_root_z, motion_time_indices)
            hfs = torch.clamp(hfs, min=self._min_h, max=self._max_h)

            if self._relative_z_style == RelativeZStyle.RELATIVE_TO_ROOT_FLOOR:
                # We also need to make sure the heights of the root positions are relative to 0
                root_pos[..., 2] = root_pos[..., 2] - center_h.unsqueeze(-1)

        ##### CANONICALIZE ######
        root_pos = root_pos - canon_root_pos.unsqueeze(1)
        canon_heading_quat_inv = torch_util.calc_heading_quat_inv(canon_root_rot)
        root_rot = torch_util.quat_multiply(canon_heading_quat_inv.unsqueeze(1), root_rot)
        root_vel = torch_util.quat_rotate(canon_heading_quat_inv.unsqueeze(1), root_vel)
        root_pos = torch_util.quat_rotate(canon_heading_quat_inv.unsqueeze(1), root_pos)
        
        body_pos, body_rot = self._kin_char_model.forward_kinematics(root_pos, root_rot, joint_rot)
        # ignore the root body pos, since that is the root position
        body_pos = body_pos[..., 1:, :]
        #joint_dof = self._kin_char_model.rot_to_dof(joint_rot)

        motion_ret = dict()

        for frame_type in self._frame_components:
            if frame_type == MDMFrameType.ROOT_POS:
                motion_ret[MDMFrameType.ROOT_POS] = root_pos
            if frame_type == MDMFrameType.ROOT_ROT:
                motion_ret[MDMFrameType.ROOT_ROT] = root_rot
            if frame_type == MDMFrameType.JOINT_POS:
                motion_ret[MDMFrameType.JOINT_POS] = body_pos
            if frame_type == MDMFrameType.JOINT_ROT:
                motion_ret[MDMFrameType.JOINT_ROT] = joint_rot
            if frame_type == MDMFrameType.CONTACTS:
                motion_ret[MDMFrameType.CONTACTS] = contacts
            if frame_type == MDMFrameType.FLOOR_HEIGHTS:
                assert ret_hf_obs
                floor_heights = []
                for i in range(hfs.shape[0]):
                    hf = hfs[i]
                    root_pos_xy = root_pos[:, :, 0:2]
                    
                    grid_inds = torch.round((root_pos_xy - self._grid_min_point) / self._dx).to(dtype=torch.int64)
                    grid_inds = torch.clamp(grid_inds, torch.zeros_like(self._grid_dims), self._grid_dims-1)
                    floor_heights.append(hf[grid_inds[..., 0], grid_inds[..., 1]])
                floor_heights = torch.stack(floor_heights).unsqueeze(-1)
                motion_ret[MDMFrameType.FLOOR_HEIGHTS] = floor_heights


        if ret_hf_obs or ret_target_info:
            ret = [motion_ret]
            if ret_hf_obs:
                ret.append(hfs)
            if ret_target_info:
                target_info = self._sample_target_info(motion_ids, motion_start_times, canon_root_pos, canon_heading_quat_inv)
                ret.append(target_info)

            return tuple(ret)
        else:
            return motion_ret

    def _box_hf_augmentation(self, hf, hf_maxmin):
        change_height = random.random() < self._hf_change_height_chance

        if change_height:
            new_height = random.random() * (self._max_h - self._min_h) + self._min_h
            hf[...] = new_height


        use_maxpool_1 = random.random() < self._hf_maxpool_chance
        use_maxpool_2 = random.random() < self._hf_maxpool_chance
        use_maxpool_3 = random.random() < self._hf_maxpool_chance
        
        # randomize the order of the maxpools
        maxpool_fns = [terrain_util.maxpool_hf, 
                       terrain_util.maxpool_hf_1d_x, 
                       terrain_util.maxpool_hf_1d_y]
        
        random.shuffle(maxpool_fns)

        if use_maxpool_1:
            maxpool_size = random.randint(0, self._hf_max_maxpool_size)
            maxpool_fns[0](hf, hf_maxmin, maxpool_size)

        if use_maxpool_2:
            maxpool_size = random.randint(0, self._hf_max_maxpool_size)
            maxpool_fns[1](hf, hf_maxmin, maxpool_size)

        if use_maxpool_3:
            maxpool_size = random.randint(0, self._hf_max_maxpool_size)
            maxpool_fns[2](hf, hf_maxmin, maxpool_size)
        
        num_boxes = random.randint(0, self._max_num_boxes)

        # BOXES ARE BEING ADDED TO THE NON_RELATIVE HEIGHTMAP,
        # so we need to use center_h to fix that
        # directly modifies hf
        terrain_util.add_boxes_to_hf2(hf, box_max_height=self._max_h,
                                     box_min_height=self._min_h,
                                     hf_maxmin=hf_maxmin, 
                                     num_boxes=num_boxes, 
                                     box_max_len=self._box_max_len, 
                                     box_min_len = self._box_min_len)
        return

    def _noise_hf_augmentation(self, hf, hf_maxmin):
        noisy_hf = torch.rand_like(hf) * (self._max_h - self._min_h) + self._min_h
        hf[...] = torch.clamp(noisy_hf, hf_maxmin[..., 0], hf_maxmin[..., 1])
        return
    
    def get_hfs_from_data_helper(self, motion_ids, xy_points, motion_time_indices):

        hfs = []
        hf_maxmins = []
        for i in range(len(motion_ids)):
            id = motion_ids[i]
            hf_inds = self._mlib._terrains[id].get_grid_index(xy_points[i])
            hf = self._mlib._terrains[id].hf[hf_inds[..., 0], hf_inds[..., 1]]

            hf_maxmin = torch.zeros(size=[*hf.shape] + [2], dtype=torch.float32, device=self._device)
        
            #if self._use_hf_augmentation:
            #hf_mask = self._motion_lib._terrains[id].hf_mask[hf_inds[..., 0], hf_inds[..., 1]]
            #hf_maxmin = self._motion_lib._terrains[id].hf_maxmin[hf_inds[..., 0], hf_inds[..., 1]]


            # We get the mask indices for the frames of the current motion,
            # to tell the hf augmenter which blocks to not change
            time_index_slice = slice(motion_time_indices[i][0], motion_time_indices[i][-1] + 1)
            hf_mask_inds = self._mlib._hf_mask_inds[id][time_index_slice]
            terrain_hf_mask = terrain_util.compute_hf_mask_from_inds(self._mlib._terrains[id], hf_mask_inds)
            #hf_mask = terrain_hf_mask[hf_inds[..., 0], hf_inds[..., 1]]
            terrain_hf_mask = terrain_hf_mask.unsqueeze(-1).expand(size=[-1, -1, 2])

            terrain_hf_maxmin = torch.zeros_like(self._mlib._terrains[id].hf_maxmin)
            terrain_hf_maxmin[..., 0] = self._max_h * 2.0
            terrain_hf_maxmin[..., -1] = self._min_h * 2.0

            terrain_hf_maxmin[terrain_hf_mask] = self._mlib._terrains[id].hf_maxmin[terrain_hf_mask]
            hf_maxmin = terrain_hf_maxmin[hf_inds[..., 0], hf_inds[..., 1], :]

            hfs.append(hf)
            hf_maxmins.append(hf_maxmin)
        return hfs, hf_maxmins
    
    def get_hfs_from_data(self, motion_ids, ref_root_pos, ref_root_rot, canon_root_z, motion_time_indices):
        num_samples = ref_root_pos.shape[0]
        hf_xy_points = self._generic_heightmap.clone().unsqueeze(dim=0)
        hf_xy_points = hf_xy_points.expand(num_samples, -1, -1, -1)
        char_heading = torch_util.calc_heading(ref_root_rot)
        char_heading = char_heading.unsqueeze(-1).unsqueeze(-1).expand(-1, self._grid_dim_x, self._grid_dim_y)
        hf_xy_points = torch_util.rotate_2d_vec(hf_xy_points, char_heading)
        hf_xy_points = hf_xy_points + ref_root_pos[:, 0:2].unsqueeze(1).unsqueeze(1)

        hfs, hf_maxmins = self.get_hfs_from_data_helper(motion_ids, hf_xy_points, motion_time_indices)

        hfs = torch.stack(hfs, dim=0)
        hf_maxmins = torch.stack(hf_maxmins, dim=0)
        center_h = hfs[:, self._num_x_neg, self._num_y_neg].clone()
        if self._relative_z_style == RelativeZStyle.RELATIVE_TO_ROOT_FLOOR:
            # This makes it so that the height of the grid cell directly below the canon index
            # is always "0". So if the data has the character up on a platform, the relative height of everything changes.
            hfs = hfs - center_h.unsqueeze(-1).unsqueeze(-1)
            hf_maxmins = hf_maxmins - center_h.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

            #future_pos[..., 2] = future_pos[..., 2] - center_h
        elif self._relative_z_style == RelativeZStyle.RELATIVE_TO_ROOT:
            hfs = hfs - canon_root_z.unsqueeze(-1).unsqueeze(-1)
            hf_maxmins = hf_maxmins - canon_root_z.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        else:
            assert False

        if self._use_hf_augmentation:
            for i in range(hfs.shape[0]):
                if not hasattr(self, "_hf_augmentation_mode") or self._hf_augmentation_mode == HFAugmentationMode.NOISE:
                    #self._noise_hf_augmentation(hf, hf_mask, hf_maxmin, box_heights)
                    self._noise_hf_augmentation(hfs[i], hf_maxmins[i])
                elif self._hf_augmentation_mode == HFAugmentationMode.MAXPOOL_AND_BOXES:
                    self._box_hf_augmentation(hfs[i], hf_maxmins[i])
                elif self._hf_augmentation_mode == HFAugmentationMode.NONE:
                    do_nothing = 0
                else:
                    assert False
        
        return hfs, center_h

    def generate_hfs(self, num_samples, center_floor_z):
        # center_floor_z shape: [num_samples]
        hfs = torch.zeros(size=[num_samples, self._grid_dim_x, self._grid_dim_y], dtype=torch.float32, device=self._device)
        hf_mask = torch.zeros(size=[self._grid_dim_x, self._grid_dim_y], dtype=torch.bool, device=self._device)


        if self._relative_z_style == RelativeZStyle.RELATIVE_TO_ROOT:
            hfs = hfs + center_floor_z.unsqueeze(-1).unsqueeze(-1)

        # hf mask need to be on the center 3x3 cells
        hf_mask[self._num_x_neg-2:self._num_x_neg+3, self._num_y_neg-2:self._num_y_neg+3] = True

        # this option doesn't work without this as the augmentation mode
        assert self._hf_augmentation_mode == HFAugmentationMode.MAXPOOL_AND_BOXES

        for i in range(num_samples):
            hf = hfs[i]
            num_boxes = random.randint(0, self._max_num_boxes)

            # directly modifies hf
            terrain_util.add_boxes_to_hf(hf, hf_mask, 
                                     box_max_height=self._max_h,
                                     box_min_height=self._min_h,
                                     hf_maxmin=None, 
                                     num_boxes=num_boxes, 
                                     box_max_len=self._box_max_len, 
                                     box_min_len = self._box_min_len)
        
        if self._relative_z_style == RelativeZStyle.RELATIVE_TO_ROOT_FLOOR:
            center_h = hfs[:, self._num_x_neg, self._num_y_neg]
            hfs = hfs - center_h.unsqueeze(-1).unsqueeze(-1)

        hfs = torch.clamp(hfs, min=self._min_h, max=self._max_h)

        return hfs

    def sample_mismatched_prev_states_and_hfs(
        self, num_samples, ret_contacts=True, ret_floor_heights=True
    ):
        # specifically used for an OOD generalization loss when training MDM
        prev_state_motion_ids = self._mlib.sample_motions(num_samples)
        prev_state_start_times = self._sample_motion_start_times(prev_state_motion_ids, self._sample_seq_time)
        prev_states, prev_state_contacts = self._extract_motion_data(prev_state_motion_ids, 
                                                                     prev_state_start_times,
                                                                     self._motion_times[:self._num_prev_states],
                                                                     self._num_prev_states)
        
        # get floor height associated with prev state canon idx, and subtract
        for i in range(len(prev_state_motion_ids)):
            id = prev_state_motion_ids[i]
            canon_floor_height = self._mlib._terrains[id].get_hf_val_from_points(prev_states[i, self._canon_idx, 0:2])
            prev_states[..., 2] -= canon_floor_height

        # generate heightfield for these prev states
        # Option 1: generate random boxes
        # Option 2: pick random positions in loaded terrain data

        # we will start with option 1

        prev_states = canonicalize_samples(prev_states, self._canon_idx)

        hfs = self.generate_hfs(num_samples)

        normalized_prev_states = prev_states / self._dof_high.unsqueeze(0)

        final_motion_samples = [normalized_prev_states]

        if ret_contacts:
            final_motion_samples.append(prev_state_contacts)

        if ret_floor_heights:
            floor_heights = []
            for i in range(hfs.shape[0]):
                hf = hfs[i]
                root_pos_xy = prev_states[i, :, 0:2]
                
                grid_inds = torch.round((root_pos_xy - self._grid_min_point) / self._dx).to(dtype=torch.int64)
                grid_inds = torch.clamp(grid_inds, torch.zeros_like(self._grid_dims), self._grid_dims-1)
                floor_heights.append(hf[grid_inds[..., 0], grid_inds[..., 1]])
            floor_heights = torch.stack(floor_heights)
            final_motion_samples.append(floor_heights.unsqueeze(-1))

        final_motion_samples = torch.cat(final_motion_samples, dim=2)

        return final_motion_samples, hfs