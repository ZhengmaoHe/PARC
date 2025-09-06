import torch

import envs.base_env as base_env
#from learning.return_tracker import ReturnTracker

class DMPPOReturnTracker:
    def __init__(self, num_envs, device, target_task=False):
        self._device = device
        self._episodes = 0
        self._mean_ep_len = torch.zeros([1], device=device, dtype=torch.float32)

        # self._return_buf = torch.zeros([num_envs], device=device, dtype=torch.float32)
        self._ep_len_buf = torch.zeros([num_envs], device=device, dtype=torch.long)
        self._eps_per_env_buf = torch.zeros([num_envs], device=device, dtype=torch.long)
        #super().__init__(num_envs, device)

        self._return_bufs = dict()
        self._return_bufs["total_r"] = torch.zeros([num_envs], device=device, dtype=torch.float32)
        self._return_bufs["pose_r"] = torch.zeros([num_envs], device=device, dtype=torch.float32)
        self._return_bufs["vel_r"] = torch.zeros([num_envs], device=device, dtype=torch.float32)
        self._return_bufs["root_pos_r"] = torch.zeros([num_envs], device=device, dtype=torch.float32)
        self._return_bufs["root_vel_r"] = torch.zeros([num_envs], device=device, dtype=torch.float32)
        self._return_bufs["key_pos_r"] = torch.zeros([num_envs], device=device, dtype=torch.float32)
        self._return_bufs["contact_penalty"] = torch.zeros([num_envs], device=device, dtype=torch.float32)
        if target_task:
            self._return_bufs["task_r1"] = torch.zeros([num_envs], device=device, dtype=torch.float32)
            self._return_bufs["task_r2"] = torch.zeros([num_envs], device=device, dtype=torch.float32)
            self._return_bufs["total_task_r"] = torch.zeros([num_envs], device=device, dtype=torch.float32)

        for key in self._return_bufs:
            print(key)
        self._mean_returns = dict()
        for key in self._return_bufs:
            self._mean_returns[key] = torch.zeros([1], device=device, dtype=torch.float32)

        return

    def get_mean_return(self):
        return self._mean_returns["total_r"]

    def get_specific_mean_return(self, key):
        return self._mean_returns[key]

    def get_mean_ep_len(self):
        return self._mean_ep_len

    def get_episodes(self):
        return self._episodes

    def get_eps_per_env(self):
        return self._eps_per_env_buf

    def reset(self):
        self._episodes = 0
        self._eps_per_env_buf[:] = 0

        for key in self._mean_returns:
            self._mean_returns[key][:] = 0.0
        self._mean_ep_len[:] = 0.0
        
        for key in self._return_bufs:
            self._return_bufs[key][:] = 0.0
        self._ep_len_buf[:] = 0
        return

    def update(self, info, done):
        for key in self._return_bufs:
            assert key in info["rewards"], key
            assert(done.shape == self._return_bufs[key].shape)

        for key in info["rewards"]:
            self._return_bufs[key] += info["rewards"][key]
        self._ep_len_buf += 1

        reset_mask = done != base_env.DoneFlags.NULL.value
        reset_ids = reset_mask.nonzero(as_tuple=False).flatten()
        num_resets = len(reset_ids)

        if (num_resets > 0):
            #new_mean_return = torch.mean(self._return_buf[reset_ids])
            new_mean_ep_len = torch.mean(self._ep_len_buf[reset_ids].type(torch.float))

            new_count = self._episodes + num_resets
            w_new = float(num_resets) / new_count
            w_old = float(self._episodes) / new_count

            self._mean_ep_len = w_new * new_mean_ep_len + w_old * self._mean_ep_len
            self._episodes += num_resets

            for key in info["rewards"]:
                new_mean_return = torch.mean(self._return_bufs[key][reset_ids])
                old_mean_return = self._mean_returns[key]
                self._mean_returns[key] = w_new * new_mean_return + w_old * old_mean_return
                self._return_bufs[key][reset_ids] = 0.0

            self._ep_len_buf[reset_ids] = 0
            self._eps_per_env_buf[reset_ids] += 1

        return