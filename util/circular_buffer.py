import torch

class CircularBuffer():
    def __init__(self, batch_size, buffer_len, shape, dtype, device):
        self._buffer = torch.zeros([batch_size, buffer_len] + list(shape), dtype=dtype, device=device)
        self._head = 0
        return

    def get_batch_size(self):
        return self._buffer.shape[0]

    def get_buffer_len(self):
        return self._buffer.shape[1]

    def push(self, data):
        self._buffer[:, self._head, ...] = data
        n = self.get_buffer_len()
        self._head = (self._head + 1) % n
        return

    def fill(self, batch_idx, data):
        buffer_len = self.get_buffer_len()
        self._buffer[batch_idx, self._head:, ...] = data[:, :buffer_len - self._head, ...]
        self._buffer[batch_idx, :self._head, ...] = data[:, buffer_len - self._head:, ...]
        return

    def get(self, idx):
        n = self.get_buffer_len()
        assert(idx >= 0 and idx < n)
        data = self._buffer[:, idx, ...]
        return data

    def get_all(self):
        if (self._head == 0):
            data = self._buffer
        else:
            data_beg = self._buffer[:, self._head:, ...]
        
            n = self.get_buffer_len()
            end_idx = (self._head + n) % n
            data_end = self._buffer[:, 0:end_idx, ...]

            data = torch.cat([data_beg, data_end], dim=1)
        return data

    def reset(self):
        self._head = 0
        return