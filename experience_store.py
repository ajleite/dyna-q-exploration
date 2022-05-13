import numpy as np

import network
import util

# The buffers expect observations to be float32s normalized to the range [0, 1].
# Please note that the buffers receive int32 action indices as input,
# but produce one-hot encodings corresponding to these indices as output.


class TD0Buffer:
    def __init__(self, obs_shape, action_count, buffer_size):
        self.S_samples = np.zeros((buffer_size,)+obs_shape, dtype=np.float32)
        self.A_samples = np.zeros((buffer_size,), dtype=np.int32)
        self.R_samples = np.zeros((buffer_size,), dtype=np.float32)
        self.T_samples = np.zeros((buffer_size,), dtype=bool)

        self.action_count = action_count

        # buffer_size represents the size of the buffer.
        # cur_index represents the next index that will be written.
        # filled represents whether the buffer has been filled at least once (can be sampled freely).
        self.buffer_size = buffer_size
        self.cur_index = 0
        self.filled = False

    def store(self, obs, action, reward, terminal, obs_2):
        ''' store should be called every timestep.


            - obs should be a float32 array whose axes match obs_shape.
            - action should be an int or int32.
            - reward should be a float32.
            - terminal should be a boolean.

            Unless terminal was True for the last call to store,
            this sample must correspond to the timestep immediately
            following the sample passed to the last call. '''

        self.S_samples[self.cur_index] = obs
        self.A_samples[self.cur_index] = action
        self.R_samples[self.cur_index] = reward
        self.T_samples[self.cur_index] = terminal

        self.cur_index += 1

        if self.cur_index == self.buffer_size:
            self.filled = True
            self.cur_index = 0

    def sample_SARTS2(self, batch_size, rng):
        ''' Samples `batch_size` samples using the numpy random generator `rng`.

            Returns them as a tuple (observations, actions, rewards, terminals, next observations),
            where actions is now one-hot encoded.

            Refuses to return the most recently stored tuple (since there is nothing to follow it) unless it was terminal.
            Remember not to interpret next observation if terminal flag was set. '''

        if not self.T_samples[self.cur_index-1]:
            avoid_last_stored_sample = True
        else:
            avoid_last_stored_sample = False

        # the strategy for avoiding the last stored sample is as follows:
        # just don't generate it, if we have not looped.
        # if have, then add 1 to all indices at or above it to avoid it.

        if self.filled and avoid_last_stored_sample:
            limit_index = self.buffer_size - 1
        elif self.filled:
            limit_index = self.buffer_size
        elif avoid_last_stored_sample:
            limit_index = self.cur_index - 1
        else:
            limit_index = self.cur_index

        if limit_index > 0:
            sample_indices = rng.integers(limit_index, size=(batch_size))
        else:
            sample_indices = np.zeros((0,), dtype=np.int32)

        if self.filled and avoid_last_stored_sample and self.cur_index != 0:
            sample_indices = np.where(sample_indices >= self.cur_index - 1, sample_indices + 1, sample_indices)

        # construct the indices for S2 by adding 1 to the existing sample indices, being sure to wrap around.
        next_indices = np.where(sample_indices == self.buffer_size - 1, 0, sample_indices + 1)

        return self.S_samples[sample_indices], self.A_samples[sample_indices], \
            self.R_samples[sample_indices], self.T_samples[sample_indices], self.S_samples[next_indices]

    def clear(self):
        self.cur_index = 0
        self.filled = False

class HybridBuffer:
    ''' This buffer stores all of the S, A pairs verbatim, but uses a dynamics network to obtain R, T, and S2. '''
    def __init__(self, obs_shape, action_count, hidden_layer_sizes, learning_rate, buffer_size):
        self.S_samples = np.zeros((buffer_size,)+obs_shape, dtype=np.float32)
        self.A_samples = np.zeros((buffer_size,), dtype=np.int32)

        self.action_count = action_count

        self.dynann = network.DynaNN(obs_shape, action_count, hidden_layer_sizes, learning_rate)

        # buffer_size represents the size of the buffer.
        # cur_index represents the next index that will be written.
        # filled represents whether the buffer has been filled at least once (can be sampled freely).
        self.buffer_size = buffer_size
        self.cur_index = 0
        self.filled = False

    def store(self, obs, action, reward, terminal, obs_2):
        ''' store should be called every timestep.


            - obs should be a float32 array whose axes match obs_shape.
            - action should be an int or int32.
            - reward should be a float32.
            - terminal should be a boolean.

            Unless terminal was True for the last call to store,
            this sample must correspond to the timestep immediately
            following the sample passed to the last call. '''

        self.S_samples[self.cur_index] = obs
        self.A_samples[self.cur_index] = action

        self.cur_index += 1

        if self.cur_index == self.buffer_size:
            self.filled = True
            self.cur_index = 0

        pre = self.dynann.apply(obs, action)
        loss = self.dynann.fit(obs, action, reward, terminal, obs_2)
        post = self.dynann.apply(obs, action)

        # print(reward, terminal, obs_2.numpy())
        # print(*(i.numpy() for i in pre))
        # print(*(i.numpy() for i in post))

        return loss

    def sample_SARTS2(self, batch_size, rng):
        ''' Samples `batch_size` samples using the numpy random generator `rng`.

            Returns them as a tuple (observations, actions, rewards, terminals, next observations),
            where actions is now one-hot encoded.

            Refuses to return the most recently stored tuple (since there is nothing to follow it) unless it was terminal.
            Remember not to interpret next observation if terminal flag was set. '''

        avoid_last_stored_sample = True

        # the strategy for avoiding the last stored sample is as follows:
        # just don't generate it, if we have not looped.
        # if have, then add 1 to all indices at or above it to avoid it.

        if self.filled and avoid_last_stored_sample:
            limit_index = self.buffer_size - 1
        elif self.filled:
            limit_index = self.buffer_size
        elif avoid_last_stored_sample:
            limit_index = self.cur_index - 1
        else:
            limit_index = self.cur_index

        if limit_index > 0:
            sample_indices = rng.integers(limit_index, size=(batch_size))
        else:
            sample_indices = np.zeros((0,), dtype=np.int32)

        if self.filled and avoid_last_stored_sample and self.cur_index != 0:
            sample_indices = np.where(sample_indices >= self.cur_index - 1, sample_indices + 1, sample_indices)

        # construct the indices for S2 by adding 1 to the existing sample indices, being sure to wrap around.
        next_indices = np.where(sample_indices == self.buffer_size - 1, 0, sample_indices + 1)

        S, A = self.S_samples[sample_indices], self.A_samples[sample_indices]
        R, T, S2 = self.dynann.apply(S, A)
        T = T > 0.5

        return S, A, R, T, S2

    def clear(self):
        self.cur_index = 0
        self.filled = False