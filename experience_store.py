import numpy as np

import util

# Unfortunately, it is not reasonable to use a single programmatical method for both the Monte Carlo trajectory buffer and the TD(0) experience buffer,
# but the two methods can share an interface.

# The buffers expect observations to be float32s normalized to the range [0, 1].
# Please note that the buffers receive int32 action indices as input,
# but produce one-hot encodings corresponding to these indices as output.

class MonteCarloBuffer:
    def __init__(self, obs_shape, action_count, buffer_size, discount_factor):
        self.S_samples = np.zeros((buffer_size,)+obs_shape, dtype=np.float32)
        self.A_samples = np.zeros((buffer_size,), dtype=np.int32)
        self.Q_samples = np.zeros((buffer_size,), dtype=np.float32)

        self.action_count = action_count

        # buffer_size represents the size of the buffer.
        # cur_index represents the next index that will be written.
        # filled represents whether the buffer has been filled at least once (can be sampled freely).
        self.buffer_size = buffer_size
        self.cur_index = 0
        self.filled = False

        self.discount_factor = discount_factor

        self.episode_buffer = []

    def store_episode(self, observations, actions, rewards):
        ''' store_episode should be called at the end of an episode.

            observations, actions, and rewards should be numpy arrays
            whose shapes align along their first axis (timestep axis).

            - observations should be a float32 array whose subsequent axes match obs_shape.
            - actions should be an int32 array with no subsequent axes.
            - rewards should be a float32 array with no subsequent axis,
              and it should represent the rewards for the transitions
              following the (observation, action) pairs in the corresponding indices. '''

        trajectory_length = rewards.shape[0]

        # discard unusable information ASAP
        if trajectory_length > self.buffer_size:
            observations = observations[-self.buffer_size:]
            actions = actions[-self.buffer_size:]
            rewards = rewards[-self.buffer_size:]
            trajectory_length = self.buffer_size

        # calculate trajectory rewards for each timestep
        # (ensure double precision for this calculation)
        traj_rewards = np.array(rewards, dtype=np.float64)
        for offset in range(1, trajectory_length):
            traj_rewards[:-offset] += rewards[offset:] * self.discount_factor**offset
        traj_rewards = np.float32(traj_rewards)
        # print(traj_rewards)

        # store the trajectory in the buffer
        # (split based on whether we will wrap around the end of the buffer)
        will_loop = self.cur_index + trajectory_length >= self.buffer_size
        if will_loop:
            can_store = self.buffer_size - self.cur_index
            self.S_samples[self.cur_index:] = observations[:can_store]
            self.A_samples[self.cur_index:] = actions[:can_store]
            self.Q_samples[self.cur_index:] = traj_rewards[:can_store]

            leftover = trajectory_length - can_store
            if leftover:
                self.S_samples[:leftover] = observations[can_store:]
                self.A_samples[:leftover] = actions[can_store:]
                self.Q_samples[:leftover] = traj_rewards[can_store:]

            self.filled = True
            self.cur_index = (self.cur_index + trajectory_length) - self.buffer_size
        else:
            new_index = self.cur_index + trajectory_length

            self.S_samples[self.cur_index:new_index] = observations
            self.A_samples[self.cur_index:new_index] = actions
            self.Q_samples[self.cur_index:new_index] = traj_rewards

            self.cur_index = new_index

    def store(self, obs, action, reward, terminal):
        ''' This is a convenience function to allow the MonteCarloBuffer to act more like the TD0Buffer.
            It matches the signature of TD0Bufer.store. '''

        self.episode_buffer.append((obs, action, reward))

        # This is an unusual idiom that allows one to effectively take the transpose of a Python list.
        # I often use it in RL contexts.
        if terminal:
            all_S, all_A, all_R = [np.array(all_samples) for all_samples in zip(*self.episode_buffer)]
            self.store_episode(all_S, all_A, all_R)
            self.episode_buffer = []

    def sample_target_values(self, batch_size, rng):
        ''' samples `batch_size` samples using the numpy random generator `rng`.

            returns them as a tuple (observations, actions, qualities),
            where actions is now one-hot encoded. '''

        if self.filled:
            limit_index = self.buffer_size
        else:
            limit_index = self.cur_index

        sample_indices = rng.integers(limit_index, size=(batch_size))

        return self.S_samples[sample_indices], util.make_one_hot(self.A_samples[sample_indices], self.action_count), self.Q_samples[sample_indices]

    def clear(self):
        self.cur_index = 0
        self.filled = False

        self.episode_buffer = []


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

    def store(self, obs, action, reward, terminal):
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

        sample_indices = rng.integers(limit_index, size=(batch_size))

        if self.filled and avoid_last_stored_sample and self.cur_index != 0:
            sample_indices = np.where(sample_indices >= self.cur_index - 1, sample_indices + 1, sample_indices)

        # construct the indices for S2 by adding 1 to the existing sample indices, being sure to wrap around.
        next_indices = np.where(sample_indices == self.buffer_size - 1, 0, sample_indices + 1)

        return self.S_samples[sample_indices], util.make_one_hot(self.A_samples[sample_indices], self.action_count), \
            self.R_samples[sample_indices], self.T_samples[sample_indices], self.S_samples[next_indices]

    def clear(self):
        self.cur_index = 0
        self.filled = False
