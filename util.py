import numpy as np

def make_one_hot(action_indices, action_count):
    sample_size = action_indices.shape[0]

    one_hot_actions = np.zeros((sample_size, action_count), dtype=np.float32)
    expanded_action_indices = np.expand_dims(action_indices, axis=1)

    np.put_along_axis(one_hot_actions, expanded_action_indices, 1., axis=1)

    return one_hot_actions
