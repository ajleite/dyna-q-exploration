import pickle

import numpy as np
import tensorflow as tf

import gym

import pybullet_data
import pybulletgym

import network

def downscale(image):
    new_image = np.zeros((image.shape[0]//2, image.shape[1]//2, image.shape[2]), image.dtype)
    for new_row in range(new_image.shape[0]):
        old_row = new_row * 2
        for new_col in range(new_image.shape[1]):
            old_col = new_col * 2
            new_image[new_row, new_col] = image[old_row, old_col]
    return new_image

def preprocess(image):
    bg = image[0, -1]

    # restrict to the playing field
    field = image[17:97]

    ball_color = np.uint8([[[236, 236, 236]]])
    P1_color = np.uint8([[[92, 186, 92]]])
    P2_color = np.uint8([[[213, 130, 74]]])

    # disentangle features insofar as possible
    is_ball = np.all(field == ball_color, axis=-1)
    is_P1 = np.all(field == P1_color, axis=-1)
    is_P2 = np.all(field == P2_color, axis=-1)
    new_image = np.float32(np.stack([is_ball, is_P1, is_P2], axis=2))

    return new_image

def load_CNN():
    obs_shape = (80, 80, 6)
    action_count = 2

    def network_factory(input):
        exceptional_cues = tf.keras.layers.GlobalMaxPool2D()(tf.keras.layers.Conv2D(12, 5, padding='same', activation='relu')(input)) # params: 6x5x5x12+12, size: 12
        local_dynamics_cues = tf.keras.layers.Conv2D(12, 3, padding='same', activation='relu')(input) # params: 6x3x3x12+12, size: 80x80x12
        coarse_dynamics = tf.keras.layers.MaxPool2D(pool_size=(8, 8), padding='same')(local_dynamics_cues) # size: 10x10x12
        locationwise_coarse_dynamics = tf.keras.layers.Flatten()(coarse_dynamics) # size: 1200
        all_cues = tf.keras.layers.Concatenate(axis=-1)([locationwise_coarse_dynamics, exceptional_cues]) # size: 1212

        linear_decisions_1 = tf.keras.layers.Dense(50, activation='relu')(all_cues) # params: 1212x50+50, size: 50
        linear_decisions_2 = tf.keras.layers.Dense(20, activation='relu')(linear_decisions_1) # params: 50x20+20, size: 20

        return linear_decisions_2

    n = network.CNN(obs_shape, action_count, network_factory, 0.0004)

    p = pickle.load(open(f'out/MC-Pong-1.pickle','rb'))
    n.keras_network.set_weights(p['best_weights'])

    return n

class PongTaskWithCNN:
    def __init__(self, rng):
        self.pong_env = gym.make('PongNoFrameskip-v0')
        self.obs_buffer = [None, None, None]
        self.obs_index = -3
        self.points = 0

        self.rng = rng
        self.CNN = load_CNN()

        self.real_reset()
    def real_reset(self):
        self.obs_index = -3
        self.points = 0

        self.pong_env.seed(int(self.rng.integers(2**31)))
        raw_obs = self.pong_env.reset()
        obs = preprocess(downscale(raw_obs))
        self.obs_buffer[self.obs_index] = obs
        self.obs_index += 1

        return self.CNN.apply_headless(np.concatenate([obs, obs], axis=2))

    def reset(self):
        if self.obs_index >= 0:
            self.obs_buffer[0] = self.obs_buffer[self.obs_index-1]
            self.obs_index = -3
        return self.CNN.apply_headless(np.concatenate([self.obs_buffer[0], self.obs_buffer[0]], axis=2))

    def step(self, action):
        if action == 0:
            action = 2
        else:
            action = 5
        raw_obs, reward, terminal, info = self.pong_env.step(action)

        if terminal:
            if not reward:
                if self.points < 0:
                    reward = -1.
                else:
                    reward = 1.
    
            combined_obs = self.real_reset()
        else:
            obs = preprocess(downscale(raw_obs))

            if self.obs_index < 0:
                old_obs = self.obs_buffer[0]
            else:
                old_obs = self.obs_buffer[self.obs_index]

            self.obs_buffer[self.obs_index] = obs
            self.obs_index += 1
            if self.obs_index >= 3:
                self.obs_index = 0

            combined_obs = self.CNN.apply_headless(np.concatenate([obs, old_obs], axis=2))

            if reward:
                terminal = 1
                self.points += reward

        return combined_obs, reward, terminal, info
    def render(self):
        view = self.pong_env.render()



class CartPoleTask:
    def __init__(self):
        self.cartpole_env = gym.make("InvertedPendulumMuJoCoEnv-v0")
    def reset(self):
        return self.cartpole_env.reset()
    def step(self, action):
        return self.cartpole_env.step(((-1.,), (1.,))[action])
    def render(self):
        return self.cartpole_env.render()

class TrivialTask:
    def __init__(self, rng):
        self.rng = rng
        self.reset()
    def step(self, action):
        orig_lt0 = self.loc < 0
        if action:
            self.loc += .1
        else:
            self.loc -= .1

        if self.loc <= -1 or self.loc >= 1:
            return (self.loc, -1, True, None)

        now_lt0 = self.loc < 0
        if orig_lt0 and not now_lt0 or now_lt0 and not orig_lt0:
            return (self.loc, 1, True, None)

        return (np.array((self.loc,)), 0, False, None)
    def reset(self):
        self.loc = self.rng.random()*2-1
        return np.array((self.loc,))
