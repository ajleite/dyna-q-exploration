import numpy as np
import tensorflow as tf

import gym

import experience_store
import agent
import network
import simulation
import tasks

def cart_pole_config(task_rng):
    obs_shape = (4,)
    action_count = 2

    Q_network = network.FFANN(obs_shape, action_count, [20, 10], 0.0001)

    task = tasks.CartPoleTask()

    return obs_shape, action_count, Q_network, task

def pong_config(task_rng, boring_network=False):
    obs_shape = (80, 80, 6)
    action_count = 2

    def network_factory(input):
        exceptional_cues = tf.keras.layers.GlobalMaxPool2D()(tf.keras.layers.Conv2D(12, 5, padding='same', activation='relu')(input)) # params: 6x5x5x12+12, size: 12
        local_dynamics_cues = tf.keras.layers.Conv2D(6, 3, padding='same', activation='relu')(input) # params: 6x3x3x6+6, size: 80x80x6
        coarse_dynamics = tf.keras.layers.MaxPool2D(pool_size=(8, 8), padding='same')(local_dynamics_cues) # size: 10x10x6
        locationwise_coarse_dynamics = tf.keras.layers.Flatten()(coarse_dynamics) # size: 600
        all_cues = tf.keras.layers.Concatenate(axis=-1)([locationwise_coarse_dynamics, exceptional_cues]) # size: 612

        linear_decisions_1 = tf.keras.layers.Dense(50, activation='relu')(all_cues) # params: 612x50+50, size: 50
        linear_decisions_2 = tf.keras.layers.Dense(20, activation='relu')(linear_decisions_1) # params: 50x20+20, size: 20

        return linear_decisions_2

    def boring_network_factory(input):
        conv1 = tf.keras.layers.Conv2D(24, 3, padding='same', activation='relu')(input) # params: 6x3x3x24+24, size: 80x80x24
        pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')(conv1) # size: 40x40x24
        conv2 = tf.keras.layers.Conv2D(384, 4, padding='same', activation='relu')(pool1) # params: 24x4x4x384+384, size: 40x40x384
        pool2 = tf.keras.layers.MaxPool2D(pool_size=(4, 4), padding='same')(conv2) # size: 10x10x384
        conv3 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(pool2) # params: 384x3x3x128+128, size: 10x10x128

        linear_decisions_1 = tf.keras.layers.Dense(200, activation='relu')(conv3) # params: 128x10x10x200+200, size: 200
        linear_decisions_2 = tf.keras.layers.Dense(100, activation='relu')(linear_decisions_1) # params: 200x100+100, size: 100
        linear_decisions_3 = tf.keras.layers.Dense(10, activation='relu')(linear_decisions_2) # params: 100x10+10, size: 10

        return linear_decisions_2

    if boring_network:
        network_factory = boring_network_factory

    task = tasks.PongTask(task_rng)

    Q_network = network.CNN(obs_shape, action_count, network_factory, 0.0004)

    return obs_shape, action_count, Q_network, task

def test_MC_Agent(seed):
    agent_rng = np.random.default_rng(seed)
    task_rng = np.random.default_rng(seed+234579672983459873)

    obs_shape, action_count, Q_network, task = cart_pole_config(task_rng)

    discount_factor = 0.99
    experience_buffer_size = 100000
    training_samples_per_experience_step = 256
    minibatch_size = 1024
    experience_period_length = 8192

    ag = agent.MonteCarloAgent(agent_rng, obs_shape, action_count, Q_network, discount_factor, experience_buffer_size, training_samples_per_experience_step, minibatch_size, experience_period_length)

    sim = simulation.Simulation(ag, task, 4000, 1.0, 0.1, 10000, path=f'MC-CartPole-{seed}.pickle')
    sim.run(False)

def test_FQI_Agent(seed):
    agent_rng = np.random.default_rng(seed)
    task_rng = np.random.default_rng(seed+234579672983459873)

    obs_shape, action_count, Q_network, task = cart_pole_config(task_rng)

    discount_factor = 0.99
    experience_buffer_size = 100000
    training_samples_per_experience_step = 2048
    minibatch_size = 1024
    experience_period_length = 512
    target_Q_network_update_rate = 0.000001

    ag = agent.TD0Agent(agent_rng, obs_shape, action_count, Q_network, discount_factor, experience_buffer_size, training_samples_per_experience_step, minibatch_size, experience_period_length, target_Q_network_update_rate)

    sim = simulation.Simulation(ag, task, 4000, 1.0, 0.1, 100000, path=f'FQI-CartPole-{seed}.pickle')
    sim.run(True)

def test_DQN_Agent(seed):
    agent_rng = np.random.default_rng(seed)
    task_rng = np.random.default_rng(seed+234579672983459873)

    obs_shape, action_count, Q_network, task = cart_pole_config(task_rng)

    discount_factor = 0.99
    experience_buffer_size = 100000
    training_samples_per_experience_step = 2048
    minibatch_size = 1024
    experience_period_length = 1
    target_Q_network_update_rate = 0.00001

    ag = agent.TD0Agent(agent_rng, obs_shape, action_count, Q_network, discount_factor, experience_buffer_size, training_samples_per_experience_step, minibatch_size, experience_period_length, target_Q_network_update_rate)

    sim = simulation.Simulation(ag, task, 4000, 1.0, 0.1, 100000, path=f'DQN-CartPole-{seed}.pickle')
    sim.run(False)

def test_MC_Agent_Pong(seed):
    agent_rng = np.random.default_rng(seed)
    task_rng = np.random.default_rng(seed+234579672983459873)

    obs_shape, action_count, Q_network, task = pong_config(task_rng)

    discount_factor = 0.99
    experience_buffer_size = 100000
    training_samples_per_experience_step = 64
    minibatch_size = 512
    experience_period_length = 8192

    ag = agent.MonteCarloAgent(agent_rng, obs_shape, action_count, Q_network, discount_factor, experience_buffer_size, training_samples_per_experience_step, minibatch_size, experience_period_length)
    ag.use_tqdm = True

    sim = simulation.Simulation(ag, task, 100000, 1.0, 0.1, 10000, path=f'MC-Pong-{seed}.pickle')
    sim.run(False)

if __name__ == '__main__':
    test_MC_Agent_Pong(1)
