import pickle

import numpy as np
import tensorflow as tf

import gym

import experience_store
import agent
import network
import simulation
import tasks

def pong_config(task_rng):
    # use this to train a CNN
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

    task = tasks.PongTask(task_rng)

    Q_network = network.CNN(obs_shape, action_count, network_factory, 0.0004)

    task_name = 'Pong'

    return task_name, obs_shape, action_count, Q_network, task

def pong_PostCNN_config(task_rng):
    obs_shape = (20,)
    action_count = 2

    Q_network = network.FFANN(obs_shape, action_count, [50, 20], 0.0001)

    task = tasks.PongTaskWithCNN(task_rng)

    return 'Pong-PostCNN', obs_shape, action_count, Q_network, task


def test_DQN_Agent(seed, config, *args, replay=False, render=False, episodes=2500, **kwargs):
    agent_rng = np.random.default_rng(seed)
    task_rng = np.random.default_rng(seed+234579672983459873)

    task_name, obs_shape, action_count, Q_network, task = config(task_rng, *args, **kwargs)

    discount_factor = 0.99
    experience_buffer_size = 100000
    training_samples_per_experience_step = 2048
    minibatch_size = 512
    experience_period_length = 1
    target_Q_network_update_rate = 0.00001

    if replay:
        experience_period_length = -1
        target_Q_network_update_rate = 0

    ag = agent.TD0Agent(agent_rng, obs_shape, action_count, Q_network, discount_factor, experience_buffer_size, training_samples_per_experience_step, minibatch_size, experience_period_length, target_Q_network_update_rate)

    path = f'out/DQN-{task_name}-{seed}.pickle'

    sim = simulation.Simulation(ag, task, episodes, 0.25, path=path)

    if replay:
        p = pickle.load(open(path,'rb'))
        ag.Q_network.keras_network.set_weights(p['best_weights'])
        sim.path = None
        sim.evaluate(render=True)
    else:
        sim.run(render)

    return sim

def test_DynaQ_Agent(seed, config, *args, replay=False, render=False, episodes=2500, **kwargs):
    agent_rng = np.random.default_rng(seed)
    task_rng = np.random.default_rng(seed+234579672983459873)

    task_name, obs_shape, action_count, Q_network, task = config(task_rng, *args, **kwargs)

    discount_factor = 0.99
    experience_buffer_size = 100000
    training_samples_per_experience_step = 2048
    minibatch_size = 512
    experience_period_length = 1
    target_Q_network_update_rate = 0.00001

    if replay:
        experience_period_length = -1
        target_Q_network_update_rate = 0

    ag = agent.DynaQAgent(agent_rng, obs_shape, action_count, Q_network, discount_factor, experience_buffer_size, training_samples_per_experience_step, minibatch_size, experience_period_length, target_Q_network_update_rate)

    path = f'out/DynaQ-{task_name}-{seed}.pickle'

    sim = simulation.Simulation(ag, task, episodes, 0.25, path=path)

    if replay:
        p = pickle.load(open(path,'rb'))
        ag.Q_network.keras_network.set_weights(p['best_weights'])
        sim.path = None
        sim.evaluate(render=True)
    else:
        sim.run(render)

    return sim

if __name__ == '__main__':
    import sys
    a = sys.argv[1]
    t = sys.argv[2]
    seed = int(sys.argv[3])
    if len(sys.argv) > 4 and sys.argv[4] == '-r':
        replay = True
    else:
        replay = False

    if t == 'pong':
        tas = pong_PostCNN_config
    elif t == 'pong-conv':
        tas = lambda rng: pong_config(rng, boring_network=True)
    elif t == 'cartpole':
        tas = cart_pole_config
    else:
        print('invalid task')
        sys.exit()

    if a == 'DynaQ':
        test_DynaQ_Agent(seed, tas, replay=replay)
    elif a == 'DQN':
        test_DQN_Agent(seed, tas, replay=replay)
    else:
        print('invalid agent')
