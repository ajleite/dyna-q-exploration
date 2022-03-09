import numpy as np

import gym

import experience_store
import agent
import network
import simulation
import tasks

def test_MC_Agent():
    agent_rng = np.random.default_rng(5)
    task_rng = np.random.default_rng(6)

    obs_shape = (4,)
    action_count = 2

    Q_network = network.FFANN(obs_shape, action_count, [20, 10], 0.0001)
    discount_factor = 0.99
    experience_buffer_size = 100000
    training_samples_per_experience_step = 256
    minibatch_size = 1024
    experience_period_length = 8192

    ag = agent.MonteCarloAgent(agent_rng, obs_shape, action_count, Q_network, discount_factor, experience_buffer_size, training_samples_per_experience_step, minibatch_size, experience_period_length)
    task = tasks.CartPoleTask()

    sim = simulation.Simulation(ag, task, 100000, 1.0, 0.1, 10000)
    sim.run(False)

def test_TD0_Agent():
    rng = np.random.default_rng(5)
    task_rng = np.random.default_rng(6)

    obs_shape = (4,)
    action_count = 2

    Q_network = network.FFANN(obs_shape, action_count, [20, 10], 0.0001)
    discount_factor = 0.99
    experience_buffer_size = 100000
    training_samples_per_experience_step = 2048
    minibatch_size = 1024
    experience_period_length = 512
    target_Q_network_update_rate = 0.000001

    ag = agent.TD0Agent(rng, obs_shape, action_count, Q_network, discount_factor, experience_buffer_size, training_samples_per_experience_step, minibatch_size, experience_period_length, target_Q_network_update_rate)
    # task = gym.make('CartPole-v1')
    # task = TrivialTask(task_rng)
    task = tasks.CartPoleTask()

    sim = simulation.Simulation(ag, task, 100000, 1.0, 0.1, 100000)
    sim.run(True)

def test_DQN_Agent():
    rng = np.random.default_rng(5)
    task_rng = np.random.default_rng(6)

    obs_shape = (4,)
    action_count = 2

    Q_network = network.FFANN(obs_shape, action_count, [20, 10], 0.0001)
    discount_factor = 0.99
    experience_buffer_size = 100000
    training_samples_per_experience_step = 2048
    minibatch_size = 1024
    experience_period_length = 1
    target_Q_network_update_rate = 0.00001

    ag = agent.TD0Agent(rng, obs_shape, action_count, Q_network, discount_factor, experience_buffer_size, training_samples_per_experience_step, minibatch_size, experience_period_length, target_Q_network_update_rate)
    # task = gym.make('CartPole-v1')
    # task = tasks.TrivialTask(task_rng)
    task = tasks.CartPoleTask()

    sim = simulation.Simulation(ag, task, 100000, 1.0, 0.1, 100000)
    sim.run(False)

def test_MC_Agent_Pong():
    agent_rng = np.random.default_rng(5)
    task_rng = np.random.default_rng(6)

    obs_shape = (80, 80, 6)
    action_count = 2

    Q_network = network.CNN(obs_shape, action_count, [tf.keras.layers.Conv2D(8, 3, padding='same', activation='relu')], 0.0001)
    discount_factor = 0.99
    experience_buffer_size = 100000
    training_samples_per_experience_step = 256
    minibatch_size = 1024
    experience_period_length = 8192

    ag = agent.MonteCarloAgent(agent_rng, obs_shape, action_count, Q_network, discount_factor, experience_buffer_size, training_samples_per_experience_step, minibatch_size, experience_period_length)
    task = tasks.CartPoleTask()

    sim = simulation.Simulation(ag, task, 100000, 1.0, 0.1, 10000)
    sim.run(False)

if __name__ == '__main__':
    test_DQN_Agent()
