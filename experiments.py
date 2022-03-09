import numpy as np

import gym

import experience_store
import agent
import network
import simulation

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
    task = gym.make('CartPole-v1')

    sim = simulation.Simulation(ag, task, 100000, 1.0, 0.1, 10000)
    sim.run(False)

def test_TD0_Agent():
    rng = np.random.default_rng(5)
    task_rng = np.random.default_rng(6)

    obs_shape = (1,)
    action_count = 2

    Q_network = network.FFANN(obs_shape, action_count, [20, 10], 0.0001)
    discount_factor = 0.99
    experience_buffer_size = 100000
    training_samples_per_experience_step = 1024
    minibatch_size = 1024
    experience_period_length = 2048
    target_Q_network_update_rate = 0.00000001

    ag = agent.TD0Agent(rng, obs_shape, action_count, Q_network, discount_factor, experience_buffer_size, training_samples_per_experience_step, minibatch_size, experience_period_length, target_Q_network_update_rate)
    # task = gym.make('CartPole-v1')
    task = TrivialTask(task_rng)

    sim = simulation.Simulation(ag, task, 100000, 1.0, 0.1, 100000)
    sim.run(False)

if __name__ == '__main__':
    test_TD0_Agent()