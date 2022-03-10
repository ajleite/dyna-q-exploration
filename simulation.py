import pickle

import numpy as np

class Simulation:
    def __init__(self, agent, task, num_episodes, epsilon_init, epsilon_final=None, epsilon_final_timestep=None, path=None):
        self.agent = agent
        self.task = task

        self.num_episodes = num_episodes

        self.loss_samples = []
        self.episode_rewards = []
        self.episode_behavior_entropy = []

        self.epsilon_init = epsilon_init
        if not epsilon_final is None and not epsilon_final_timestep is None:
            self.epsilon_final = epsilon_final
            self.epsilon_final_timestep = epsilon_final_timestep
        else:
            self.epsilon_final = epsilon_init
            self.epsilon_final_timestep = 1

        self.best_weights = None
        self.best_100_episode_return = None
        self.path = path

    def save_trace(self):
        if not self.path:
            return

        to_save = {'best_weights': self.best_weights, 'best_100_episode_return': self.best_100_episode_return,
            'episode_rewards': self.episode_rewards, 'loss_samples': self.loss_samples, 'episode_behavior_entropy': self.episode_behavior_entropy}

        pickle.dump(to_save, open(self.path,'wb'))

    def run(self, render=False):
        timestep = 0

        n_left = 0
        n_right = 0
        last_ep_index = 0

        epsilon = self.epsilon_init

        for n in range(self.num_episodes):
            s = self.task.reset()
            t = False
            total_r = 0

            last_a = None
            hold_steps = 0
            while not t:
                a = self.agent.act(s, epsilon)

                if a == 0:
                    n_left += 1
                else:
                    n_right += 1

                s2, r, t, _ = self.task.step(a)
                # ar = -1 if t else 0
                ar = r
                mean_loss = self.agent.store(s, a, ar, t)
                s = s2

                if not mean_loss is None:
                    self.loss_samples.append((timestep, mean_loss))
                    # print('Loss: ', n, timestep, mean_loss)

                    epsilon_coordinate = timestep / self.epsilon_final_timestep
                    if epsilon_coordinate > 1:
                        epsilon_coordinate = 1

                    epsilon = self.epsilon_init * (1 - epsilon_coordinate) + self.epsilon_final * epsilon_coordinate

                    # print(self.agent.Q_network.keras_network(np.linspace(-1, 1, 11).reshape(-1, 1)))

                    # import matplotlib.pyplot as plt
                    # S = self.agent.experience_buffer.S_samples[self.agent.experience_buffer.cur_index-1000:self.agent.experience_buffer.cur_index]
                    # A = self.agent.experience_buffer.A_samples[self.agent.experience_buffer.cur_index-1000:self.agent.experience_buffer.cur_index]
                    # T = self.agent.experience_buffer.T_samples[self.agent.experience_buffer.cur_index-1000:self.agent.experience_buffer.cur_index]
                    # Q = self.agent.experience_buffer.R_samples[self.agent.experience_buffer.cur_index-1000:self.agent.experience_buffer.cur_index]
                    # Q += self.agent.target_Q_network.apply_V(S) * self.agent.discount_factor
                    # plt.subplot(1, 2, 1)
                    # plt.scatter(S[A == 0][:,2], S[A == 0][:,3], c=Q[A == 0], alpha=.1)
                    # plt.subplot(1, 2, 2)
                    # plt.scatter(S[A == 1][:,2], S[A == 1][:,3], c=Q[A == 1], alpha=.1)
                    # plt.colorbar()
                    # plt.show()


                total_r += r
                timestep += 1

                if timestep % 2000 == 0:
                    if self.episode_rewards and self.loss_samples:
                        print('episode', n, 'last episode:', 'reward', self.episode_rewards[-1], 'loss', self.loss_samples[-1], 'entropy', self.episode_behavior_entropy[-1])
                    self.save_trace()

                if render:
                    self.task.render()

            # episode is over, record it
            self.episode_rewards.append((timestep, total_r))

            # calculate behavior entropy
            total_actions = n_left + n_right
            p_left = n_left / total_actions
            p_right = n_right / total_actions
            if p_left == 0 or p_right == 0:
                entropy = 0.
            else:
                entropy = -np.log2(p_left)*p_left + -np.log2(p_right)*p_right
            self.episode_behavior_entropy.append((timestep, entropy))
            n_left = 0
            n_right = 0

            # calculate running average
            if n >= 100:
                mean_reward = np.mean(self.episode_rewards[-100:], axis=0)[1]
                if self.best_100_episode_return is None or mean_reward > self.best_100_episode_return:
                    self.best_100_episode_return = mean_reward
                    self.best_weights = self.agent.Q_network.keras_network.get_weights()
                if not n % 100:
                    print('mean reward for episode', n-100, 'to', n, mean_reward)
                    print('best is', self.best_100_episode_return)
        
        # everything is done!
        self.save_trace()
