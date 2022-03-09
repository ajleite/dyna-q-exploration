import numpy as np

class Simulation:
    def __init__(self, agent, task, num_episodes, epsilon_init, epsilon_final=None, epsilon_final_timestep=None):
        self.agent = agent
        self.task = task

        self.num_episodes = num_episodes

        self.loss_samples = []
        self.episode_rewards = []

        self.epsilon_init = epsilon_init
        if not epsilon_final is None and not epsilon_final_timestep is None:
            self.epsilon_final = epsilon_final
            self.epsilon_final_timestep = epsilon_final_timestep
        else:
            self.epsilon_final = epsilon_init
            self.epsilon_final_timestep = 1

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
                    print('Loss: ', n, timestep, mean_loss)
                    print(n_left, n_right)
                    n_left = 0
                    n_right = 0
                    if last_ep_index < len(self.episode_rewards):
                        print(np.mean(self.episode_rewards[last_ep_index:],axis=0)[1])
                        last_ep_index = len(self.episode_rewards)

                    epsilon_coordinate = timestep / self.epsilon_final_timestep
                    if epsilon_coordinate > 1:
                        epsilon_coordinate = 1

                    epsilon = self.epsilon_init * (1 - epsilon_coordinate) + self.epsilon_final * epsilon_coordinate

                    # print(self.agent.Q_network.keras_network(np.linspace(-1, 1, 11).reshape(-1, 1)))

                    # import matplotlib.pyplot as plt
                    # S = self.agent.experience_buffer.S_samples[self.agent.experience_buffer.cur_index-1000:self.agent.experience_buffer.cur_index]
                    # A = self.agent.experience_buffer.A_samples[self.agent.experience_buffer.cur_index-1000:self.agent.experience_buffer.cur_index]
                    # Q = self.agent.experience_buffer.R_samples[self.agent.experience_buffer.cur_index-1000:self.agent.experience_buffer.cur_index]
                    # plt.scatter(S[A == 0], Q[A == 0], alpha=.1)
                    # plt.scatter(S[A == 1], Q[A == 1], alpha=.1)
                    # plt.show()


                total_r += r
                timestep += 1

                if render:
                    self.task.render()

            self.episode_rewards.append((timestep, total_r))