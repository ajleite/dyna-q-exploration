import numpy as np

class Simulation:
    def __init__(self, agent, task, num_episodes):
        self.agent = agent
        self.task = task

        self.num_episodes = num_episodes

        self.loss_samples = []
        self.episode_rewards = []

    def run(self, render=False):
        timestep = 0

        n_left = 0
        n_right = 0
        last_ep_index = 0

        for n in range(self.num_episodes):
            s = self.task.reset()
            t = False
            total_r = 0

            last_a = None
            hold_steps = 0
            while not t:
                if hold_steps == 0:
                    a = self.agent.act(s, (250)/(n+1) + 0.1)
                    last_a = a
                    hold_steps = 0
                else:
                    a = last_a
                    hold_steps -= 1

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
                    import matplotlib.pyplot as plt
                    S = self.agent.experience_buffer.S_samples[self.agent.experience_buffer.cur_index-1000:self.agent.experience_buffer.cur_index]
                    A = self.agent.experience_buffer.A_samples[self.agent.experience_buffer.cur_index-1000:self.agent.experience_buffer.cur_index]
                    Q = self.agent.experience_buffer.Q_samples[self.agent.experience_buffer.cur_index-1000:self.agent.experience_buffer.cur_index]
                    plt.scatter(S[A == 0], Q[A == 0], alpha=.1)
                    plt.scatter(S[A == 1], Q[A == 1], alpha=.1)
                    plt.show()


                total_r += r
                timestep += 1

                if render:
                    self.task.render()

            self.episode_rewards.append((timestep, total_r))