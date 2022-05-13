import pickle

import numpy as np

class Simulation:
    def __init__(self, agent, task, num_episodes, epsilon=0.25, path=None):
        self.agent = agent
        self.task = task

        self.num_episodes = num_episodes
        self.epsilon = epsilon
        self.path = path

        self.training_episode_rewards = []
        self.eval_episode_rewards = []
        self.eval_indices = []

        self.episode_behavior_entropy = []
        self.episode_dyn_loss = []
        self.episode_Q_loss = []

        self.best_weights = None
        self.best_100_episode_return = None

    def save_trace(self):
        if not self.path:
            return

        to_save = {'best_weights': self.best_weights, 'best_100_episode_return': self.best_100_episode_return,
            'training_episode_rewards': self.training_episode_rewards, 'eval_episode_rewards': self.eval_episode_rewards,
            'eval_indices': self.eval_indices,
            'dyn_loss': self.episode_dyn_loss, 'Q_loss': self.episode_Q_loss, 'behavior_entropy': self.episode_behavior_entropy}

        pickle.dump(to_save, open(self.path,'wb'))

    def evaluate(self, eval_index=None, render=False):
        eval_rewards = []
        for i in range(50):
            s = self.task.reset()
            t = False
            ep_return = 0

            while not t:
                a = self.agent.act(s, 0)

                s2, r, t, _ = self.task.step(a)
                s = s2
                ep_return += r

                if render:
                    self.task.render()

            # episode is over, record it
            eval_rewards.append(ep_return)
            print((eval_index, 'e', i, ep_return))

        self.eval_indices.append(eval_index)
        self.eval_episode_rewards.append(eval_rewards)

        # maintain best stats
        if self.best_100_episode_return is None or np.mean(eval_rewards) > self.best_100_episode_return:
            self.best_100_episode_return = np.mean(eval_rewards)
            self.best_weights = self.agent.Q_network.keras_network.get_weights()
        print('cycle', eval_index, 'mean eval:', np.mean(eval_rewards), 'best:', self.best_100_episode_return)
        self.save_trace()

    def run(self, render=False):
        timestep = 0

        for n in range(self.num_episodes):
            # gather greedy evaluation trajectories every 50 training episodes
            if not n % 50:
                self.evaluate(n, render)

            # 1. gather training trajectories
            s = self.task.reset()
            t = False
            total_r = 0

            n_left = 0
            n_right = 0

            ep_Q_loss = 0
            ep_Q_loss_den = 0
            ep_dyn_loss = 0
            ep_dyn_loss_den = 0

            while not t:
                a = self.agent.act(s, self.epsilon)

                if a == 0:
                    n_left += 1
                else:
                    n_right += 1

                s2, r, t, _ = self.task.step(a)
                mean_Q_loss, mean_dyn_loss = self.agent.store(s, a, r, t, s2)
                s = s2
                total_r += r

                if not mean_Q_loss is None:
                    ep_Q_loss += mean_Q_loss
                    ep_Q_loss_den += 1
                if not mean_dyn_loss is None:
                    ep_dyn_loss += mean_dyn_loss
                    ep_dyn_loss_den += 1

            # episode is over, record it
            self.training_episode_rewards.append(total_r)
            print((n, 't', total_r))

            # calculate behavior entropy
            total_actions = n_left + n_right
            p_left = n_left / total_actions
            p_right = n_right / total_actions
            if p_left == 0 or p_right == 0:
                entropy = 0.
            else:
                entropy = -np.log2(p_left)*p_left + -np.log2(p_right)*p_right
            self.episode_behavior_entropy.append(entropy)

            if ep_Q_loss_den:
                self.episode_Q_loss.append(ep_Q_loss/ep_Q_loss_den)
                print(ep_Q_loss/ep_Q_loss_den)
            if ep_dyn_loss_den:
                self.episode_dyn_loss.append(ep_dyn_loss/ep_dyn_loss_den)
                print(ep_dyn_loss/ep_dyn_loss_den)

        # everything is done!
        self.evaluate(self.num_episodes, render)
