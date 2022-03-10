import numpy as np

import experience_store
import util


class BaseAgent:
    def __init__(self, rng, action_count, Q_network, experience_buffer, training_samples_per_experience_step, minibatch_size, experience_period_length, target_Q_network_update_rate):
        self.rng = rng

        self.action_count = action_count
        self.possible_actions = np.arange(self.action_count)

        self.Q_network = Q_network
        self.target_Q_network = self.Q_network.zero_like()
        self.target_Q_network_update_rate = target_Q_network_update_rate

        self.experience_buffer = experience_buffer

        self.training_samples_per_experience_step = training_samples_per_experience_step
        self.minibatch_size = minibatch_size
        self.experience_period_length = experience_period_length

        self.experience_period_step = 0

        self.use_tqdm = False

    def sample_target_values(self, minibatch_size):
        raise NotImplementedError()

    def act(self, obs, epsilon=0.1):
        if epsilon and self.rng.random() < epsilon:
            return self.rng.integers(self.action_count)

        return np.array(self.Q_network.apply_A(np.expand_dims(obs, axis=0))[0])

    def store(self, obs, action, reward, terminal):
        self.experience_buffer.store(obs, action, reward, terminal)
        self.experience_period_step += 1

        if self.experience_period_step == self.experience_period_length:
            total_training_samples = self.training_samples_per_experience_step * self.experience_period_length
            num_minibatches =  total_training_samples // self.minibatch_size
            last_minibatch_size = total_training_samples % self.minibatch_size

            total_loss = 0

            if self.use_tqdm:
                import tqdm
                minibatches = tqdm.tqdm(range(num_minibatches))
            else:
                minibatches = range(num_minibatches)

            for _ in minibatches:
                S, A, Q = self.sample_target_values(self.minibatch_size)
                if S.size > 0:
                    total_loss += self.Q_network.fit(S, A, Q)
                    # print(num_minibatches, _, total_loss)

            if last_minibatch_size:
                S, A, Q = self.sample_target_values(last_minibatch_size)
                if S.size > 0:
                    total_loss += self.Q_network.fit(S, A, Q)

            # if S.size > 0:
            #     print(S[0], A[0], Q[0])
            mean_loss = np.sqrt(np.array(total_loss)) / total_training_samples

            if self.target_Q_network_update_rate:
                total_target_update = 1 - (1-self.target_Q_network_update_rate)**total_training_samples
                self.target_Q_network.copy_from(self.Q_network, amount=total_target_update)

            # print(self.Q_network.keras_network(np.linspace(-1, 1, 11).reshape(-1, 1)))

            self.experience_period_step = 0

            return mean_loss

        return None

class MonteCarloAgent(BaseAgent):
    def __init__(self, rng, obs_shape, action_count, Q_network, discount_factor, experience_buffer_size, training_samples_per_experience_step, minibatch_size, experience_period_length):
        experience_buffer = experience_store.MonteCarloBuffer(obs_shape, action_count, experience_buffer_size, discount_factor)
        target_Q_network_update_rate = 0
        BaseAgent.__init__(self, rng, action_count, Q_network, experience_buffer, training_samples_per_experience_step, minibatch_size, experience_period_length, target_Q_network_update_rate)

    def sample_target_values(self, minibatch_size):
        vals = self.experience_buffer.sample_target_values(minibatch_size, self.rng)
        # print(vals[2][0])
        return vals


def get_TD0_target_values(S, A, R, T, S2, target_V_function, discount_factor):
    ''' Used to get a target value for the TD0 algorithm. '''

    sample_count = S.shape[0]

    ## generate the Q2 values
    Q2 = np.array(target_V_function(S2))

    # set the future reward to 0 when the transition is terminal
    Q2 = np.where(T, 0, Q2)

    ## apply the Bellman equation to derive the target Q1 values
    return R + discount_factor * Q2

class TD0Agent(BaseAgent):
    def __init__(self, rng, obs_shape, action_count, Q_network, discount_factor, experience_buffer_size, training_samples_per_experience_step, minibatch_size, experience_period_length, target_Q_network_update_rate):
        experience_buffer = experience_store.TD0Buffer(obs_shape, action_count, experience_buffer_size)
        self.discount_factor = discount_factor
        BaseAgent.__init__(self, rng, action_count, Q_network, experience_buffer, training_samples_per_experience_step, minibatch_size, experience_period_length, target_Q_network_update_rate)

    def sample_target_values(self, minibatch_size):
        S, A, R, T, S2 = self.experience_buffer.sample_SARTS2(minibatch_size, self.rng)
        if S.size > 0:
            Q = get_TD0_target_values(S, A, R, T, S2, self.target_Q_network.apply_V, self.discount_factor)
        else:
            Q = np.zeros((), dtype=np.float32)

        return S, A, Q
