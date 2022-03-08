# I don't usually use keras but it seems simpler than doing things directly,
# and I'm not trying to do anything unusual like batch a whole population of
# networks, so I'll take that route for this assignment.

import tensorflow as tf

class FFANN:
    def __init__(self, obs_shape, action_count, hidden_layer_sizes, learning_rate, action_layer = 0):
        self.obs_shape = obs_shape
        self.action_count = action_count
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.action_layer = action_layer

        obs_input = tf.keras.layers.Input(shape=obs_shape)
        action_input = tf.keras.layers.Input(shape=(action_count,))

        next_input = obs_input
        for hidden_layer, hidden_layer_size in enumerate(hidden_layer_sizes):
            if hidden_layer == action_layer:
                next_input = tf.keras.layers.Concatenate(axis=-1)([next_input, action_input])

            next_input = tf.keras.layers.Dense(hidden_layer_size, activation='sigmoid', kernel_initializer='random_normal', bias_initializer='random_normal')(next_input)

        linear_output = tf.keras.layers.Dense(1)(next_input)

        self.keras_network = tf.keras.Model([obs_input, action_input], linear_output)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    def apply(self, S, A):
        return self.keras_network([S, A])

    @tf.function
    def fit(self, S, A, Q):
        Q_predicted = self.apply(S, A)
        # tf.print(S[0], A[0], Q[0], Q_predicted[0])
        Q_loss = tf.reduce_sum((Q_predicted - Q) ** 2)

        Q_gradient = tf.gradients(Q_loss, self.keras_network.weights)
        self.optimizer.apply_gradients(zip(Q_gradient, self.keras_network.weights))

        return Q_loss

    def copy_from(self, other, amount):
        for self_w, other_w in zip(other.keras_network.weights, self.keras_network.weights):
            self_w.assign(self_w*(1-amount) + other_w*amount)

    def copy(self):
        other = FFANN(self.obs_shape, self.action_count, self.hidden_layer_sizes, self.learning_rate, action_layer=self.action_layer)
        other.copy_from(self, 1)
        return other
