# I don't usually use keras but it seems simpler than doing things directly,
# and I'm not trying to do anything unusual like batch a whole population of
# networks, so I'll take that route for this assignment.

import tensorflow as tf

class FFANN:
    def __init__(self, obs_shape, action_count, hidden_layer_sizes, learning_rate):
        self.obs_shape = obs_shape
        self.action_count = action_count
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate

        obs_input = tf.keras.layers.Input(shape=obs_shape)

        next_input = obs_input
        for hidden_layer, hidden_layer_size in enumerate(hidden_layer_sizes):
            next_input = tf.keras.layers.Dense(hidden_layer_size, activation='relu', kernel_initializer='random_normal', bias_initializer='random_normal')(next_input)

        linear_output = tf.keras.layers.Dense(action_count)(next_input)

        self.keras_network = tf.keras.Model(obs_input, linear_output)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    @tf.function
    def apply(self, S, A):
        return tf.gather(self.keras_network(S), A, batch_dims=1)

    @tf.function
    def apply_V(self, S):
        return tf.reduce_max(self.keras_network(S), axis=1)

    @tf.function
    def apply_A(self, S):
        a = tf.argmax(self.keras_network(S), axis=1)
        return a

    @tf.function
    def fit(self, S, A, Q):
        Q_predicted = self.apply(S, A)
        # tf.print(S[0], A[0], Q[0], Q_predicted[0])
        Q_loss = tf.reduce_sum((Q_predicted - Q) ** 2)

        Q_gradient = tf.gradients(Q_loss, self.keras_network.weights)
        self.optimizer.apply_gradients(zip(Q_gradient, self.keras_network.weights))

        return Q_loss

    def copy_from(self, other, amount):
        for self_w, other_w in zip(self.keras_network.weights, other.keras_network.weights):
            self_w.assign(self_w*(1-amount) + other_w*amount)

    def copy(self):
        other = FFANN(self.obs_shape, self.action_count, self.hidden_layer_sizes, self.learning_rate)
        other.copy_from(self, 1)
        return other

    def zero_like(self):
        other = FFANN(self.obs_shape, self.action_count, self.hidden_layer_sizes, self.learning_rate)
        for other_w in other.keras_network.weights:
            other_w.assign(tf.zeros_like(other_w))
        return other

class CNN:
    def __init__(self, obs_shape, action_count, network_factory, learning_rate):
        self.obs_shape = obs_shape
        self.action_count = action_count
        self.network_factory = network_factory
        self.learning_rate = learning_rate

        obs_input = tf.keras.layers.Input(shape=obs_shape)

        last_layer = network_factory(obs_input)

        linear_output = tf.keras.layers.Dense(action_count)(last_layer)

        self.keras_network = tf.keras.Model(obs_input, linear_output)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    @tf.function
    def apply(self, S, A):
        return tf.gather(self.keras_network(S), A, batch_dims=1)

    @tf.function
    def apply_V(self, S):
        return tf.reduce_max(self.keras_network(S), axis=1)

    @tf.function
    def apply_A(self, S):
        a = tf.argmax(self.keras_network(S), axis=1)
        return a

    @tf.function
    def fit(self, S, A, Q):
        Q_predicted = self.apply(S, A)
        # tf.print(S[0], A[0], Q[0], Q_predicted[0])
        Q_loss = tf.reduce_sum((Q_predicted - Q) ** 2)

        Q_gradient = tf.gradients(Q_loss, self.keras_network.weights)
        self.optimizer.apply_gradients(zip(Q_gradient, self.keras_network.weights))

        return Q_loss

    def copy_from(self, other, amount):
        for self_w, other_w in zip(self.keras_network.weights, other.keras_network.weights):
            self_w.assign(self_w*(1-amount) + other_w*amount)

    def copy(self):
        other = FFANN(self.obs_shape, self.action_count, self.hidden_layer_sizes, self.learning_rate)
        other.copy_from(self, 1)
        return other

    def zero_like(self):
        other = FFANN(self.obs_shape, self.action_count, self.hidden_layer_sizes, self.learning_rate)
        for other_w in other.keras_network.weights:
            other_w.assign(tf.zeros_like(other_w))
        return other
