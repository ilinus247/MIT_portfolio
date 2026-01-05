import tensorflow as tf
import tensorflow_probability as tfp

Dense = tf.keras.layers.Dense
Input = tf.keras.layers.Input
Concatenate = tf.keras.layers.Concatenate
Sequential = tf.keras.models.Sequential
Model = tf.keras.models.Model
generator = tf.random.get_global_generator()


class Actor(Model):
    def __init__(self, action_bound: float):
        super().__init__()
        self.policy: Model | None = None
        # Gym environments require squashing
        self.action_bound = action_bound

    def initialize_policy(self, input_dim: tuple, means_dim: int, logstd_dim: int):
        """Initialize a neural network that outputs the parameters for a Gaussian distribution"""
        input_layer = Input(input_dim)
        middle = Sequential([
            Dense(256, activation='leaky_relu'),
            Dense(256, activation='leaky_relu'),
            Dense(256, activation='leaky_relu'),
        ])(input_layer)
        means = tf.keras.layers.Dense(means_dim)(middle)
        logstd = tf.clip_by_value(tf.keras.layers.Dense(logstd_dim)(middle), -20, 2)
        self.policy = Model(inputs=[input_layer], outputs=[means, logstd])

    @staticmethod
    def get_action(means: tf.Tensor, logstd: tf.Tensor):
        epsilon = generator.normal(logstd.shape)
        std = tf.exp(logstd)
        return means + epsilon * std

    def squash_action(self, action: tf.Tensor):
        return tf.nn.tanh(action) * self.action_bound

    @staticmethod
    def get_log_prob(means, logstd, unbounded_action):
        std = tf.exp(logstd)
        dist = tfp.distributions.Normal(means, std)
        log_prob = tf.math.log(dist.prob(unbounded_action) + 1e-6)
        # Correct logprobs as defined in the paper
        log_prob_correction = tf.reduce_sum(tf.math.log(1 - tf.math.square(tf.nn.tanh(unbounded_action)) + 1e-6),
                                            axis=-1)
        log_prob -= tf.reshape(tf.repeat(log_prob_correction, log_prob.shape[-1], axis=-1), [-1, log_prob.shape[-1]])
        return log_prob

    @tf.function
    def learn(self, state: tf.Tensor, critic_1: Model,
              critic_2: Model, log_alpha: tf.Variable, optimizer: tf.keras.optimizers.Optimizer, use_entropy=True):
        """Train Actor as defined in the paper"""
        alpha = tf.exp(log_alpha)
        with tf.GradientTape() as tape:
            means, logstd = self.policy(state)
            action = self.get_action(means, logstd)
            q_1, q_2 = critic_1([state, action]), critic_2([state, action])
            min_q = tf.minimum(q_1, q_2)
            entropy_term = alpha * tf.reduce_mean(self.get_log_prob(means, logstd, action))
            if use_entropy:
                loss = entropy_term - min_q
            else:
                loss = - min_q
        grads = tape.gradient(loss, self.policy.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.policy.trainable_variables))
