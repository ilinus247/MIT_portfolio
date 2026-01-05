import tensorflow as tf
from actor import Actor
from hyperparams import gamma, tau

Dense = tf.keras.layers.Dense
Input = tf.keras.layers.Input
Concatenate = tf.keras.layers.Concatenate
Sequential = tf.keras.models.Sequential
Model = tf.keras.models.Model


class Critic(Model):
    def __init__(self):
        super().__init__()
        self.q_func: Model | None = None

    def initialize_q_func(self, obs_dim: tuple, action_dim: tuple):
        """Initialize a neural network that outputs a Q-value"""
        obs_input = Input(obs_dim)
        action_input = Input(action_dim)
        concat = Concatenate()([obs_input, action_input])
        middle = Sequential([
            Dense(256, activation='leaky_relu'),
            Dense(256, activation='leaky_relu'),
            Dense(256, activation='leaky_relu'),
        ])(concat)
        q_layer = tf.squeeze(Dense(1)(middle), -1)
        self.q_func = Model(inputs=[obs_input, action_input], outputs=[q_layer])

    @tf.function
    def learn(self, state: tf.Tensor, action: tf.Tensor, next_state: tf.Tensor,
              done: tf.Tensor, reward: tf.Tensor, actor: Actor, target_net: 'Critic',
              log_alpha: tf.Variable, optimizer: tf.keras.optimizers.Optimizer, use_entropy = True):
        """Train Critic as defined in the paper"""
        alpha = tf.exp(log_alpha)
        with tf.GradientTape() as tape:
            qval = target_net.q_func([state, action])
            means, logstd = actor.policy(next_state)
            next_action = actor.get_action(means, logstd)
            next_prob = actor.get_log_prob(means, logstd, next_action)
            target = self.q_func([next_state, next_action])
            bootstrap_value = (tf.ones_like(done) - done)
            if use_entropy:
                target = reward + gamma * bootstrap_value * (target - alpha * tf.reduce_mean(next_prob, -1))
            else:
                target = reward + gamma * bootstrap_value * target
            q_loss = tf.keras.losses.mse(qval, target)
            loss = q_loss
        grads = tape.gradient(loss, self.q_func.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.q_func.trainable_variables))

    @tf.function
    def polyak(self, critic: 'Critic'):
        """Soft update target critic parameters"""
        for learned_weight, targ_weight in zip(critic.q_func.weights, self.q_func.weights):
            targ_weight.assign((learned_weight * tau) + (targ_weight * (1 - tau)))
