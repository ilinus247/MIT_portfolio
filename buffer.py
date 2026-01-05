from typing import SupportsFloat

from hyperparams import max_buffer_size, batch_size
import tensorflow as tf
import numpy as np


class ReplayBuffer:
    """A simple Replay Buffer for Soft Actor-Critic"""
    def __init__(self, max_size=max_buffer_size):
        self.max_size = max_size
        self.states = tf.TensorArray(tf.float32, max_size)
        self.actions = tf.TensorArray(tf.float32, max_size)
        self.next_states = tf.TensorArray(tf.float32, max_size)
        self.rewards = tf.TensorArray(tf.float32, max_size)
        self.dones = tf.TensorArray(tf.float32, max_size)
        self.times_written = 0
        self.current_write_index = 0

    def write(self, state: np.ndarray, action: tf.Tensor, next_state: np.ndarray, reward: SupportsFloat, done: bool):
        self.states = self.states.write(self.current_write_index, state)
        self.actions = self.actions.write(self.current_write_index, action)
        self.next_states = self.next_states.write(self.current_write_index, next_state)
        self.rewards = self.rewards.write(self.current_write_index, reward)
        self.dones = self.dones.write(self.current_write_index, done)
        # Overwrite the oldest entries to make space for new ones
        self.times_written += 1
        self.current_write_index += 1
        self.current_write_index %= self.max_size

    def sample_batch(self, size=batch_size):
        upper_bound = min(self.times_written, self.max_size)
        indices = np.random.randint(0, upper_bound, [size])
        return self.states.gather(indices), self.actions.gather(indices), self.next_states.gather(indices), \
            self.rewards.gather(indices), self.dones.gather(indices)
