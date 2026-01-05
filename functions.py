import gymnasium as gym
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from buffer import ReplayBuffer
from actor import Actor


def run_env_step(env: gym.Env, state: np.ndarray, actor: Actor):
    """Run a step in the environment using the actor policy"""
    _state = np.expand_dims(state, 0)
    means, logstd = actor.policy(_state)
    action = actor.get_action(means, logstd)[0]
    squashed_action = actor.squash_action(action)
    next_state, reward, terminated, truncated, _ = env.step(squashed_action)
    done = terminated
    _next_state = env.reset()[0] if (terminated or truncated) else next_state
    return state, action, next_state, reward, done, _next_state


def explore(state, steps: int, env: gym.Env, actor: Actor, buffer: ReplayBuffer):
    """Collect simulation data by running through the environment"""
    for _ in tqdm(range(steps), desc='Exploring'):
        state, action, next_state, reward, done, _next_state = run_env_step(env, state, actor)
        buffer.write(state, action, next_state, reward, done)
        state = _next_state
    return state


def metrics(actor: Actor, env: gym.Env, num_episodes: int = 5):
    """Metrics computation"""
    total_reward = 0
    total_entropy_reward = 0
    total_length = 0
    for _ in range(num_episodes):
        obs = env.reset()[0]
        episode_length = 0
        episode_reward = 0
        episode_entropy_reward = 0
        terminated, truncated = False, False
        with tqdm(total=1000, desc='Evaluating') as bar:
            while not (terminated or truncated):
                episode_length += 1
                obs = tf.expand_dims(obs, 0)
                means, logstd = actor.policy(obs)
                action = actor.get_action(means, logstd)
                action = actor.squash_action(action)
                action = action[0]
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                bar.update(1)
        total_entropy_reward += episode_entropy_reward
        total_length += episode_length
        total_reward += episode_reward
    return total_reward / num_episodes, total_entropy_reward / num_episodes, total_length / num_episodes


@tf.function
def learn_alpha(state: tf.Tensor, action: tf.Tensor, log_alpha: tf.Variable, actor: Actor,
                action_dim: int, alpha_optimizer: tf.keras.optimizers.Optimizer):
    """Adjust entropy temperature via gradient updates"""
    with tf.GradientTape() as tape:
        means, logstd = actor.policy(state)
        log_prob = actor.get_log_prob(means, logstd, action)
        alpha_loss = -tf.exp(log_alpha) * (tf.reduce_mean(log_prob, -1) - action_dim)
    grad = tape.gradient(alpha_loss, log_alpha)
    alpha_optimizer.update_step(grad, log_alpha)
