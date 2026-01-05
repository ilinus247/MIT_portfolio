import gymnasium as gym
import tensorflow as tf
from tqdm import tqdm

from actor import Actor

# Load and run agent
tf.random.set_seed(12)
env = gym.make('HalfCheetah-v4', render_mode='human')
actor = Actor(1)
actor.policy = tf.keras.models.load_model('cp/cp_14/actor.h5')
for _ in range(1):
    obs = env.reset()[0]
    done = False
    with tqdm(total=1000, desc='Rendering') as bar:
        while not done:
            obs = tf.expand_dims(obs, 0)
            means, logstd = actor.policy(obs)
            action = actor.get_action(means.numpy(), logstd.numpy())
            action = actor.squash_action(action)
            obs, reward, terminated, truncated, _ = env.step(action[0])
            done = terminated or truncated
            bar.update(1)
