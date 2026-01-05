import gymnasium as gym
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
from buffer import ReplayBuffer
from functions import explore, metrics, learn_alpha
from actor import Actor
from critic import Critic
from hyperparams import actor_lr, critic_lr, alpha_lr
tf.compat.v1.logging.set_verbosity(40)

# random seed
tf.random.set_seed(12)

# Initialize environments, networks, buffer, and optimizers
env = gym.make('HalfCheetah-v4')
test_env = gym.make('HalfCheetah-v4')

obs_shape = env.observation_space.shape
action_shape = env.action_space.shape


actor = Actor(action_bound=1)

critic_1 = Critic()
critic_2 = Critic()

target_critic_1 = Critic()
target_critic_2 = Critic()

log_alpha = tf.Variable(0.2, dtype=tf.float32)

buffer = ReplayBuffer()

actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
critic_1_optimizer = tf.keras.optimizers.Adam(critic_lr)
critic_2_optimizer = tf.keras.optimizers.Adam(critic_lr)
alpha_optimizer = tf.keras.optimizers.Adam(alpha_lr)

alpha_optimizer.build([log_alpha])

actor.initialize_policy(obs_shape, action_shape[0], action_shape[0])
critic_1.initialize_q_func(obs_shape, action_shape)
critic_2.initialize_q_func(obs_shape, action_shape)

target_critic_1.q_func = tf.keras.models.clone_model(critic_1.q_func)
target_critic_2.q_func = tf.keras.models.clone_model(critic_2.q_func)

# train models with checkpointing and collect metrics
collected_metrics = []
state = env.reset()[0]
try:
    for iteration in range(1, 5001):
        state = explore(state, 1000, env, actor, buffer)
        for _ in tqdm(range(256), desc=f'Learning iteration {iteration}'):
            (states, actions, next_states, rewards, dones) = buffer.sample_batch()
            actor.learn(states, critic_1.q_func, critic_2.q_func, log_alpha, actor_optimizer)
            critic_1.learn(states, actions, next_states, dones, rewards,
                           actor, target_critic_1, log_alpha, critic_1_optimizer)
            critic_2.learn(states, actions, next_states, dones, rewards,
                           actor, target_critic_2, log_alpha, critic_2_optimizer)
            target_critic_1.polyak(critic_1)
            target_critic_2.polyak(critic_2)
            learn_alpha(states, actions, log_alpha, actor, action_shape[0], alpha_optimizer)
        if iteration % 5 == 0:
            cp_num = int(iteration / 5)
            actor.policy.save(f'cp/cp_{cp_num}/actor.h5')
            critic_1.q_func.save(f'cp/cp_{cp_num}/critic_1.h5')
            critic_2.q_func.save(f'cp/cp_{cp_num}/critic_2.h5')
            target_critic_1.q_func.save(f'cp/cp_{cp_num}/target_critic_1.h5')
            target_critic_2.q_func.save(f'cp/cp_{cp_num}/target_critic_2.h5')
            with open('saved_alpha', 'a') as f:
                f.write(f'\nepoch {str(iteration)}:{str(log_alpha.numpy())}')
        if iteration % 5 == 0:
            evaluation = metrics(actor, test_env)
            collected_metrics += [evaluation]
            print(evaluation, tf.exp(log_alpha).numpy())
except KeyboardInterrupt:
    print('Interrupted')

plt.plot(range(len(collected_metrics)), [metric[0] for metric in collected_metrics])
plt.plot(range(len(collected_metrics)), [metric[1] for metric in collected_metrics])
plt.plot(range(len(collected_metrics)), [metric[2] for metric in collected_metrics])
plt.legend(['Environment reward', 'Entropy objective return', 'Episode length'])
plt.show()
