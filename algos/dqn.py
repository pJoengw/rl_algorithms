import numpy as np
import tensorflow as tf
from tensorflow import keras
import gym
from collections import deque
import os


# Visualize 
root_logdir = os.path.join(os.curdir, 'my_logs')
def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


env = gym.make('CartPole-v0')

# fixed random seed
env.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# config
[state_size] = env.observation_space.shape
action_space = env.action_space.n
buffer = deque(maxlen=2000)
n_max_step = 200
batch_size = 32
optimizer = keras.optimizers.Adam(lr=0.001)
loss_fn = keras.losses.mean_squared_error
discount_factor = 0.95

model = keras.models.Sequential([
    keras.layers.Dense(32, activation='elu', input_shape=[state_size]),
    keras.layers.Dense(32, activation='elu'),
    keras.layers.Dense(action_space)
])


def greedy_epsilon_action(state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(2)
    else:
        # NOTE: model接受数据[batch_size, ***]
        # 所以输出以后应该指定维度[0] 表示当前 batch
        return np.argmax(model.predict(state[np.newaxis])[0])


# sample one batch data
def sample(batch_size):
    indicates = np.random.randint(len(buffer), size=batch_size)
    batch = [buffer[index] for index in indicates]
    states, actions, rewards, newstates, dones = [
        np.array([experience[var_index] for experience in batch])
        for var_index in range(5)
    ]
    return states, actions, rewards, newstates, dones


def play_one_step(env: gym.Env, obs, epsilon):
    action = greedy_epsilon_action(obs, epsilon)
    newstate, reward, done, info = env.step(action)
    buffer.append([obs, action, reward, newstate, done])
    return newstate, reward, done, info

'''
NOTE: 
        train_step目标是优化 model 对于当前状态s的预测值
        使其接近 reward + γ * Q(s',a')
        所以不应该把 Q(s',a')的计算过程带入 tf.GrandientTape
'''
def train_step(batch_size):
    states, actions, rewards, newstates, dones = sample(batch_size)
    Q_pre = np.max(model(newstates), axis=1)
    target = rewards + discount_factor * Q_pre
    mask = tf.one_hot(actions, depth=action_space)
    with tf.GradientTape() as tape:
        Q_values_s = model(states)
        Q_value = tf.reduce_sum(Q_values_s * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(Q_value, target))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

loss_logdir = get_run_logdir()
writer = tf.summary.create_file_writer(loss_logdir)

import heapq

with writer.as_default():
    best_score = 0
    rewards = []
    for episode in range(600):
        episode_reward = 0.0
        obs = env.reset()
        reward_episode = 0
        for step in range(200):
            epsilon = max(1-episode / 500, 0.01)
            obs, reward, done, _ = play_one_step(env, obs, epsilon)
            reward_episode += reward
            if done:
                break
        heapq.heappush(rewards, reward_episode)
        if reward_episode > best_score:
            best_score = reward_episode
            best_weight = model.get_weights()
        tf.summary.scalar("reward", reward_episode,step=episode)
        if episode > 50:
            train_step(batch_size)
    model.set_weights(best_weight)
    top5 = sum(heapq.nlargest(5, rewards)) / 5
    top1 = heapq.nlargest(1, rewards)
    model.save("./models/model_top5_{}_top1_{}.h5".format(top5, top1))
