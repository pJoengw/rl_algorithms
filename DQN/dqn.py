import numpy as np
import tensorflow as tf
from tensorflow import keras
import gym
from collections import deque

env = gym.make('CartPole-v0')

# config
[state_size] = env.observation_space.shape
action_space = env.action_space.n
buffer_size = 1e3
buffer = deque(maxlen=buffer_size)
n_max_step = 200
batch_size = 32
optimizer = keras.optimizers.Adam(lr=0.01)
loss_fn = keras.losses.mean_squared_error
discount_factor = 0.95

model = keras.models.Sequential([
    keras.layers.Dense(32, activation='relu', input_shape=[state_size]),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(action_space)
])


def greedy_epsilon_action(state, epsilon):
    if np.random.randn() < epsilon:
        return np.random.randint(2)
    else:
        return np.argmax(model.predict(state))


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
        Q_values_s = model.predict(states)
        Q_value = tf.reduce_sum(Q_values_s * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(Q_value, target))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


for episode in range(600):
    obs = env.reset()
    for step in range(n_max_step):
        epsilon = max(1-episode / 600, 0.01)
        obs, reward, done, _ = play_one_step(env, obs, epsilon)
        if done:
            break
    if episode > 50:
        train_step(batch_size)
