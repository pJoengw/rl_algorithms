import os
import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque


class get_config():
    max_len = 1e4
    iterations = 1e4
    n_max_steps = 200
    n_fixed_steps = 2048
    discount_factor = 0.95
    ENV_NAME = 'CartPole-v0'


class ActorCritic(keras.Model):
    def __init__(self, num_actions, bound):
        self.num_actions = num_actions
        self.bound = bound
        # action
        ac_layer1 = keras.layers.Dense(64)
        ac_layer2 = keras.layers.Dense(64)(ac_layer1)
        ac_layer3 = keras.layers.Dense(
            num_actions, activation='tanh')(ac_layer2)
        ac_mean = keras.layers.Lambda(lambda x: x * bound)(ac_layer3)
        ac_logstd = keras.layers.Dense(
            num_actions, dtype=tf.float32)(ac_layer2)

        # critic
        cr_layer1 = keras.layers.Dense(64)
        cr_layer2 = keras.layers.Dense(64)(cr_layer1)
        cr_outputs = keras.layers.Dense(1)(cr_layer2)

        self.actor = keras.Model(
            inputs=ac_layer1, outputs=[ac_mean, ac_logstd])
        self.critic = keras.Model(inputs=cr_layer1, outputs=[cr_outputs])

    def call(self, state):
        ac_mean, ac_logstd = self.actor(state)
        value = self.critic(state)
        # action, old_logprobs = _choose_action(ac_mean, ac_logstd)
        return ac_mean, ac_logstd, value

    def _choose_action(self, mean, logstd, return_old_log = True):
        std = tf.math.exp(logstd)
        action = tf.random.normal([self.num_actions], mean, std, tf.float32)
        log_probs = _normal_logproba(action, mean, logstd, std)
        cliped_action = tf.clip_by_value(action, -1 * bound,  1 * bound)
        if return_old_log:
            return cliped_action, log_probs
        else:
            return cliped_action

    @staticmethod
    def _normal_logproba(x, mean, logstd, std=None):
        if std is None:
            std = tf.exp(logstd)

        std_sq = tf.math.pow(std, 2)
        logproba = - 0.5 * tf.math.log(2 * np.math.pi) - logstd - tf.math.pow((x - mean),2) / (2 * std_sq)
        return tf.squeeze(logproba)


def ppo():

    config = get_config()
    buffer = deque(maxlen=config.max_len)
    interations = config.iterations
    n_max_steps = config.n_max_steps
    n_fixed_steps = config.n_fixed_steps
    env = gym.make(config.ENV_NAME)
    ac_net = ActorCritic(env.action_space.shape[0], 2)
    for interation in range(interations):
        all_values = []
        all_rewards = []
        global_steps = 0
        for _ in range(n_fixed_steps):
            obs = env.reset()

            rewards = 0
            for step in range(n_max_steps):
                ac_mean, ac_logstd, value= ac_net(obs)
                action, log_prob = ac_net._choose_action(ac_mean, ac_logstd)
                action = action.numpy()[0]  # action.numpy() 返回的虽然只有一个值，但仍是数组形式
                log_prob = log_prob.numpy()[0]

                next_state, reward, done, info = env.step(action)
                mask = 0 if done else 1
                rewards += reward

                buffer.append([obs, value, action, log_prob, mask, next_state, mask])

                if done:
                    break
                state=next_state
            
