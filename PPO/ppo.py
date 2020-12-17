import os
import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque

from tensorflow._api.v2.data import Dataset
from tensorflow.python.keras.backend_config import epsilon


class get_config():
    max_len = 1e4
    iterations = 1e4
    n_max_steps = 200
    n_fixed_steps = 2048
    discount_factor = 0.95
    ENV_NAME = 'Reacher-v2'
    batch_size = 64
    discount_factor = 0.95
    lamda = 0.9
    epsilon = 0.2


class ActorCritic(keras.Model):
    def __init__(self, state_space, action_space,  bound):
        '''
        # Arguments
            state_space:    状态空间的维度
            action_space:   Type:int or tuple  动作空间的维度，不同场景可能会控制多个action
            bound:          放大倍数

        '''

        self.state_space = self._to_list(state_space)
        self.action_space = self._to_list(action_space)
        # self.bound = self._to_array(bound)
        self.bound = bound

        if len(self.action_space != 1):
            raise ValueError('动作空间的维度是{}，模型只接受一维动作序列'.format(self.action_space))
        
        # if self.action_space != self.bound.shape[0]:
        #     raise ValueError('动作空间的维度是{}, bound的维度是{}，'
        #                     '每一个动作都应该有一组对应的bound'.format(action_space, bound))
        # if len(self.bound.shape) == 1:
        #     self.bound = self.bound.repeat(2).reshape(self.bound.shape[0], -1)
        # action
        ac_input = keras.layers.Flatten(input_shape=self.state_space)
        ac_layer1 = keras.layers.Dense(64)(ac_input)
        ac_layer2 = keras.layers.Dense(64)(ac_layer1)
        ac_layer3 = keras.layers.Dense(
            self.action_space[0], activation='tanh')(ac_layer2)
        # tanh函数取值范围是[-1,1]，因此，需要放大bound倍，以适应action_space 的不同取值。
        ac_mean = keras.layers.Lambda(lambda x: x * bound)(ac_layer3)
        ac_logstd = keras.layers.Dense(
            self.action_space[0], activation='softplus')(ac_layer3)

        # critic
        cr_input = keras.layers.Flatten(input_shape=self.state_space)
        cr_layer1 = keras.layers.Dense(64)(cr_input)
        cr_layer2 = keras.layers.Dense(64)(cr_layer1)
        cr_outputs = keras.layers.Dense(1)(cr_layer2)

        self.actor = keras.Model(
            inputs=ac_input, outputs=[ac_mean, ac_logstd])
        self.critic = keras.Model(inputs=cr_input, outputs=[cr_outputs])

    def call(self, state):
        ac_mean, ac_logstd = self.actor(state)
        value = self.critic(state)
        return ac_mean, ac_logstd, value

    @staticmethod
    def _normal_logproba(x, mean, logstd, std=None):
        if std is None:
            std = tf.exp(logstd)

        std_sq = tf.math.pow(std, 2)
        logproba = - 0.5 * \
            tf.math.log(2 * np.math.pi) - logstd - \
            tf.math.pow((x - mean), 2) / (2 * std_sq)
        return tf.squeeze(logproba)

    def _choose_action(self, mean, logstd, return_old_log=True):
        std = tf.math.exp(logstd)
        action = tf.random.normal([self.action_space[0]], mean, std, tf.float32)
        log_probs = ActorCritic._normal_logproba(action, mean, logstd, std)
        cliped_action = tf.clip_by_value(action, -1 * self.bound,  1 * self.bound)
        if return_old_log:
            return cliped_action, log_probs
        else:
            return cliped_action

    def _to_list(self, x):
        """Normalizes a list/tensor into a list.

        If a tensor is passed, we return
        a list of size 1 containing the tensor.

        # Arguments
            x: target object to be normalized.

        # Returns
            A list.
        """
        if isinstance(x,  list):
            return x
        return [i.__int__() for i in x]

    def _to_array(self, x):
        return np.array(x)



def ppo():

    config = get_config()
    batch_size = config.batch_size
    buffer = deque(maxlen=config.max_len)
    interations = config.iterations
    n_max_steps = config.n_max_steps
    n_fixed_steps = config.n_fixed_steps
    env = gym.make(config.ENV_NAME)
    discount_factor = config.discount_factor
    lamda = config.lamda
    epsilon = config.epsilon

    out_net = ActorCritic(env.observation_space.shape,env.action_space.shape, 1)
    new_net = ActorCritic(env.observation_space.shape,env.action_space.shape, 1)
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)


    for interation in range(interations):
        # all_values = []
        # all_rewards = []
        global_steps = 0
        out_net.set_weights(new_net.get_weights())

        for t in range(n_fixed_steps):
            state = env.reset()

            rewards = 0
            step = 0
            for step in range(n_max_steps):
                ac_mean, ac_logstd, value = out_net(state)
                action, log_prob = out_net._choose_action(ac_mean, ac_logstd)
                action = action.numpy()
                log_prob = log_prob.numpy()

                next_state, reward, done, info = env.step(action)
                mask = 0 if done else 1
                rewards += reward

                buffer.append(
                    [state, value, action, log_prob, mask, reward])

                if done:
                    break
                state = next_state

            t = step + 1

            # Update Parameters
            # Get Data

            all_states, all_values, all_actions, all_old_probs, all_masks, all_rewards = [
                [buffer[i][var_index] for i in range(len(buffer))]
                for var_index in range(6)
            ]
            all_advantages = np.zeros(shape=len(all_values))
            # tf.GradientTape Context:
            with tf.GradientTape(persistent=True) as tape:

                pre_reward = 0
                pre_value = 0
                pre_advantage = 0
                for i in reversed(range(len(buffer))):
                    delta = all_rewards[i] + discount_factor * \
                        pre_value * all_masks[i] - all_values[i]
                    all_advantages[i] = delta + discount_factor * \
                        lamda * pre_advantage * all_masks[i]
                    # compute go-to-reward
                    all_rewards[i] += discount_factor * pre_reward * all_masks[i]

                    pre_reward = all_rewards[i]
                    pre_value = all_values[i]
                    pre_advantage = all_advantages[i]

                ac_mean, ac_logstd, cr_values = new_net(all_states)
                ac_logproba = ActorCritic._normal_logproba(all_actions, ac_mean, ac_logstd)

                rio = tf.math.exp(ac_logproba / all_old_probs)
                cliped_rio = tf.clip_by_value(rio, 1-epsilon, 1+epsilon)
                loss = tf.math.minimum(rio * all_advantages, cliped_rio * all_advantages)
                ac_loss = -tf.reduce_mean(loss)

                loss = keras.losses.mean_squared_error(cr_values, all_rewards)
                cr_loss = tf.reduce_mean(loss)
            new_net.critic.trainable = False
            ac_gradients = tape.gradient(ac_loss, new_net.trainable_variables)
            optimizer.apply_gradients(zip(ac_gradients, new_net.trainable_variables))
            new_net.actor.trainable = False
            new_net.critic.trainable = True
            cr_gradients = tape.gradient(cr_loss, new_net.trainable_variables)
            optimizer.apply_gradients(zip(cr_gradients, new_net.trainable_variables))
            del tape

        
