import os
import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import os
from tensorflow.python.types.core import Value
import tensorflow_probability as tfp

tfd = tfp.distributions


# Visualize 
root_logdir = os.path.join(os.curdir, 'my_logs')
def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

class get_config():
    max_len = 1000

    n_episodes = 300
    n_fixed_steps = 2048
    n_max_steps = 2048
    batch_size = 2048
    repeat_size = 6

    discount_factor = 0.995
    ENV_NAME = 'Hopper-v2'
    lamda = 0.97
    epsilon = 0.2


class ActorCritic(keras.Model):
    def __init__(self, state_space, action_space,  bound, actor_union):
        '''
        # Arguments
            state_space:    Type:int 状态空间的维度
            action_space:   Type:int 动作空间的维度，不同场景可能会控制多个action
            bound:          放大倍数

        '''
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.bound = bound

        input_ = keras.layers.Input(shape=[self.state_space])
        # Actor
        ac_layer1 = keras.layers.Dense(64)(input_)
        ac_layer2 = keras.layers.Dense(64)(ac_layer1)
        ac_layer3 = keras.layers.Dense(
            self.action_space, activation='tanh')(ac_layer2)
        # tanh函数取值范围是[-1,1]，因此，需要放大bound倍，以适应action_space 的不同取值。
        ac_mean = keras.layers.Lambda(lambda x: x * bound)(ac_layer3)
        ac_logstd = keras.layers.Dense(
            self.action_space)(ac_layer3)

        # Critic
        cr_layer1 = keras.layers.Dense(400, activation='tanh')(input_)
        cr_layer2 = keras.layers.Dense(300, activation='tanh')(cr_layer1)
        cr_layer3 = keras.layers.Dense(100, activation='tanh')(cr_layer2)
        cr_outputs = keras.layers.Dense(1)(cr_layer3)

        self.actor = keras.Model(
            inputs=[input_], outputs=[ac_mean, ac_logstd])
        self.critic = keras.Model(inputs=[input_], outputs=[cr_outputs])
        # self.critic = tf.Variable(shape=[self.state_space])
    def call(self, state):
        ac_mean, ac_logstd = self.actor(state)
        value = self.critic(state)
        return tf.squeeze(ac_mean), tf.squeeze(ac_logstd), tf.squeeze(value)

    def get_dist(self, mean, logstd):
        std = tf.exp(logstd)
        return tfd.Normal(loc=mean, scale=std)

    @staticmethod
    def _normal_logproba(action, mean, logstd, std=None):
        if std is None:
            std = tf.exp(logstd)

        std_sq = tf.math.pow(std, 2)
        logproba = - 0.5 * \
            tf.math.log(2 * np.math.pi) - logstd - \
            tf.math.pow((action - mean), 2) / (2 * std_sq)
        return tf.reduce_sum(logproba)

    def _choose_action(self, mean, logstd, return_old_log=True):
        std = tf.math.exp(logstd)
        print("mean.shape = {}, logstd.shape = {}".format(mean.shape, logstd.shape))
        action = tf.random.normal([self.action_space], mean, std, tf.float32)
        log_probs = ActorCritic._normal_logproba(action, mean, logstd, std)
        cliped_action = tf.clip_by_value(action, -1 * self.bound,  1 * self.bound)
        if return_old_log:
            return tf.squeeze(cliped_action), log_probs
        else:
            return tf.squeeze(cliped_action)

def ppo(config, env):

    # Hyper Params Setting
    buffer = deque(maxlen=config.max_len)
    n_episodes = config.n_episodes
    n_fixed_steps = config.n_fixed_steps
    n_max_steps = config.n_max_steps

    discount_factor = config.discount_factor
    lamda = config.lamda
    epsilon = config.epsilon

    # Define Network
    ac_net = ActorCritic(env.observation_space.shape[0],env.action_space.shape[0], 1)
    optimizer = keras.optimizers.Adam(learning_rate=4e-4)

    # visual 
    loss_logdir = get_run_logdir()
    writer = tf.summary.create_file_writer(loss_logdir)
    global_epoch = 0

    for i_episode in range(n_episodes):

        print("epoch: {}".format(i_episode))
        num_steps = 0
        rewards = 0.0
        buffer.clear()

        while num_steps < n_fixed_steps:
            
            state = env.reset()
            state = np.expand_dims(state, axis=0).astype('float32')

            # 收集数据
            for t in range(n_max_steps):
                ac_mean, ac_logstd, value = ac_net(state)
                action, log_prob = ac_net._choose_action(ac_mean, ac_logstd)
                action = action.numpy()
                log_prob = log_prob.numpy()
                next_state, reward, done, info = env.step(action)
                next_state = next_state.reshape(1,-1).astype(np.float32)
                reward = reward.astype(np.float32)
                rewards += reward
                mask = 0 if done else 1
                buffer.append(
                    [state, value, action, log_prob, mask, reward])

                with writer.as_default():
                    tf.summary.scalar('Main/reward', reward, step=i_episode * n_fixed_steps + num_steps)

                state = next_state
                if done:
                    break
                
            
            num_steps += (t +1)

        all_states, all_values, all_actions, all_old_probs, all_masks, all_rewards = [
            [buffer[i][var_index] for i in range(len(buffer))]
            for var_index in range(6)
        ]
        all_states = np.squeeze(all_states)

        # 计算 GAE
        all_returns = np.zeros(shape=len(all_values), dtype=np.float32)
        all_advantages = np.zeros(shape=len(all_values), dtype=np.float32)
        all_deltas = np.zeros(shape=len(all_values), dtype=np.float32)

        pre_return = 0
        pre_value = 0
        pre_advantage = 0
        for i in reversed(range(len(buffer))):
            all_returns[i] = all_rewards[i] + discount_factor * \
                pre_return * all_masks[i]
            all_deltas[i] = all_rewards[i] + discount_factor * \
                pre_value * all_masks[i] - all_values[i]
            all_advantages[i] = all_deltas[i] + discount_factor * \
                lamda * pre_advantage * all_masks[i]

            pre_return = all_returns[i]
            pre_value = all_values[i]
            pre_advantage = all_advantages[i]

        dataset = tf.data.Dataset.from_tensor_slices((all_states,all_advantages, all_returns, all_actions, all_old_probs))
        dataset = dataset.repeat(config.repeat_size).shuffle(3000).batch(config.batch_size)
    
        for index, batch_data in enumerate(dataset):
            batch_states, batch_advantages, batch_rewards, batch_actions, batch_old_probs = batch_data

            with tf.GradientTape() as tape:
                ac_mean, ac_logstd, cr_values = ac_net(batch_states)
                ac_logproba = ActorCritic._normal_logproba(batch_actions, ac_mean, ac_logstd)

                rio = tf.math.exp(tf.reduce_mean(ac_logproba - batch_old_probs))
                cliped_rio = tf.clip_by_value(rio, 1-epsilon, 1+epsilon)

                a_loss = tf.math.minimum(rio * tf.expand_dims(batch_advantages,1 ), cliped_rio * tf.expand_dims(batch_advantages,1))
                ac_loss = -tf.reduce_mean(a_loss)
                
                c_loss = keras.losses.mean_squared_error(cr_values, batch_rewards)
                cr_loss = tf.reduce_mean(c_loss)

                all_loss = cr_loss + ac_loss
            
            global_epoch += 1
            with writer.as_default():
                tf.summary.scalar('Loss/rio', rio, step = global_epoch)

                tf.summary.scalar('Loss/ac_loss', ac_loss, step=global_epoch)
                tf.summary.scalar('Loss/cr_loss', cr_loss, step=global_epoch)
                #tf.summary.scalar('Gradient/cr_loss', tape.gradient(cr_loss, cr_loss), step = global_epoch)
            writer.flush()
            
            # Update Gradient
            gradients = tape.gradient(all_loss, ac_net.trainable_variables)
            optimizer.apply_gradients(zip(gradients, ac_net.trainable_variables))
        
if __name__=='__main__':

    config = get_config()

    np.random.seed(42)
    tf.random.set_seed(42)
    env = gym.make(config.ENV_NAME)
    env.seed(42)

    ppo(config, env)
    
