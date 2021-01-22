import os
import gym
import copy
import imageio
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import tensorflow_probability as tfp
tfd = tfp.distributions

tf.keras.backend.set_floatx('float32')

# Visualize
root_logdir = os.path.join(os.curdir, 'ppo_logs')


def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


def model(state_shape, action_dim, units=(400, 300, 100), discrete=False):
    state = keras.Input(shape=state_shape)

    vf = keras.layers.Dense(
        units[0], name="Value_L0", activation='tanh')(state)
    for index in range(1, len(units)):
        vf = keras.layers.Dense(
            units[index], name="Value_L{}".format(index), activation='tanh')(vf)

    value_pred = keras.layers.Dense(1, name="Out_value", activation='tanh')(vf)

    pi = keras.layers.Dense(
        units[0], name="Policy_L0", activation='tanh')(state)
    for index in range(1, len(units)):
        pi = keras.layers.Dense(
            units[index], name="Policy_L{}".format(index), activation='tanh')(pi)

    if discrete:
        action_probs = keras.layers.Dense(
            action_dim, name="Out_probs", activation='softmax')(pi)
        model = keras.Model(inputs=state, outputs=[action_probs, value_pred])
    else:
        actions_mean = keras.layers.Dense(
            action_dim, name="Out_mean", activation='tanh')(pi)
        model = keras.Model(inputs=state, outputs=[actions_mean, value_pred])

    return model


class PPO:
    def __init__(
            self,
            env,
            discrete=False,
            lr=5e-4,
            hidden_units=(64, 64),
            c1=1.0,
            c2=0.01,
            clip_ratio=0.2,
            gamma=0.98,
            lam=0.98,
            batch_size=2048,
            mini_batch=64,
            repeat_size=4,
            writer=None
    ):
        self.env = env
        self.state_shape = env.observation_space.shape  # shape of observations
        # number of actions
        self.action_dim = env.action_space.n if discrete else env.action_space.shape[0]
        self.discrete = discrete
        if not discrete:
            self.action_bound = (env.action_space.high -
                                 env.action_space.low) / 2
            self.action_shift = (env.action_space.high +
                                 env.action_space.low) / 2

        # Define and initialize network
        self.policy = model(self.state_shape, self.action_dim,
                            hidden_units, discrete=discrete)
        self.model_optimizer = keras.optimizers.Adam(learning_rate=lr)
        print(self.policy.summary())

        # Stdev for continuous action
        if not discrete:
            self.policy_log_std = tf.Variable(
                tf.zeros(self.action_dim, dtype=tf.float32), trainable=True)

        # Set hyperparameters
        self.gamma = gamma  # discount factor
        self.lam = lam
        self.c1 = c1  # value difference coeff
        self.c2 = c2  # entropy coeff
        self.clip_ratio = clip_ratio  # for clipped surrogate
        self.batch_size = batch_size
        self.mini_batch = mini_batch
        self.repeat_size = repeat_size  # number of epochs per episode

        # Tensorboard
        self.summaries = {}
        self.writer = writer

        # test
        if not discrete:
            self.dist = np.random.normal

    @tf.function
    def evaluate_actions(self, state, action):
        mean, value = self.policy(state)

        action = (action - self.action_shift) / self.action_bound
        log_probs = self.get_logproba(mean, action)
        
        log_probs = tf.reduce_sum(log_probs, axis=-1)

        return log_probs, value

    @tf.function
    def _normal_logproba(self, x, mean, logstd, std=None):
        if std is None:
            std = tf.math.exp(logstd)

        std_sq = tf.math.pow(std,2)
        logproba = - 0.5 * tf.math.log(2 * np.math.pi) - logstd - tf.math.pow((x - mean), 2) / (2 * std_sq)
        return tf.reduce_sum(logproba)

    @tf.function
    def get_logproba(self, action_mean, actions):

        logproba = self._normal_logproba(actions, action_mean, self.policy_log_std)
        return logproba

    @tf.function
    def act(self, state, test=False):
        
        state = tf.cast(tf.expand_dims(state, axis=0),tf.float32)

        mean, value = self.policy(state)

        std = tf.exp(self.policy_log_std)
        
        action = mean if test else self.dist(loc=mean, scale=std)
        action = tf.constant(action, dtype=tf.float32)
        action = tf.clip_by_value(action, -1, 1)
        action = action * self.action_bound + self.action_shift
        log_probs = self._normal_logproba(action, mean, self.policy_log_std, std)
        return action[0].numpy(), value[0][0], log_probs

    def save_model(self, fn):
        self.policy.save(fn)

    def load_model(self, fn):
        self.policy.load_weights(fn)
        print(self.policy.summary())

    @tf.function
    def learn(self, all_states, all_advantages, all_rewards, all_actions, all_old_probs):
        all_rewards = tf.expand_dims(all_rewards, axis=-1)
        all_rewards = tf.cast(all_rewards, tf.float32)

        with tf.GradientTape() as tape:
            new_log_probs, state_values = self.evaluate_actions(all_states, all_actions)

            entropy = tf.reduce_mean(tf.math.exp(new_log_probs) * new_log_probs)

            ratios = tf.exp(new_log_probs - all_old_probs)
            clipped_ratios = tf.clip_by_value(ratios, clip_value_min=1-self.clip_ratio,
                                            clip_value_max=1+self.clip_ratio)
            loss_clip = tf.minimum(all_advantages * ratios, all_advantages * clipped_ratios)
            loss_clip = tf.reduce_mean(loss_clip)

            vf_loss = keras.losses.mean_squared_error(state_values, all_rewards)
            vf_loss = tf.reduce_mean(vf_loss)

            total_loss = -loss_clip + self.c1 * vf_loss - self.c2 * entropy

        train_variables = self.policy.trainable_variables
        train_variables += [self.policy_log_std]

        grad = tape.gradient(total_loss, train_variables)  # compute gradient
        self.model_optimizer.apply_gradients(zip(grad, train_variables))

        # tensorboard info
        self.summaries['total_loss'] = total_loss
        self.summaries['surr_loss'] = loss_clip
        self.summaries['vf_loss'] = vf_loss
        self.summaries['entropy'] = entropy

    @tf.function
    def train(self, num_episode=2000, max_steps=2000, save_freq=50):

        cnts = 0
        epoch = 0
        buffer = deque(maxlen=3000)
        
        for i_episode in range(num_episode):

            buffer.clear()
            num_steps = 0
            rewards = []
            while num_steps < self.batch_size:
                
                state = self.env.reset()
                reward_sum = 0
                start_ = datetime.datetime.now()
                for t in range(max_steps):
                    #start = datetime.datetime.now()
                    action, value, log_prob = self.act(state) 
                    #print("动作耗时:{}".format(datetime.datetime.now()-start))
                    next_state, reward, done, _ = self.env.step(action)
                    reward_sum += reward
                    mask = 0 if done else 1
                    
                    buffer.append([state, value, action, log_prob, mask, reward])
                    if done:
                        break

                    state = next_state
                print("采集一条轨迹时间：{}".format(datetime.datetime.now() - start_))
                num_steps += (t+1)
                with self.writer.as_default():
                    tf.summary.scalar('Main/reward', reward_sum, cnts)
                    tf.summary.scalar('Main/step', (t+1), cnts)
                self.writer.flush()
                rewards.append(reward_sum)
                cnts += 1
            avg_rewards = tf.reduce_mean(rewards)
            print("episode: {} , avg_rewards = {}".format(i_episode, avg_rewards))

            all_states, all_values, all_actions, all_old_probs, all_masks, all_rewards = [
                [buffer[i][var_index] for i in range(len(buffer))]
                for var_index in range(6)
            ]
            all_states = np.squeeze(all_states)
            all_advantages = np.zeros(shape=len(all_values), dtype=np.float32)

            # 计算 GAE
            all_returns = np.zeros(shape=len(all_values), dtype=np.float32)
            all_advantages = np.zeros(shape=len(all_values), dtype=np.float32)
            all_deltas = np.zeros(shape=len(all_values), dtype=np.float32)

            pre_return = 0
            pre_value = 0
            pre_advantage = 0
            for i in reversed(range(len(buffer))):
                all_returns[i] = all_rewards[i] + self.gamma * \
                    pre_return * all_masks[i]
                all_deltas[i] = all_rewards[i] + self.gamma * \
                    pre_value * all_masks[i] - all_values[i]
                all_advantages[i] = all_deltas[i] + self.gamma * \
                    self.lam * pre_advantage * all_masks[i]

                pre_return = all_returns[i]
                pre_value = all_values[i]
                pre_advantage = all_advantages[i]

            dataset = tf.data.Dataset.from_tensor_slices((all_states,all_advantages, all_returns, all_actions, all_old_probs))
            dataset = dataset.repeat(self.repeat_size).shuffle(3000).batch(self.mini_batch)

            for batch_data in dataset:
                self.learn(*batch_data)

                with self.writer.as_default():
                    tf.summary.scalar('Loss/total_loss', self.summaries['total_loss'], step=epoch)
                    tf.summary.scalar('Loss/clipped_surr', self.summaries['surr_loss'], step=epoch)
                    tf.summary.scalar('Loss/vf_loss', self.summaries['vf_loss'], step=epoch)
                    tf.summary.scalar('Loss/entropy', self.summaries['entropy'], step=epoch)

                self.writer.flush()
                epoch += 1
            print("episode {}: {} total reward".format(
                i_episode, np.sum(all_rewards)))
            if i_episode+1 % save_freq == 0:
                self.save_model('../models/ppo/')

    def test(self, render=True, fps=30, filename='test_render.mp4'):
        cur_state, done, rewards = self.env.reset(), False, 0
        video = imageio.get_writer(filename, fps=fps)
        while not done:
            action, value, log_prob = self.act(cur_state, test=True)
            next_state, reward, done, _ = self.env.step(action)
            cur_state = next_state
            rewards += reward
            if render:
                video.append_data(self.env.render(mode='rgb_array'))
        video.close()
        return rewards


if __name__ == "__main__":

    loss_logdir = get_run_logdir()
    writer = tf.summary.create_file_writer(loss_logdir)

    gym_env = gym.make('Hopper-v2')
    
    try:
        assert ((gym_env.action_space.high == -gym_env.action_space.low).all())
        is_discrete = False
        print('Continuous Action Space')
    except AttributeError:
        is_discrete = True
        print('Discrete Action Space')
    
    gym_env.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)


    ppo = PPO(gym_env, discrete=is_discrete, writer=writer)

    ppo.train(num_episode=1000, save_freq=50)
    # reward = ppo.test()
    # print("Total rewards: ", reward)  
