# WARN:  自己写的 逻辑有问题
# 懒得改了
from operator import ne
import os
import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras

# 针对tensorboard设置
root_logdir = os.path.join(os.path.dirname(__file__), 'my_logs')


def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


# 定义环境
env = gym.make('CartPole-v0')

# 固定随机种子
np.random.seed(42)
tf.random.set_seed(42)
env.seed(42)


def config():
    return {
        'n_episodes': 600,
        'step_per_episode': 200,
        'state_space': 4,
        'action_space': 2,
        'discount_factor': 0.95,
        'lamda': 0.9,
        'batch_size': 1000
    }


optimizer = keras.optimizers.Adam(learning_rate=1e-3)


# 定义目标函数_policy_gradient_loss
def clip_loss(epsilon, pi_new, pi_old, A):
    ratio = pi_new / pi_old
    cliped = tf.clip_by_value(ratio, 1 - epsilon, 1 + epsilon)
    return tf.math.minimum(ratio * A, cliped * A)


def vf_loss(values, rewards):
    # args:
    #   v.shape = [steps,1]
    # Return.shape == [steps, 1]
    tf.square(values - rewards)


# 输出各个动作发生的概率
action_model = keras.models.Sequential([
    keras.layers.Dense(32, activation='relu', input_shape=[state_space]),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(action_space, activation='softmax')
])

# 输出每个state的Value
value_model = keras.models.Sequential([
    keras.layers.Dense(32, activation='relu', input_shape=[state_space]),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1)
])


# NOTE: 这里计算 advantage_function 使用原paper公式11，
# 考虑到计算 δ 需要使用 V(s+1) 而一共只有 t 个state ，所以需要再添加一个结束状态的 Value
#   Args：
#       values.shape    =  [t+1,]
#       rewards.shape   =  [t,]
#   Return:
#       rewards(实质是Advantage).shape = [t,]
def advantage_estimations(rewards: np.ndarray, values: np.ndarray,
                          discount_factor, lamda):
    rewards = np.array(rewards)
    values = np.array(values)
    advantages = rewards.copy()
    # 先计算最后一个state的A，下面正常迭代
    advantages[len(advantages) -
               1] = advantages[len(advantages) - 1] + discount_factor * values[
                   len(advantages)] - values[len(advantages) - 1]

    for i in range(len(advantages) - 2, -1, -1):
        delta = advantages[i] + discount_factor * values[i + 1] - values[i]
        advantages[i] = delta + lamda * discount_factor * advantages[i + 1]
        rewards[i] += discount_factor * rewards[i + 1]
    return advantages, rewards


def play_one_step(policy_net, env: gym.Env, obs):
    # 计算 action
    action_probs = policy_net(obs)
    action = tf.random.categorical(logits=action_probs[np.newaxis],
                                   num_samples=1)
    # 计算当前state的Value
    # v_state = value_net(obs)
    # 改成存储state
    obs, reward, done, info = env.step(action)

    return obs, reward, done, info, action


def play_multiple_episodes(env: gym.Env, policy_net, n_episodes, n_max_steps):
    all_rewards = []
    all_states = []
    all_actions = []
    for episode in range(n_episodes):
        obs = env.reset()
        rewards = []
        states = []
        actions = []
        states.append(obs)
        for step in range(n_max_steps):
            obs, reward, done, info, action = play_one_step(
                policy_net, env, obs)
            rewards.append(reward)
            states.append(obs)
            actions.append(action)
            if done or step == n_max_steps - 1:
                # 基于 ‘advantage_estimations’ 给出的原因，需要多存储一个Value
                # values.append(value_net(obs))
                # WARN: 应该存储state 这样后续所有需要使用Value的地方只需要计算一下即可
                states.append(obs)
                break
        all_rewards.append(rewards)
        all_states.append(states)
        all_actions.append(actions)
    return all_rewards, all_states, all_actions


def process_advantages_rewards(all_rewards, all_values, discount_factor,
                               lamda):
    #all_advantages = [
    #   advantage_estimations(rewards, values, discount_factor, lamda)
    #    for rewards, values in zip(all_rewards, all_values)]
    all_rewards = []
    all_advantages = []
    for rewards, values in zip(all_rewards, all_values):
        advantages, rewards = advantage_estimations(rewards, values,
                                                    discount_factor, lamda)
        all_advantages.append(advantages)
        all_rewards.append(rewards)
    all_advantages_flatted = np.concatenate(all_advantages, axis=0)
    all_rewards_flatted = np.concatenate(all_rewards, axis=0)
    return all_advantages_flatted, all_rewards_flatted


def train(env: gym.Env, policy_net: keras.Model, value_net: keras.Model,
          n_episodes, n_max_steps, n_iterations, discount_factor, lamda,
          batch_size, action_space, epsilon):
    policy_new = policy_net
    for iteration in range(n_iterations):
        policy_old = policy_new
        all_rewards, all_states, all_actions = play_multiple_episodes(
            env, policy_old, n_episodes, n_max_steps)

        all_values = []
        for states in all_states:
            values = value_net(states)
            all_values.append(values)

        # 之所以多保留一维state 是因为计算 A的时候 需要V(t+1)
        # 所以计算完成V之后不需要多保存一维state
        all_states = all_states[:, :-1, :]

        # 返回 flatted advantages和 flatted rewards
        all_advantages_flatted, all_rewards_flatted = process_advantages_rewards(
            all_rewards, all_values, discount_factor, lamda)
        all_states_flatted = np.concatenate(all_states, axis=0)
        all_actions_flatted = np.concatenate(all_actions, axis=0)
        # values在计算完advantages就不需要保存 end state 的Value了
        all_values = all_values[:,:-1,:]
        all_values_flatted = np.concatenate(all_values, axis=0)
        # Update Policy Net θ need mini-batch
        # since θ_new is equal to θ_old at iteration 1

        # 转换成 dataset
        dataset_values = tf.data.Dataset.from_tensor_slices(
            all_values_flatted).batch(batch_size).prefetch(1)
        dataset_advantages = tf.data.Dataset.from_tensor_slices(
            all_advantages_flatted).batch(batch_size).prefetch(1)
        dataset_rewards = tf.data.Dataset.from_tensor_slices(
            all_rewards_flatted).batch(batch_size).prefetch(1)
        dataset_states = tf.data.Dataset.from_tensor_slices(
            all_states_flatted).batch(batch_size).prefetch(1)
        dataset_actions = tf.data.Dataset.from_tensor_slices(
            all_actions_flatted).batch(batch_size).prefetch(1)

        # Update Policy Network
        for batch_actions, batch_states, batch_advantages in zip(
                dataset_actions, dataset_states, dataset_advantages):

            mask = tf.one_hot(batch_actions, depth=action_space)
            with tf.GradientTape() as tape:
                probs_new = policy_new(batch_states)
                pi_new = tf.reduce_sum(probs_new * mask, axis=1, keepdims=True)
                probs_old = policy_old(batch_states)
                pi_old = tf.reduce_sum(probs_old * mask, axis=1, keepdims=True)
                loss_tmp = clip_loss(epsilon, pi_new, pi_old, batch_advantages)
                loss = -tf.reduce_mean(loss_tmp)
            grads = tape.gradient(loss, policy_new.trainable_variables)
            optimizer.apply_gradients(
                zip(grads, policy_new.trainable_variables))

        # Update Value Network
        #for batch_rewards , batch
        for batch_values, batch_rewards in zip(dataset_values, dataset_rewards):
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(tf.square(batch_values - batch_rewards))

        