import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras

def play_one_step(env, obs, model, loss_fn):
    '''
    Args:
        env: gym object
        obs: a list of observation
        model: nn model
        loss_fn: caculate loss value
    '''
    with tf.GradientTape() as tape:
        # obs[np.newaxis] change obs.shape from (4,) to (4,1)
        left_prob = model(obs[np.newaxis])
        # 0(left) 1(right)
        action = (tf.random.uniform([1,1]) > left_prob)
        # the target probability of going to left
        # tf.constant([[1.]]).shape = (1,1)
        y_target = tf.constant([[1.]]) - tf.cast(action, tf.float32)
        loss = tf.reduce_mean(loss_fn(y_target, left_prob))
    
    grads = tape.gradient(loss, model.trainable_variables) 
    obs, reward, done, info = env.step(int(action[0, 0].numpy())) 
    return obs, reward, done, grads


def play_multiple_episodes(env, n_episodes, n_max_step, model, loss_fn):
    '''
    return:
        all_rewards: a 2-d vector , i-th cow is a list means i-th episode all rewards
        all_grads: a 3-d vector, all_grads[episode][step][variable_index]
    '''
    all_rewards = []
    all_grads = []
    for episode in range(n_episodes):
        print("episode: {}".format(episode))
        current_rewards = []
        current_grads = []
        obs = env.reset()
        for step in range(n_max_step):
            obs, reward, done, grads = play_one_step(env, obs, model, loss_fn)
            current_rewards.append(reward)
            current_grads.append(grads)
            if done:
                break            
       
        all_rewards.append(current_rewards)
        all_grads.append(current_grads)
    return all_rewards , all_grads

def discount_rewards(rewards, discount_factor):
    discounted = np.array(rewards)
    for step in range(len(rewards) - 2, -1, -1):
        discounted[step] += discount_factor * discounted[step+1]
    return discounted

# 对获得的所有rewards进行正则化
def discount_and_normalize_rewards(all_rewards, discount_factor):
    all_discounted_rewards = [discount_rewards(rewards, discount_factor) for rewards in all_rewards]

    # transfer Matrix to Vector
    flat_rewards = np.concatenate(all_discounted_rewards, axis=0)
    rewards_mean = np.mean(flat_rewards)
    rewards_std = np.std(flat_rewards)
    return [(discounted - rewards_mean) / rewards_std for discounted in all_discounted_rewards]


if __name__ == '__main__':

    n_iterations = 15
    n_episodes_per_update = 10
    n_max_steps = 200
    discount_factor = 0.95

    env = gym.make('CartPole-v1')
    n_inpus = 4

    model = keras.models.Sequential([
        keras.layers.Dense(5, activation='elu', input_shape=[n_inpus]),
        keras.layers.Dense(1, activation='sigmoid'),
    ])

    optimizer = keras.optimizers.Adam(lr=0.01)
    loss_fn = keras.losses.binary_crossentropy

    for iteration in range(n_iterations):
        print("iteration:{}".format(iteration))
        # get rewards & grads
        all_rewards, all_grads = play_multiple_episodes(env, n_episodes_per_update, n_max_steps, model, loss_fn)

        # Normalization 
        all_final_rewards = discount_and_normalize_rewards(all_rewards, discount_factor)

        all_mean_grads = []
        # for every trainable_variables
        for var_index in range(len(model.trainable_variables)):
            mean_grads = tf.reduce_mean(
                [final_reward * all_grads[episode_index][step][var_index]
                # final_rewards is a list , means all rewards in episode_index-th episode
                for episode_index, final_rewards in enumerate(all_final_rewards)
                    # per-step 
                    for step, final_reward in enumerate(final_rewards)], axis=0
            )
            all_mean_grads.append(mean_grads)
        optimizer.apply_gradients(zip(all_mean_grads, model.trainable_variables))
        