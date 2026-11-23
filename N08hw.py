from __future__ import print_function

import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gymnasium as gym
import scipy.signal

np.random.seed(42)
tf.random.set_seed(42)

NUM_EPISODES = 25
GAMMA = 0.99
VALUE_FN_ITERS = 80
POLICY_FN_ITERS = 80
KL_MARGIN = 1.2
KL_TARGET = 0.01
CLIP_RATIO = 0.2


def network_fn(shape):
    # Same two-hidden-layer network architecture as the notebook.
    shape = int(shape)
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(4,)),
        tf.keras.layers.Dense(256, activation=tf.nn.tanh),
        tf.keras.layers.Dense(256, activation=tf.nn.tanh),
        tf.keras.layers.Dense(shape),
    ])


def discount_cumsum(x, discount):
    """Discounted sums, as in the notebook's scipy.signal helper."""
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
def categorical_kl(logp0, logp1):
    # KL(old || new): old probabilities weight the log-probability ratio.
    return tf.reduce_mean(tf.reduce_sum(tf.exp(logp0) * (logp0 - logp1), axis=1))


def categorical_entropy(logits):
    return -1 * tf.reduce_mean(
        tf.reduce_sum(tf.nn.softmax(logits) * tf.nn.log_softmax(logits), axis=-1)
    )


def collect_episode(env, policy_network):
    """One episode of CartPole. Save the state and action for each reward."""
    state_buffer = []
    action_buffer = []
    reward_buffer = []

    curr_state, _ = env.reset()
    done = False

    while not done:
        state_buffer.append(curr_state)
        action_logits = policy_network(np.expand_dims(curr_state, 0), training=False)
        action = int(tf.random.categorical(action_logits, 1)[0, 0].numpy())
        action_buffer.append(action)

        obs, reward, terminated, truncated, _ = env.step(action)
        reward_buffer.append(reward)
        done = terminated or truncated
        curr_state = obs

    return (
        np.asarray(state_buffer, dtype=np.float32),
        np.asarray(action_buffer, dtype=np.int32),
        np.asarray(reward_buffer, dtype=np.float32),
        np.asarray(curr_state, dtype=np.float32),
        bool(terminated),
    )
def update_fn(policy_network, value_network, states, actions, rewards,
              last_state, terminated, policy_optimizer, value_optimizer,
              gamma=GAMMA):
    """Notebook-style clipped PPO actor-critic update for one complete episode.

    Unlike the notebook's sketch, the last transition is included. The value
    network predicts a scalar per state, and terminated episodes bootstrap 0.
    """
    values = tf.squeeze(value_network(states, training=False), axis=-1).numpy()
    if terminated:
        bootstrap_value = 0.0
    else:
        bootstrap_value = float(value_network(last_state[None, :], training=False)[0, 0])

    next_values = np.concatenate((values[1:], [bootstrap_value]))
    deltas = rewards + gamma * next_values - values
    advantage = discount_cumsum(deltas, gamma).astype(np.float32)
    returns = discount_cumsum(
        np.concatenate((rewards, [bootstrap_value])).astype(np.float32), gamma
    )[:-1].astype(np.float32)

    states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
    actions_tensor = tf.convert_to_tensor(actions, dtype=tf.int32)
    advantage_tensor = tf.convert_to_tensor(advantage, dtype=tf.float32)
    returns_tensor = tf.convert_to_tensor(returns, dtype=tf.float32)
    indices = tf.stack([tf.range(len(actions), dtype=tf.int32), actions_tensor], axis=1)

    old_logits = policy_network(states_tensor, training=False)
    initial_all_logp = tf.stop_gradient(tf.nn.log_softmax(old_logits))
    initial_action_logp = tf.gather_nd(initial_all_logp, indices)

    for _ in range(POLICY_FN_ITERS):
        with tf.GradientTape() as tape:
            action_logits = policy_network(states_tensor, training=True)
            all_logp = tf.nn.log_softmax(action_logits)
            action_logp = tf.gather_nd(all_logp, indices)
            ratio = tf.exp(action_logp - initial_action_logp)
            clipped_ratio = tf.clip_by_value(ratio, 1 - CLIP_RATIO, 1 + CLIP_RATIO)
            pi_objective = tf.reduce_mean(tf.minimum(
                ratio * advantage_tensor,
                clipped_ratio * advantage_tensor,
            ))
            pi_loss = -pi_objective

        grads = tape.gradient(pi_loss, policy_network.trainable_variables)
        policy_optimizer.apply_gradients(zip(grads, policy_network.trainable_variables))

        new_logp = tf.nn.log_softmax(policy_network(states_tensor, training=False))
        kl = categorical_kl(initial_all_logp, new_logp)
        if float(kl) > KL_MARGIN * KL_TARGET:
            break

    for _ in range(VALUE_FN_ITERS):
        with tf.GradientTape() as tape:
            estimated_values = tf.squeeze(value_network(states_tensor, training=True), axis=-1)
            val_loss = tf.reduce_mean(tf.square(estimated_values - returns_tensor))
        grads = tape.gradient(val_loss, value_network.trainable_variables)
        value_optimizer.apply_gradients(zip(grads, value_network.trainable_variables))

def main():
    print('Starting Hackathon 8 homework: 25 CartPole PPO episodes', flush=True)
    env = gym.make('CartPole-v1')
    num_actions = env.action_space.n

    policy_network = network_fn(num_actions)
    value_network = network_fn(1)
    policy_optimizer = tf.keras.optimizers.Adam()
    value_optimizer = tf.keras.optimizers.Adam()

    episode_lengths = []
    mean_entropies = []

    try:
        for episode in range(NUM_EPISODES):
            states, actions, rewards, last_state, terminated = collect_episode(
                env, policy_network
            )

            # Mean entropy of the policy's action distribution along this episode.
            # Measure before updating the policy so it describes actions just taken.
            logits = policy_network(states, training=False)
            mean_entropy = float(categorical_entropy(logits).numpy())
            episode_len = len(rewards)

            episode_lengths.append(episode_len)
            mean_entropies.append(mean_entropy)
            print(f'Episode {episode+1}/{NUM_EPISODES}: '
                  f'length={episode_len}, mean entropy={mean_entropy:.4f}', flush=True)

            update_fn(policy_network, value_network, states, actions, rewards,
                      last_state, terminated, policy_optimizer, value_optimizer)
    finally:
        env.close()

    with open('N08hw_results.csv', 'w', newline='', encoding='utf-8') as output:
        writer = csv.writer(output)
        writer.writerow(['episode', 'length', 'mean_entropy'])
        for i, (length, entropy) in enumerate(zip(episode_lengths, mean_entropies), 1):
            writer.writerow([i, length, entropy])

    episodes = np.arange(1, len(episode_lengths) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(episodes, episode_lengths, 'o-')
    plt.xlabel('Episode')
    plt.ylabel('Episode length (steps)')
    plt.title('CartPole episode length over 25 PPO training episodes')
    plt.xticks(episodes[::2])
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('N08hw_episode_length.png', dpi=180)

    plt.figure(figsize=(8, 4))
    plt.plot(episodes, mean_entropies, 'o-')
    plt.xlabel('Episode')
    plt.ylabel('Mean action-distribution entropy')
    plt.title('CartPole policy entropy over 25 PPO training episodes')
    plt.xticks(episodes[::2])
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('N08hw_entropy.png', dpi=180)
    plt.show()

    print('\nSaved N08hw_results.csv, N08hw_episode_length.png, N08hw_entropy.png')


if __name__ == '__main__':
    main()
