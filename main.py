import pickle

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from gymnasium import Env


def train(env: Env, env_name, episode_count: int,
          alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 1, epsilon_decay: float = 0.001):
    rng = np.random.default_rng()
    rewards = np.zeros(episode_count)

    q = np.zeros((env.observation_space[0].n, env.observation_space[1].n, env.observation_space[2].n, env.action_space.n))

    for episode in range(episode_count):
        state, info = env.reset()

        terminated = False
        truncated = False

        while not terminated and not truncated:
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state[0], state[1], state[2], :])

            new_state, reward, terminated, truncated, info = env.step(action)

            q[state, action] += alpha * (
                    reward + gamma * (np.max(q[new_state[0], new_state[1], new_state[2], :]) - q[new_state[0], new_state[1], new_state[2], action])
            )

            state = new_state

        epsilon *= (1 - epsilon_decay)
        rewards[episode] = reward

    env.close()

    file = open(f"{env_name}.pkl", "wb")
    pickle.dump(q, file)
    file.close()

def print_v_from_q(q):
    v = np.max(q, axis=3)
    sns.kdeplot(v[:, :, 0])
    plt.show()
    print(v[:, :, 0])
    # plt.plot(v[:, :, 0])
    # print(v[:, :, 1])
