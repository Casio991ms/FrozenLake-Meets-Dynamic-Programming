import pickle

import gymnasium as gym
import numpy as np


def load_pickle(path):
    file = open(path, "rb")
    q = pickle.load(file)
    file.close()
    return q


env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")
env = gym.wrappers.RecordVideo(env, 'video')

state, info = env.reset()

done = False

while not done:
    action = np.argmax(q[state[0], state[1], state[2], :])

    new_state, reward, terminated, truncated, info = env.step(action)
    state = new_state
    done = done or terminated or truncated

env.close()
