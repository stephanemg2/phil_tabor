# frozen-lake-ex2.py
import gym
import numpy as np
import matplotlib.pyplot as plt

MAX_ITERATIONS = 10
NUM_GAMES = 1000

env = gym.make("FrozenLake-v1", render_mode="rgb_array")
env.reset()
env.render()
win_pct = []
scores = []
for i in range (NUM_GAMES):
    done = False
    score = 0
    env.reset()
    while not done:
        random_action = env.action_space.sample()
        new_state, reward, done, info, _ = env.step(
            random_action)
        score += reward
    scores.append(score)
    if i % 10 == 0:
        average = np.mean(scores[-10:])
        win_pct.append(average)
plt.plot(win_pct)
plt.show()
#le dessin montre une chance de 10% de succes en jouant aléatoirement.