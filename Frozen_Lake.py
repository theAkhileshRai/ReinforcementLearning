import gym
import numpy as np
import matplotlib.pyplot as plt


env = gym.make('FrozenLake-v0')


#LEFT = 0 DOWN = 1 RIGHT = 2 UP = 3
# SFFF
# FHFH
# FFFH
# HFFG

policy = {0: 1, 1: 2, 2: 1, 3: 0, 4: 1, 5: 1, 6: 2, 7: 1, 8: 2, 9: 1, 10: 1, 13: 2, 14: 2}

n_games = 1000
win_pct = []
scores = []

for i in range(n_games):
    done = False
    obs = env.reset()
    score = 0
    while not done:
         action = policy[obs]
         obs,rewards, done, info = env.step(action)
         score += rewards
    scores.append(score)
    if i%10 == 0:
        average = np.mean(scores[-10:])
        win_pct.append((average))

plt.plot(win_pct)
plt.show()
