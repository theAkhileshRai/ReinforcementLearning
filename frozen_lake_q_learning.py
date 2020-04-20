import gym
import matplotlib.pyplot as plt
from q_learning_agent import Agent
import numpy as np

#Frozen Lake.py
if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
    agent = Agent( lr=0.001,gamma=0.9,n_actions=4,  n_states=16,eps_start=1.0,eps_end=0.01,eps_dec=0.9999995)
    scores = []
    win_pct_lst= []
    n_games = 500000

    for i in range(n_games):
        done = False
        observation = env.reset()
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done,info = env.step(action)
            agent.learn(observation,action,reward,observation_)
            score += reward
            observation = observation_

        scores.append(score)
        if i%100 == 0:
            win_pct = np.mean(scores[-100:])
            win_pct_lst.append(win_pct)
            if i%1000 == 0:
                print('episode',i,'win_pct %.2f' % win_pct, 'epislon %.2f'%agent.epsilon)

    plt.plot(win_pct_lst)
    plt.show()
