import gym
import qlearning_agent
import matplotlib.pyplot as plt
import numpy as np


def train(env_name,
          T=100000,
          lr=0.1,
          gamma=0.95,
          epsilon_i=1.0,
          epsilon_f=0.0,
          n_epsilon=0.1):
    env = gym.make(env_name)
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    agent = qlearning_agent.Agent(num_states, num_actions,
                                  lr=lr,
                                  gamma=gamma,
                                  epsilon_i=epsilon_i,
                                  epsilon_f=epsilon_f,
                                  n_epsilon=n_epsilon)

    scores = []
    win_pct_list = []

    state = env.reset()[0]

    for t in range(T):
        score = 0
        action = agent.act(state)
        next_state, reward, d_t, info, _ = env.step(action)
        agent.update(state, action, reward, next_state)
        agent.decay_epsilon(t / T)
        state = next_state
        score += reward

        if d_t:
            scores.append(score)
            state = env.reset()[0]
        if t % 100 == 0:
            win_pct = np.mean(scores[-100:])
            win_pct_list.append(win_pct)
            if t % 1000 == 0:
                print('episode ', t, 'win pct %.2f' % win_pct,
                      'epsilon %.2f' % agent.epsilon)
    plt.plot(win_pct_list)
    plt.show()


train("FrozenLake-v1", T=500000)
