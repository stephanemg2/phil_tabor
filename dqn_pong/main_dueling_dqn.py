import time

import numpy as np
from agents import DuelingDQNAgent
from utils import make_env, plot_learning_curve

if __name__ == '__main__':
    render = False
    env_name = 'PongNoFrameskip-v4'
    env = make_env(env_name, render)
    # works for games with negative score
    best_score = -np.inf

    load_checkpoint = False
    save_checkpoint = True

    n_games = 300
    agent = DuelingDQNAgent(gamma=0.99, epsilon=1.0, lr=0.0001, input_dims=env.observation_space.shape,
                     n_actions=env.action_space.n, mem_size=50000, eps_min=0.1, batch_size=32, replace=1000,
                     eps_dec=1e-5, chkpt_dir='models/', algo='DuelingDQNAgent', env_name=env_name)
    if load_checkpoint:
        agent.load_models()
        agent.epsilon = 0.1

    fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) + '_' + str(n_games) + '_games'
    figure_file = 'plots/' + fname + '.png'
    n_steps = 0
    scores, eps_history, steps_array = [], [], []

    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info, _ = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_
            n_steps += 1

        scores.append(score)
        steps_array.append(n_steps)
        avg_score = np.mean(scores[-100:])
        print(
            f'episode {i} score:{score} average score {avg_score} best score {best_score} epsilon {agent.epsilon} steps {n_steps}')
        # save checkpoint if avg_score > best_score
        if avg_score > best_score:
            if save_checkpoint:
                agent.save_models()
            best_score = avg_score
        eps_history.append(agent.epsilon)
    env.close()
    plot_learning_curve(steps_array, scores, eps_history, figure_file)
