import gym
from deep_q_learning_phil import Agent
from utils import plot_learning_curve
import numpy as np

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4, eps_end=0.01, input_dims=[8],lr=0.003)
    #on affiche la croissance des scores et le decay d'epsilon
    scores, eps_history, games_history = [], [], []
    n_games = 500

    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()[0]
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info, _ = env.step(action)
            score += reward
            #on stocke la transition dans l'agent
            agent.store_transition(observation, action, reward, observation_,done)
            agent.learn()
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)
        #on prends la moyenne des 100 derniers scores
        avg_score = np.mean(scores[-100:])
        print ('episode ', i, 'score %.2f' % score, 'average score %.2f' % avg_score, 'epsilon %.2f' % agent.epsilon)
        games_history.append(i)
        plot_learning_curve(games_history, scores, eps_history, show_result=False)
    plot_learning_curve(games_history, scores, eps_history,show_result=True)

