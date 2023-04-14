import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gym
from utils import plot_learning_curve


# 2 linear layers 128 x n_actions
class LinearDeepQNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, lr=0.001):
        super(LinearDeepQNetwork, self).__init__()
        self.input_dims = input_dims
        # fully connected layer
        self.n_actions = n_actions
        # on passe une liste de 8 vecteurs d'observation
        self.fc1 = nn.Linear(*self.input_dims, 128)
        self.fc2 = nn.Linear(128, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    # retourne les array estimations d'action en fonction des etats envoyés
    def forward(self, state):
        x = F.relu(self.fc1(state))
        actions = self.fc2(x)
        # pas de fonction d'activation sur fc2
        return actions


class Agent:
    def __init__(self, input_dims, lr, n_actions, gamma=0.99, eps_min=0.01, eps_dec=1e-5, epsilon=1.0):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.lr = lr
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.action_space = [i for i in range(self.n_actions)]
        self.Q = LinearDeepQNetwork(self.input_dims, n_actions, self.lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            # on recupere le state lié a l'observation en envoyant l'observation a la device (gpu)
            state = T.tensor(np.array(observation), dtype=T.float).to(self.Q.device)
            # on passe le state au DQN qui retourne une liste d'actions possibles / valeur
            actions = self.Q.forward(state)
            # on recupere l'action avec la valorisation maximale
            action = T.argmax(actions).item()
        # sinon on explore avec une ramdon action appartement au champ d'actions du jeu
        else:
            action = np.random.choice(self.action_space)
        return action

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

#met a jour la fonction NN d'estimation de la fonction d'estimation de valorisation Q(s,a)
    def learn(self, state, action, reward, state_):
        self.Q.optimizer.zero_grad()
        # on convertit en tensors
        states = T.tensor(state).to(self.device)
        states_ = T.tensor(state_).to(self.device)
        actions = T.tensor(action).to(self.device)
        rewards = T.tensor(reward).to(self.device)

        #on recupere le tensor action (correspondant au tensor gpu)
        q_pred = self.Q.forward(states)[actions]
        #on prends la valorisation maximale prévue par le nouveau state envoyée au NN
        q_next = self.Q.forward(states_).max()

        #la valeur target est selon time difference le reward * gamma * la prediction max suivante
        q_target = reward + self.gamma*q_next

        #le loss est la difference entre l'action prise et celle qui aurait pu etre prise
        loss = self.Q.loss(q_target, q_pred).to(self.Q.device)

        loss.backward()
        self.Q.optimizer.step()
        self.decrement_epsilon()

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    n_games = 10000
    scores, eps_history = [], []

    agent = Agent(lr=0.0001,input_dims=env.observation_space.shape, n_actions=env.action_space.n)

    for i in range(n_games):
        score = 0
        done = False
        obs = env.reset()[0]

        while not done:
            action = agent.choose_action(obs)
            obs_, reward, done, info, _ = env.step(action)
            score += reward
            agent.learn(obs, action, reward, obs_)
            obs = obs_
        scores.append(score)
        eps_history.append(agent.epsilon)

        if i % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print (f"episode {i} score {score} avg score {avg_score} epsilon {agent.epsilon}")

    filename = 'cartpole_naive_dqn.png'
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, scores, eps_history, filename)
