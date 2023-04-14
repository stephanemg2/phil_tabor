import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        # fully connected layer
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        # on passe une liste de 8 vecteurs d'observation
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

#retourne les array estimations d'action en fonction des etats envoyés
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # on recupere les probas d'actions apres les 2 fully connected network (en fonction activation relu) qui ont pris les states
        actions = self.fc3(x)
        # pas de fonction d'activation sur fc3
        return actions


class Agent():
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, max_mem_size=100000, eps_end=0.01, eps_dec=5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.Q_eval = DeepQNetwork(
            self.lr, n_actions=n_actions, input_dims=input_dims, fc1_dims=256, fc2_dims=256)
        # on utilise pas les deque pour le buffer de transitions
        self.state_memory = np.zeros(
            (self.mem_size, *input_dims), dtype=np.float32)
        # buffer pour nouveaux states que l'agent rencontre
        self.new_state_memory = np.zeros(
            (self.mem_size, *input_dims), dtype=np.float32)
        # buffer pour les actions
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        # buffer pour les rewards
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        # buffer pour le terminal state, si le jeu est fini, terminal state est 0
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

    # fonction pour stocker la transition dans la mémoire de l'agent
    # state_ nouvel etat
    def store_transition(self, state, action, reward, state_, done):
        # index de la premiere zone inoccupée de la mémoire
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            # on recupere le state lié a l'observation en envoyant l'observation a la device (gpu)
            state = T.tensor(np.array(observation)).to(self.Q_eval.device)
            # on passe le state au DQN qui retourne une liste d'actions possibles / valeur
            actions = self.Q_eval.forward(state)
            # on recupere l'action avec la valorisation maximale
            action = T.argmax(actions).item()
        # sinon on explore avec une ramdon action appartement au champ d'actions du jeu
        else:
            action = np.random.choice(self.action_space)
        return action

#fonction d'apprentissage, envoie des zones aléatoires de la mémoire d'apprentissage au DQN
    def learn(self):
        # on apprends uniquement quand le buffer commence a etre rempli
        if self.mem_cntr < self.batch_size:
            return
        #pytorch specific. on reset le gradient de l'optimizer
        self.Q_eval.optimizer.zero_grad()
#on prends la position de la derniere zone de mémoire remplie
        max_mem = min(self.mem_cntr, self.mem_size)
        #on randomise les indices du batch array
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        #on envoie les echantillons de positions, ordonnées selon le tableau d'indices aléatoire
        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
        #le batch d'action n'a pas besoin d'etre un Tensor
        action_batch = self.action_memory[batch]
        #on ajuste l'estimation pour les valeurs qu'on a vraiment prises
        #donc on decoupe l'array d'etats du batch uniquement pour les actions stockées correspondantes pour l'estimation de la valorisation
        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        #on veut les estimations de valorisation pour le prochain etat
        #cette fonction serait appliquée sur un NN target mais la on en a pas
        q_next = self.Q_eval.forward(new_state_batch)
        #les valorisation du state terminal sont de 0
        q_next[terminal_batch] = 0
        #on calcule les valeurs target pour la mise a jour de l'estimation de valorisation, gamma est le facteur discount
        #on prends le Tensor max de l'evaluation du prochain state dans la dimension d'action, premier element car T.max donne la valeur,index
        #purely greedy action, pour maj de la fonction de loss
        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]
        #fonction d'estimation entre valeur prédite et valeur target obtenue
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        #back propagation et optimisation
        loss.backward()
        self.Q_eval.optimizer.step()
        #a chaque étape d'apprentissage on decroit le facteur d'exploration epsilon
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min


        

