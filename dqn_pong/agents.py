import numpy as np
import torch as T
from deep_q_network import DeepQNetwork
from dueling_deep_q_network import DuelingDeepQNetwork
from replay_memory import ReplayBuffer


class Agent():
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims, mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
                 replace=1000, algo=None, env_name=None, chkpt_dir='tmp/dqn'):
        self.chkpt_dir = chkpt_dir
        self.env_name = env_name
        self.algo = algo
        self.replace = replace
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.batch_size = batch_size
        self.mem_size = mem_size
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.replace_target_cnt = replace
        self.lr = lr
        self.epsilon = epsilon
        self.gamma = gamma
        # choix d'action en cas d'epsilon random
        self.action_space = [i for i in range(self.n_actions)]
        # pour maj du target network
        self.learn_step_cntr = 0
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

    def choose_action(self, observation):
        raise NotImplementedError

    def learn(self):
        raise NotImplementedError

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(self):
        state, action, reward, new_state, done = self.memory.sample_memory(self.batch_size)
        # on stocke ces samples en tensors dans la device
        states = T.tensor(state).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        return states, actions, rewards, states_, dones

    # update des weights du target network
    def replace_target_network(self):
        if self.learn_step_cntr % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()


class DQNAgent(Agent):
    # on envoie tous les parametres au parent
    def __init__(self, *args, **kwargs):
        super(DQNAgent, self).__init__(*args, **kwargs)
        self.q_eval = DeepQNetwork(self.lr, self.n_actions, input_dims=self.input_dims,
                                   name=self.env_name + '_' + self.algo + '_q_eval.pth', chkpt_dir=self.chkpt_dir)
        self.q_next = DeepQNetwork(self.lr, self.n_actions, input_dims=self.input_dims,
                                   name=self.env_name + '_' + self.algo + '_q_next.pth', chkpt_dir=self.chkpt_dir)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            # on envoie une dimension batch size * input_dims
            # state = T.tensor(np.asarray(observation), dtype=T.float).to(self.q_eval.device)
            state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
            # on feed le NN avec le tensor contenant l'observation
            actions = self.q_eval.forward(state)
            # on choisit la best value en utilisant la fonction argmax du tensor
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

        # learns after batch size is full

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        self.q_eval.optimizer.zero_grad()
        # les valeurs du q_eval vont changer, elles sont sauvegardées dans le target network
        self.replace_target_network()
        states, actions, rewards, states_, dones = self.sample_memory()

        # create a liste of indices [0...32]
        indices = np.arange(self.batch_size)

        # feed the states to q_eval and get the actions values  that he tooked for those states
        # forward(states) gives a tensor.shape (32,6) for all values estimated
        # actions are an array of indices of actions tooked by the agent shape 32 (batch size)
        # we need to dereference for each indice of the return the number line, indice of action tooked
        # so [indices, actions] to get for each line the value of the action that was taken
        q_pred = self.q_eval.forward(states)[indices, actions]

        # for the next action to take, q_next.max gives a tuple (32,) [0] : actions values
        # [1] : indices
        # so we take the values of the 32 actions that will be choosed for the agent
        q_next = self.q_next.forward(states_).max(dim=1)[0]
        # 1 - take the maximal actions in the next states
        # 2 - move the estimates actions bias toward those predictions
        # if next state is terminal, target = reward
        # dones are the indices of the terminal values
        # every q_next entry where dones is true, the q_next referenced value = 0.0

        q_next[dones] = 0.0
        # if dones.any()==1:
        #     print(q_next, dones)
        # we calculate the target value for the prediction
        q_target = rewards + self.gamma * q_next
        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        # to update the target network on the right frequency
        self.learn_step_cntr += 1
        self.decrement_epsilon()


class DDQNAgent(Agent):
    def __init__(self, *args, **kwargs):
        super(DDQNAgent, self).__init__(*args, **kwargs)
        self.q_eval = DeepQNetwork(self.lr, self.n_actions, input_dims=self.input_dims,
                                   name=self.env_name + '_' + self.algo + '_q_eval.pth', chkpt_dir=self.chkpt_dir)
        self.q_next = DeepQNetwork(self.lr, self.n_actions, input_dims=self.input_dims,
                                   name=self.env_name + '_' + self.algo + '_q_next.pth', chkpt_dir=self.chkpt_dir)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            # on envoie une dimension batch size * input_dims
            # state = T.tensor(np.asarray(observation), dtype=T.float).to(self.q_eval.device)
            state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
            # on feed le NN avec le tensor contenant l'observation
            actions = self.q_eval.forward(state)
            # on choisit la best value en utilisant la fonction argmax du tensor
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

        # learns after batch size is full

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        self.q_eval.optimizer.zero_grad()
        self.replace_target_network()
        states, actions, rewards, states_, dones = self.sample_memory()
        indices = np.arange(self.batch_size)

        q_pred = self.q_eval.forward(states)[indices, actions]
        q_next = self.q_next.forward(states_)
        q_eval = self.q_eval.forward(states_)

        # calculates the max action with q_eval network
        max_actions = T.argmax(q_eval, dim=1)
        q_next[dones] = 0.0
        # we calculate the target value for the prediction
        q_target = rewards + self.gamma * q_next[indices, max_actions]
        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        # to update the target network on the right frequency
        self.learn_step_cntr += 1
        self.decrement_epsilon()

class DuelingDQNAgent(Agent):
    def __init__(self, *args, **kwargs):
        super(DuelingDQNAgent, self).__init__(*args, **kwargs)
        self.q_eval = DuelingDeepQNetwork(self.lr, self.n_actions, input_dims=self.input_dims,
                                          name=self.env_name + '_' + self.algo + '_q_eval.pth',
                                          chkpt_dir=self.chkpt_dir)
        self.q_next = DuelingDeepQNetwork(self.lr, self.n_actions, input_dims=self.input_dims,
                                          name=self.env_name + '_' + self.algo + '_q_next.pth',
                                          chkpt_dir=self.chkpt_dir)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            # on envoie une dimension batch size * input_dims
            # state = T.tensor(np.asarray(observation), dtype=T.float).to(self.q_eval.device)
            state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
            # we take only the advantage nn calculated
            _, advantage = self.q_eval.forward(state)
            action = T.argmax(advantage).item()
        else:
            action = np.random.choice(self.action_space)
        return action

        # learns after batch size is full

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        self.q_eval.optimizer.zero_grad()
        # les valeurs du q_eval vont changer, elles sont sauvegardées dans le target network
        self.replace_target_network()
        states, actions, rewards, states_, dones = self.sample_memory()

        # create a liste of indices [0...32]
        indices = np.arange(self.batch_size)
        # values and advantage for the states
        V_s, A_s = self.q_eval.forward(states)
        V_s_, A_s_ = self.q_next.forward(states_)
        q_pred = T.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]
        q_next = T.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True))).max(dim=1)[0]
        q_next[dones] = 0.0
        q_target = rewards + self.gamma * q_next
        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        # to update the target network on the right frequency
        self.learn_step_cntr += 1
        self.decrement_epsilon()