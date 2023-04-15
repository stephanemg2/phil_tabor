import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(DeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        # conv1 32 filtres, kernel 8, stride 4 avec en entrée
        # input dims[0] nombre de canaux dans nos images 4 frames de 1 canal pour nos greyscales
        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4)
        # connection aux 32 filtres du conv1 et 64 en sortie. kernel de 8, stride de 2.
        # les tailles des kernels diminuent avec le niveau d'abstraction du convolutional
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        # on calcule le nombre d'inputs pour le premier FC layer
        fc_input_dims = self.calculate_conv_output_dims(input_dims)
        # premier full connected layer output de 512
        self.fc1 = nn.Linear(fc_input_dims, 512)
        # le deuxieme a pour output le nombre d'actions possibles
        self.fc2 = nn.Linear(512, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        # loss mean squared error
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def calculate_conv_output_dims(self, input_dims):
        state = T.zeros(1, *input_dims)
        # on le passe au premier convulational layer
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))

    def forward(self, state):
        conv1 = F.relu(self.conv1(state))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        # conv3 shape is batch size * x_filters * H * W
        # on fait un reshape avec batchsize, et le reste des dimensions pour aplatir
        # la dimension batch size est faite pour le jeu d'entrainement du tensor,
        # les autres dimensions sont pour le nombre d'entrées du fully connected
        conv_state = conv3.view(conv3.size()[0], -1)
        flat1 = F.relu(self.fc1(conv_state))
        actions = self.fc2(flat1)
        return actions

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save({'model_state_dict': self.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': self.loss},
               self.checkpoint_file)

        # T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        checkpoint = T.load(self.checkpoint_file)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.loss = checkpoint['loss']
