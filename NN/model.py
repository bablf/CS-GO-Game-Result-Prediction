import csv
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim, from_numpy, unsqueeze


from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import scale

class DatasetIterator():
    """
    berechnet den Goldlabel-Satzvektor "on demand"
    """
    def __init__(self, dataset, batchsize):
        self.matches, self.winners = dataset
        self.batchsize = batchsize

    def __iter__(self):
        # !!!!!!!!!!!!!!!!  DO NOT DELETE COMMENTS !!!!!!!!!!!!!!!!!!
        #matches, winners = [], []
        #matches = [item[0] for item in self.dataset]
        #winner = [item[1] for item in self.dataset]
        #matches = np.array(matches)
        #winner = np.array(winner)

        #for i in range(0, len(matches), batchsize):
        #    yield from_numpy(matches[i:i + batchsize]).float(), from_numpy(winner[i:i + batchsize]).float()

        for match, winner in zip(self.matches, self.winners):
            yield from_numpy(match).float(), from_numpy(winner).float()

    def __len__(self):
        return len(self.matches)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        """
        in_channels (python:int) – Number of channels in the input image
        out_channels (python:int) – Number of channels produced by the convolution
        kernel_size (python:int or tuple) – Size of the convolving kernel
        """
        self.Convolution = nn.Conv1d(in_channels=1,out_channels=1,kernel_size=(5,1)) # Filter muss 1xNUMB_PLAYER sein
        self.Dense64 = nn.Linear(84, 64) # (2x NUMB_FEAT)
        self.Dense16 = nn.Linear(64,16)
        self.Dense1  = nn.Linear(16,1)
        self.Sigmoid = nn.Sigmoid()
        self.Softmax = nn.Softmax(dim=0)
        self.Dropout = nn.Dropout(p=0.2)


    def forward(self, matches):
        team1, team2 = matches
        team1 = unsqueeze(unsqueeze(team1,0),0)
        team2 = unsqueeze(unsqueeze(team2,0),0)
        #conv_out = self.Convolution(matches)
        conv_out1 = self.Convolution(team1).view(1,84)
        conv_out2 = self.Convolution(team2).view(1,84)
        # concat the matrices:
        team_feature = torch.cat((conv_out1, conv_out2),0)
        drop_feat = self.Dropout(team_feature)

        x64 = torch.tanh(self.Dense64(drop_feat))
        drop1 = self.Dropout(x64)

        x16 = torch.tanh(self.Dense16(drop1))
        drop2 = self.Dropout(x16)

        #prediction = self.Sigmoid(self.Dense1(x16).view(2))
        #prediction = torch.tanh(self.Dense1(drop2).view(2))
        prediction = self.Softmax(self.Dense1(drop2).view(2))
        return prediction # (P(Home-Won), P(Away-Won))

    def save_model(self, model, model_filepath='model.pkl'):
        """
        Speichert das model + parameter
        """
        print("===== Model wird gespeichert ======\n")
        torch.save(model.state_dict(), model_filepath)
