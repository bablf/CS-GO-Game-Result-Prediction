import csv
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim, from_numpy, unsqueeze


from tqdm import tqdm
from sklearn.model_selection import train_test_split


def import_csv(csvfilename): # https://stackoverflow.com/a/53483446
    """
    This funciton takes a csv file and returns a dataset looking like this
    [(match_matrix, winner),  ... ]
    """
    data = [] # Feature
    row_index = 1
    with open(csvfilename, "r", encoding="utf-8", errors="ignore") as scraped:
        reader = csv.reader(scraped, delimiter=';')
        first_row = next(reader)  # skip to second line, because first doent have values
        for row in reader:
            team1, team2 = [],[]
            if row:  # avoid blank lines
                y = float(row[-1]) # take Goldlabel
                winner = 1.0 if y == 2.0 else 0.0
                x = [0.0 if elem == "" else float(elem) for elem in row[5:-1]] # remove "-" und set to float
                t1 = x[:20]
                t2 = x[20:]
                for i in range(0,5):    # group all player features
                    player = t1[i::5]
                    team1.append(player)
                for i in range(0,5):    # group all player features
                    player = t2[i::5]
                    team2.append(player)

            data.append((np.array([team1,team2]),winner))

        return data

class DatasetIterator():
    """
    berechnet den Goldlabel-Satzvektor "on demand"
    """
    def __init__(self, dataset, batchsize):
        self.dataset = dataset
        self.batchsize = batchsize

    def __iter__(self):
        #matches, winners = [], []
        #matches = [item[0] for item in self.dataset]
        #winner = [item[1] for item in self.dataset]
        #matches = np.array(matches)
        #winner = np.array(winner)

        #for i in range(0, len(matches), batchsize):
        #    yield from_numpy(matches[i:i + batchsize]).float(), from_numpy(winner[i:i + batchsize]).float()
        for match, winner in self.dataset:
            yield from_numpy(match).float(), winner

    def __len__(self):
        return len(self.dataset)

                                                        # TODO: dataparam?
def train(model, dataset, epochs, batchsize, learning_rate, model_file):
    opt = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.001) # Optimizer = Stocastic Gradient Descent
    loss_func = nn.MSELoss() # loss-Function = Mean Squard Error

    for epoch in range(1, epochs + 1): # Epochen die iteriert werden soll
        i, running_loss = 0, 0
        random.shuffle(dataset) # Shuffle Matches
        train_iter = DatasetIterator(dataset, batchsize) # für alle Matches (yield)
        model.train()  # turn on training mode
        for match, goldlabel in tqdm(train_iter):
            opt.zero_grad()
            predictions = model(match)
            loss = loss_func(predictions, goldlabel)
            loss.backward()
            opt.step()

        model.eval()  # turn on evaluation mode
        print("\n--- EVALUIERUNG ---")
        test_iter = DatasetIterator(data, data.dev_parses)
        for matches, goldlabels in tqdm(test_iter):
            matches = matches.to(device)
            goldlabels = goldlabels.to(device)
            predictions = model(match)
            print("debug")
            print("predictions: ",predictions)
            loss = loss_func(predictions, goldlabel) # IF ODDS custom Loss funct.
            running_loss += loss.item()
            i += 1
            if i % 2000 == 1999:    # print every 2000 matches
            # TODO: Bestes Model abspeichern
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        print("\nEpoche fertig {}/{},".format(epoch, epochs))


def save_model(model, model_filepath='model.pkl', data_filepath='data_params.pkl'):
    """
    Speichert das model + parameter
    """
    torch.save(model.state_dict(), model_filepath)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        """
        in_channels (python:int) – Number of channels in the input image
        out_channels (python:int) – Number of channels produced by the convolution
        kernel_size (python:int or tuple) – Size of the convolving kernel
        """
        self.Convolution = nn.Conv1d(in_channels=1,out_channels=1,kernel_size=(5,1)) # Filter muss 1x10 sein
        self.Dense64 = nn.Linear(4, 64) # ((Batch_size) x 2x NUMB_FEAT)
        self.Dense16 = nn.Linear(64,16)
        self.Dense1  = nn.Linear(16,1)

    def forward(self, matches):
        print(matches.shape)

        team1, team2 = matches
        team1 = unsqueeze(unsqueeze(team1,0),0)
        team2 = unsqueeze(unsqueeze(team2,0),0)
        print(team1.shape)
        #conv_out = self.Convolution(matches)
        conv_out1 = self.Convolution(team1).view(1,4)
        conv_out2 = self.Convolution(team2).view(1,4)
        # concat the matrices:
        team_feature = torch.cat((conv_out1, conv_out2),0)
        x64 = torch.tanh(self.Dense64(team_feature))
        x16 = torch.tanh(self.Dense16(x64))
        #prediction = nn.Sigmoid()(self.Dense1(x16)) #only one of the two
        prediction = torch.tanh(self.Dense1(x16))
        print(prediction)
        return prediction

if __name__ == "__main__" :
# Values can be changed, to (maybe) improve perfermance a bit
    learning_rate = 0.0001
    epochs = 20
    batchsize = 200

    model_file = "./data/model.pkl"
    print("===== Daten werden gelesen======\n")
    dataset = import_csv("./data/past_matches.csv")
    print("===== Daten eingelesen ======")
    # TODO: train_test_split

    ezBetticus = Model()
    print(ezBetticus.Convolution.weight.shape)
    train(ezBetticus, dataset, epochs, batchsize, learning_rate,  model_file) # train and (will) save the best model EUWEST
