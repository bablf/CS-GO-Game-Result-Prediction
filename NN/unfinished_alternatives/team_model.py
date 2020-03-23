import csv
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim, from_numpy, unsqueeze


from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale


def import_csv(csvfilename): # https://stackoverflow.com/a/53483446
    """
    This funciton takes a csv file and returns a dataset looking like this
    [(match_matrix, winner),  ... ]
    """
    data_x, data_y = [],[] # Feature
    with open(csvfilename, "r", encoding="utf-8", errors="ignore") as scraped:
        reader = csv.reader(scraped, delimiter=';')
        first_row = next(reader)  # skip to second line, because first doent have values
        for row in reader:
            team1, team2 = [],[]
            if row:  # avoid blank lines
                y = float(row[-1]) # take Goldlabel
                winner = [0.0, 1.0] if y == 2.0 else [1.0, 0.0]
                x = [0.0 if elem == "" else float(elem) for elem in row[5:-1]] # remove "-" und set to float
                t1 = x[:20]
                t2 = x[20:]
                for i in range(0,5):    # group all player features
                    player = t1[i::5]
                    team1 += player
                for i in range(0,5):    # group all player features
                    player = t2[i::5]
                    team2 += player

            data_x.append(np.array([team1, team2]))
            data_y.append(np.array(winner))
        return np.array(data_x), np.array(data_y)

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

        for match, winner in zip(self.matches,self.winners):
            yield from_numpy(match).float(), from_numpy(winner).float()

    def __len__(self):
        return len(self.matches)


def converter(preds):
    if preds[0] > preds[1]: # home team won
        return torch.tensor([1.0, 0.0])
    elif preds[0] < preds[1]: # away team won
        return torch.tensor([0.1, 1.0])
    else: return torch.tensor([2.0,2.0]) # same size no correct clasification

def calc_acc(preds, goldlabel):
    right = 0
    for pred, gold in zip(preds, goldlabel):
        if torch.equal(pred, gold):
            right +=1
    return right/len(preds)

                                                        # TODO: dataparam?
def train(model, data_x, data_y, epochs, batchsize, learning_rate, model_filepath):

    opt = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.0001) # Optimizer = Stocastic Gradient Descent
    loss_func = nn.MSELoss(reduction='sum') # loss-Function = Sum Squard Error
    old_accuracy = 0.0
    for epoch in range(1, epochs + 1): # Epochen die iteriert werden soll
        i, running_loss = 0, 0
        #list_train_preds, list_train_goldlabel = [],[]
        x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.33) # Shuffle Matches
        train_iter = DatasetIterator((x_train, y_train), batchsize) # für alle Matches (yield)

        model.train()  # turn on training mode
        for match, goldlabel in tqdm(train_iter):
            opt.zero_grad()
            predictions = model(match)
            #list_train_preds.append(predictions)
            #list_train_goldlabel.append(goldlabel)
            loss = loss_func(predictions, goldlabel)
            loss.backward()
            opt.step()
        #train_accuracy = calc_acc(list_train_preds, list_train_goldlabel)
        #print("Train accuracy = ", train_accuracy)
        model.eval()  # turn on evaluation mode
        print("\n--- EVALUIERUNG ---")
        test_iter = DatasetIterator((x_test,y_test),batchsize)

        list_preds, list_goldlabel = [],[]
        for match, goldlabel in tqdm(test_iter):
            # matches = matches.to(device)
            # goldlabels = goldlabels.to(device)

            predictions = model(match)
            loss = loss_func(predictions, goldlabel)
            running_loss += loss.item()

            list_preds.append(converter(predictions))
            list_goldlabel.append(goldlabel)
            i += 1
            if i % 2000 == 1999:    # print every 2000 matches
            # TODO: Bestes Model abspeichern
                accuracy = calc_acc(list_preds, list_goldlabel)
                list_preds, list_goldlabel = [],[]
                print('[%d, %5d] loss: %.3f acc: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000, accuracy))
                running_loss = 0.0
                if accuracy > old_accuracy:
                    save_model(model, model_filepath)
                    old_accuracy = accuracy

        print("\nEpoche fertig {}/{},".format(epoch, epochs))


def save_model(model, model_filepath='model.pkl'):
    """
    Speichert das model + parameter
    """
    print("===== Model wird gespeichert ======\n")
    torch.save(model.state_dict(), model_filepath)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        """
        in_channels (python:int) – Number of channels in the input image
        out_channels (python:int) – Number of channels produced by the convolution
        kernel_size (python:int or tuple) – Size of the convolving kernel
        """
        #self.Convolution = nn.Conv1d(in_channels=1,out_channels=1,kernel_size=(5,1)) # Filter muss 1x10 sein
        self.Dense64 = nn.Linear(20, 64)
        self.Dense32 = nn.Linear(64,32)
        self.Dense16 = nn.Linear(32,16)
        self.Dense1  = nn.Linear(16,1)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, match):
        x64 = torch.tanh(self.Dense64(match))
        x32 = torch.tanh(self.Dense32(x64))
        x16 = torch.tanh(self.Dense16(x32))
        #x = torch.tanh(self.Dense1(x16)).view(2)
        #prediction = self.Sigmoid(self.Dense1(x16).view(2))
        prediction = torch.tanh(self.Dense1(x16).view(2))
        #print(prediction)
        #prediction = self.Sigmoid(x)
        #prediction = nn.Sigmoid()(self.Dense1(x16)) #only one of the two
        #return torch.tensor(max(prediction))
        return prediction # (P(Home-Won), P(Away-Won))

if __name__ == "__main__" :
# Values can be changed, to (maybe) improve perfermance a bit
    learning_rate = 0.001
    epochs = 10
    batchsize = 200 # not implemented

    model_filepath = "./data/model.pkl"
    print("===== Daten werden gelesen======\n")
    data_x, data_y = import_csv("./data/past_matches.csv")
    print("===== Daten eingelesen ======")
    # Start training
    ezBetticus = Model()
    train(ezBetticus, data_x, data_y, epochs, batchsize, learning_rate,  model_filepath) # train and (will) save the best model EUWEST
