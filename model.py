class DatasetIterator():
    """
    berechnet den Goldlabel-Satzvektor "on demand"
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        for match, winner in self.dataset:
            yield torch.from_numpy(match).long(), winner

    def __len__(self):
        return len(self.dataset)


def train(model, dataset, epochs, learning_rate, model_file, dataparam_file):
    #opt = optim.SGD(model.parameters(), lr=learning_rate)
    opt =  0 # Optimizer
    loss_func = 0 # Lossfunct gradient Descent ()
    for epoch in range(1, epochs + 1): # Epochen die iteriert werden soll
        random.shuffle(dataset) # Shuffle Matches
        train_iter, goldlabel = DatasetIterator(dataset) # für alle Matches (yield)
        model.train()  # turn on training mode
        for match, goldlabel in tqdm(train_iter):
            opt.zero_grad()
            predictions = model(match)
            loss = loss_func(prediction, goldlabel)
            loss.backward()
            opt.step()

        model.eval()  # turn on evaluation mode
        span_wrong = 0
        print("\n--- EVALUIERUNG ---")
        test_iter = DatasetIterator(data, data.dev_parses)
        for match, goldlabel in tqdm(test_iter):
            match = match.to(device)
            goldlabel = goldlabel.to(device)
            prediction = model(match)
            loss = loss_func(prediction, goldlabel) # IF ODDS custom Loss funct.

        if (span_wrong < min_span_wrong or min_span_wrong == -1):
            save_model(model, model_file, dataparam_file)
            min_span_wrong = span_wrong

        print("\nEpoche fertig {}/{}, Fehlerhafte Spans: {}, Highscore {}".format(epoch, epochs, span_wrong,
                                                                                  min_span_wrong))




def save_model(model, model_filepath='model.pkl', data_filepath='data_params.pkl'):
    """
    Speichert das model + parameter
    """
    torch.save(model.state_dict(), model_filepath)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.Convolution = nn.Conv3D
        #self.Dense =
        # sigmoid OR tanh activation beides Testen!

    def forward(self, match):
        out_conv = self.embedding(match) # out_conv.shape should be (2xNUMB_FEAT)
        d64 = self.Dense(64)
        d16 = self.Dense(16)
        prediciton = self.Dense(1)
