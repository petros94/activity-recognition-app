import torch

device="cpu"

class HARDataset(torch.utils.data.Dataset):
  def __init__(self, X, y) -> None:
    super().__init__()
    self.X = X
    self.y = y

  def __len__(self):
    return len(self.X)

  def __getitem__(self, idx):
    return (
        torch.tensor(self.X[idx], dtype=torch.float),
        torch.tensor(self.y[idx], dtype=torch.long).view(-1)
    )

def collate_pad(batch):
  in_ = []
  out_ = []
  seq_len = []
  for x,y in batch:
    in_.append(x)
    out_.append(y)
    seq_len.append(len(x))

  return torch.nn.utils.rnn.pad_sequence(in_).to(device), torch.tensor(out_).to(device), seq_len


import torch
import numpy as np
import time
from torch.nn import functional as F
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score, f1_score


def train_model(model, train_set, validation_set, max_epochs, batch_size, lr=0.002, save_path=None):
    evaluation_data = {
        'train_loss': [],
        'validation_loss': [],
    }

    # Define model and loss functions
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Define dataloaders
    dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, collate_fn=collate_pad)
    validation_dataloader = torch.utils.data.DataLoader(validation_set, batch_size=128, collate_fn=collate_pad)

    # Train loop
    best_valid_loss = 99
    best_model = None
    for epoch in range(max_epochs):
        start_time = time.time()
        model.train()

        ## Train in batches
        for batch, (x, y, seq_len_x) in enumerate(dataloader):
            if batch % 100 == 0:
                print(f'Epoch {epoch} - Batch {batch} of {int(len(train_set) / batch_size)}')
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)

            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

        ## Evaluate performance on train set
        print(f'Epoch {epoch} - Evaluating train set')
        with torch.no_grad():
            model.eval()
            epoch_loss = 0
            loss_counter = 0
            for batch, (x, y, seq_len_x) in enumerate(dataloader):
                x = x.to(device)
                y = y.to(device)

                y_pred = model(x)

                loss = criterion(y_pred, y)

                epoch_loss += loss.item()
                loss_counter += 1

            epoch_loss /= loss_counter

        ## Evaluate performance on validation set
        print(f'Epoch {epoch} - Evaluating validation set')
        with torch.no_grad():
            model.eval()
            valid_loss = 0
            loss_counter = 0
            for batch, (x, y, seq_len_x) in enumerate(validation_dataloader):
                x = x.to(device)
                y = y.to(device)

                y_pred = model(x)

                loss = criterion(y_pred, y)

                valid_loss += loss.item()
                loss_counter += 1

            valid_loss /= loss_counter

        ## update evaluation data
        evaluation_data['train_loss'].append(epoch_loss)
        evaluation_data['validation_loss'].append(valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model = model.state_dict()

            if save_path is not None:
                print("saving model")
                torch.save(best_model, save_path)

        print(
            "Train Epoch {}: Time {}s |  Loss - {} | Validation loss - {}".format(epoch, int(time.time() - start_time),
                                                                                  epoch_loss, valid_loss))
        print("----------------------------------------")
    return evaluation_data


def evaluate(model, test_set):
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=128, collate_fn=collate_pad)

    y_pred = []
    y_test = []
    with torch.no_grad():
        model.eval()
        valid_loss = 0
        loss_counter = 0
        for batch, (x, y, seq_len_x) in enumerate(test_dataloader):
            x = x.to(device)
            y = y.to(device)

            y_pred.extend(F.softmax(model(x)).argmax(-1).cpu().numpy())
            y_test.extend(y.cpu().numpy())

    print(confusion_matrix(y_test, y_pred))
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred)
    print("Accuracy: {}".format(accuracy_score(y_test, y_pred)))
    print("f1: {}".format(f1_score(y_test, y_pred, average='weighted')))
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))