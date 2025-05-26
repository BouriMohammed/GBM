from script_numba.GBM import growth_bounds_matrix
from script_numba.reformulate_weights import gates

import numpy as np
from tqdm import tqdm
import torch
from sklearn.metrics import f1_score

import torch.nn as nn
import torch.optim as optim
import utils_imdb as utils
import pickle
import argparse
import os

import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument('--hidden_size', '-d', type=int, default=64)
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--batch_size', '-b', type=int, default=64)
    parser.add_argument('--num_epochs', '-T', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=1,
                        help='Num layers of Bi-LSTM')
    parser.add_argument('--num_classes', type=int,
                        default=1, help='Num of classes')
    parser.add_argument(
        '--embedding', '-e', choices=['glove50', 'glove300', 'cf'], default='glove300')
    
    parser.add_argument('--portion', '-k', type=float, default=0.0)
    parser.add_argument('--portiona', '-ka', type=float, default=0.0)
    parser.add_argument('--portionb', '-kb', type=float, default=0.0)
    parser.add_argument('--portiong', '-kg', type=float, default=0.0)
    args = parser.parse_args()
    return args


args = parse_args()
hidden_size = args.hidden_size
batch_size = args.batch_size
num_epochs = args.num_epochs
num_layers = args.num_layers
num_classes = args.num_classes
ka = args.portiona
kb = args.portionb
kg = args.portiong


with open('./data/embedding/toc.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

embedding_matrix = np.load('./data/embedding_matrix.npy')

xtrain = np.load('./data/imdb/xtrain.npy')
xtest = np.load('./data/imdb/xtest.npy')

y_train = np.load('./data/imdb/y_train.npy')
y_test = np.load('./data/imdb/y_test.npy')

train_dataset = utils.IMDBDataset(reviews=xtrain, targets=y_train)
valid_dataset = utils.IMDBDataset(reviews=xtest, targets=y_test)

train_data_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=False)
valid_data_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=batch_size, shuffle=False)

x_interval = np.load('./inputs/x_interval.npy')


def tensor_min_max(t):
    tt_min, _ = torch.min(t, axis=0)
    tt_max, _ = torch.max(t, axis=0)
    return torch.stack((tt_min, tt_max), dim=1)


def final_tensor(tensor1, tensor2):
    if torch.isnan(tensor1).all().item() == True:
        t1 = tensor_min_max(tensor2)
        t2 = tensor_min_max(tensor2)

    else:
        t1 = tensor_min_max(tensor1)
        t2 = tensor_min_max(tensor2)

    min_tensor = torch.min(t1, t2)
    max_tensor = torch.max(t1, t2)

    return torch.stack((min_tensor[:, 0], max_tensor[:, 1]), axis=1)


def comparaison(tensor1, tensor2):
    if torch.isnan(tensor1).all().item() == True:
        return tensor2
    else:
        return torch.stack((torch.min(tensor1, tensor2)[:, 0], torch.max(tensor1, tensor2)[:, 1]), axis=1)


class BiLSTM(nn.Module):
    def __init__(self, embedding_matrix, hidden_size, num_layers, num_classes, dropout=0.2):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix), padding_idx=0)

        self.lstm = nn.LSTM(input_size=embedding_matrix.shape[1], hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=0.1 if num_layers == 1 else dropout, bidirectional=True)

        self.fc = nn.Linear(hidden_size * 2, num_classes)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(hidden_size)

        self.cs_final_f = torch.full((hidden_size, 2), float('nan'))
        self.hs_final_f = torch.full((hidden_size, 2), float('nan'))

        self.cs_final_b = torch.full((hidden_size, 2), float('nan'))
        self.hs_final_b = torch.full((hidden_size, 2), float('nan'))

        torch.manual_seed(64)
        # Initialize weights
        self.init_weights()

    def init_weights(self):
        # Initialize LSTM weights to random values
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                # Initialize with Xavier uniform initialization
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        # Initialize linear layer weights to random values
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)  # Set biases to zero

    def forward(self, x, train_mode=True):
        embedded = self.embedding(x)
        # print(embedded.shape)

        lstm_output, state = self.lstm(embedded)

        if train_mode:

            cs = torch.full((batch_size, hidden_size), float('nan'))
            hs = torch.full((batch_size, hidden_size), float('nan'))

            st_h_f = lstm_output[:, -2, :self.lstm.hidden_size]
            st_h_b = lstm_output[:, 1, self.lstm.hidden_size:]

            self.cell_state_forward = final_tensor(cs, state[1][0])
            self.cs_final_f = comparaison(
                self.cs_final_f, self.cell_state_forward)

            self.hidden_state_forward = final_tensor(hs, st_h_f)
            self.hs_final_f = comparaison(
                self.hs_final_f, self.hidden_state_forward)

            self.cell_state_backward = final_tensor(cs, state[1][1])
            self.cs_final_b = comparaison(
                self.cs_final_b, self.cell_state_backward)

            self.hidden_state_backward = final_tensor(hs, st_h_b)
            self.hs_final_b = comparaison(
                self.hs_final_b, self.hidden_state_backward)

        # Concatenate the hidden states of the last timestep from both directions
        hidden_state = torch.cat(
            (lstm_output[:, -1, :self.lstm.hidden_size], lstm_output[:, 0, self.lstm.hidden_size:]), dim=-1)

        out = self.relu(hidden_state)
        out = self.fc(self.dropout(out))

        return out


class Engine:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device

    def lips(self, input_interval, hidden_size, direction):
        weights = [self.model.state_dict()[param].cpu().detach().numpy() for param in [
            'lstm.weight_ih_l0', 'lstm.weight_hh_l0', 'lstm.bias_ih_l0', 'lstm.bias_hh_l0']]
        weights_reverse = [self.model.state_dict()[param].cpu().detach().numpy() for param in [
            'lstm.weight_ih_l0_reverse', 'lstm.weight_hh_l0_reverse', 'lstm.bias_ih_l0_reverse', 'lstm.bias_hh_l0_reverse']]

        if direction == 'f':
            h_interval_f = self.model.hs_final_f.detach().cpu().numpy()
            c_interval_f = self.model.cs_final_f.detach().cpu().numpy()

            Jaccobian_H_X, Jaccobian_H_h, Jaccobian_H_c = growth_bounds_matrix(
                weights, input_interval, h_interval_f, c_interval_f, hidden_size)

        elif direction == 'b':
            h_interval_b = self.model.hs_final_b.detach().cpu().numpy()
            c_interval_b = self.model.cs_final_b.detach().cpu().numpy()

            Jaccobian_H_X, Jaccobian_H_h, Jaccobian_H_c = growth_bounds_matrix(
                weights_reverse, input_interval, h_interval_b, c_interval_b, hidden_size)

        sum_alpha = np.sum(np.sum(Jaccobian_H_X, axis=1), axis=0)
        sum_beta = np.sum(np.sum(Jaccobian_H_h, axis=1), axis=0)
        sum_gama = np.sum(Jaccobian_H_c[0])

        return sum_alpha, sum_beta, sum_gama

    def train(self, data_loader, epoch, num_epochs, input_interval, hidden_size, ka, kb, kg):
        """
        This is model training for one epoch
        data_loader:  this is torch dataloader, just like dataset but in torch and divided into batches
        model : lstm
        optimizer : torch optimizer : adam
        device:  cuda or cpu
        """
        # Set model to training mode
        self.model.train()
        # Move the model to the desired device
        self.model.to(self.device)

        train_losses = []
        train_targets_all = []
        train_outputs_all = []
        for batch in tqdm(data_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False):
            reviews = batch['review'].to(self.device)
            # Add one dimension for BCEWithLogitsLoss
            targets = batch['target'].unsqueeze(1).to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(reviews, train_mode=True)

            loss_ce = nn.BCEWithLogitsLoss()(outputs, targets)

            alpha_f,  beta_f, gama_f = self.lips(
                input_interval, hidden_size, 'f')
            alpha_b,  beta_b, gama_b = self.lips(
                input_interval, hidden_size, 'b')

            loss = (1-ka-kb-kg)*loss_ce + ka*(alpha_f+alpha_b) + \
                kb*(beta_f+beta_b) + kg*(gama_f+gama_b)

            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.25)

            self.optimizer.step()
            train_losses.append(loss.item())
            train_targets_all.extend(targets.cpu().numpy())
            train_outputs_all.extend(torch.sigmoid(
                outputs).cpu().detach().numpy())

        train_targets_all = np.array(train_targets_all)
        train_outputs_all = np.array(train_outputs_all)
        train_outputs_all = np.round(train_outputs_all)
        train_accuracy = np.mean(train_outputs_all == train_targets_all)
        train_f1 = f1_score(train_targets_all, train_outputs_all)

        # Return the average loss
        return np.mean(train_losses), train_accuracy, train_f1

    def evaluate(self, data_loader, input_interval, hidden_size, ka, kb, kg):
        # Evaluation loop
        self.model.eval()
        dev_losses = []
        with torch.no_grad():
            dev_targets_all = []
            dev_outputs_all = []

            alpha_f,  beta_f,  gama_f = self.lips(
                input_interval, hidden_size, 'f')
            alpha_b,  beta_b,  gama_b = self.lips(
                input_interval, hidden_size, 'b')

            for batch in tqdm(data_loader, desc='Evaluating', leave=False):
                reviews = batch['review'].to(self.device)
                targets = batch['target'].unsqueeze(1).to(self.device)

                outputs = self.model(reviews, train_mode=False)

                loss_dev_ce = nn.BCEWithLogitsLoss()(outputs, targets)
                loss_dev = (1-ka-kb-kg)*loss_dev_ce + ka*(alpha_f +
                                                          alpha_b) + kb*(beta_f+beta_b) + kg*(gama_f+gama_b)

                predicted = torch.round(torch.sigmoid(outputs))

                dev_losses.append(loss_dev.item())
                dev_targets_all.extend(targets.cpu().numpy())
                dev_outputs_all.extend(predicted.cpu().numpy())

        dev_loss = np.mean(dev_losses)
        dev_targets_all = np.array(dev_targets_all)
        dev_outputs_all = np.array(dev_outputs_all)
        dev_accuracy = np.mean(dev_outputs_all == dev_targets_all)
        dev_f1 = f1_score(dev_targets_all, dev_outputs_all)

        return dev_loss, dev_accuracy, dev_f1


model = BiLSTM(embedding_matrix, hidden_size, 1, 1)
DEVICE = torch.device('cuda')
model.to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

eng = Engine(model, optimizer, device=DEVICE)


def lipschitz(model, input_interval, h_interval, c_interval, hidden_size, reverse):
    if reverse:
        weights = [model[param].cpu().detach().numpy() for param in ['lstm.weight_ih_l0_reverse',
                                                                     'lstm.weight_hh_l0_reverse', 'lstm.bias_ih_l0_reverse', 'lstm.bias_hh_l0_reverse']]
    else:
        weights = [model[param].cpu().detach().numpy() for param in [
            'lstm.weight_ih_l0', 'lstm.weight_hh_l0', 'lstm.bias_ih_l0', 'lstm.bias_hh_l0']]

    Jaccobian_H_X, Jaccobian_H_h, Jaccobian_H_c = growth_bounds_matrix(
        weights, input_interval, h_interval, c_interval, hidden_size)

    sum_alpha = np.sum(np.sum(Jaccobian_H_X, axis=1))
    sum_beta = np.sum(np.sum(Jaccobian_H_h, axis=1))
    sum_gama = np.sum(Jaccobian_H_c[0])

    return sum_alpha, sum_beta, sum_gama


# Initialize variables for early stopping
best_dev_loss = float('inf')
patience = 7
counter = 0
best_model_state = None

# Initialize intervals
best_h_interval = None
best_c_interval = None


print('----------- Training Model -----------\n')

for epoch in range(num_epochs):
    print(f'----------- Epoch {epoch+1} -----------\n')

    # Train one epoch
    train_loss, train_accuracy, train_f1 = eng.train(
        train_data_loader, epoch, num_epochs, x_interval, hidden_size, ka, kb, kg)

    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train F1 Score: {train_f1:.4f}')

    # Validate
    dev_loss, dev_accuracy, dev_f1 = eng.evaluate(
        valid_data_loader, x_interval, hidden_size, ka, kb, kg)

    print(f'Epoch {epoch + 1}/{num_epochs}, Dev Loss: {dev_loss:.4f}, Dev Accuracy: {dev_accuracy:.4f}, Dev F1 Score: {dev_f1:.4f}')

    # Early stopping
    if dev_loss < best_dev_loss:
        best_dev_loss = dev_loss
        best_model_state = model.state_dict()
        best_c_interval = [model.cs_final_f.detach().cpu(
        ).numpy(), model.cs_final_b.detach().cpu().numpy()]
        best_h_interval = [model.hs_final_f.detach().cpu(
        ).numpy(), model.hs_final_b.detach().cpu().numpy()]

        counter = 0
    else:
        counter += 1
        print('EarlyStopping step: ', counter)
        if counter >= patience:
            print(
                f'Early stopping at epoch {epoch + 1} as no improvement was seen in validation loss.')
            break

    sum_alpha_f, sum_beta_f, sum_gama_f = lipschitz(
        best_model_state, x_interval, best_h_interval[0], best_c_interval[0], hidden_size, reverse=False)
    sum_alpha_b, sum_beta_b, sum_gama_b = lipschitz(
        best_model_state, x_interval, best_h_interval[1], best_c_interval[1], hidden_size, reverse=True)

    print(f'forward pass : {sum_alpha_f}\t_\t{sum_beta_f}\t_\t{sum_gama_f}')
    print(f'backward pass : {sum_alpha_b}\t_\t{sum_beta_b}\t_\t{sum_gama_b}')

folder_path = f'./model/imdb/bilstm/{ka}_{kb}/'

if os.path.exists(folder_path):

    if best_model_state is not None:
        torch.save(best_model_state, os.path.join(folder_path, 'model.pth'))
    # Save the hidden state interval and cell state interval
    np.save(os.path.join(folder_path,'h_interval.npy'), np.array(best_h_interval))
    np.save(os.path.join(folder_path,'c_interval.npy'), np.array(best_c_interval))

else:
    os.makedirs(folder_path)
    if best_model_state is not None:
        torch.save(best_model_state, os.path.join(folder_path, 'model.pth'))
    # Save the hidden state interval and cell state interval
    np.save(os.path.join(folder_path,'h_interval.npy'), np.array(best_h_interval))
    np.save(os.path.join(folder_path,'c_interval.npy'), np.array(best_c_interval))
