import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import random
import pickle
import os
import argparse
import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument('--num_kernels', '-d', type=int, default=128)
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', '-b', type=int, default=64)
    parser.add_argument('--num_epochs', '-T', type=int, default=1)
    parser.add_argument('--num_classes', type=int,
                        default=1, help='Num of classes')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.0001)
    
    parser.add_argument(
        '--embedding', '-e', choices=['glove50', 'glove300', 'cf'], default='glove300')
    
    parser.add_argument('--portion', '-k', type=float, default=0.0)
    parser.add_argument('--kernel_sizes', default=(3, 4, 5))
    args = parser.parse_args()
    return args


args = parse_args()
num_kernels = args.num_kernels
seed = args.seed
batch_size = args.batch_size
num_epochs = args.num_epochs
num_classes = args.num_classes
k = args.portion
kernel_sizes = args.kernel_sizes
learning_rate = args.learning_rate

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom IMDBDataset class
class IMDBDataset:
    def __init__(self, reviews, targets):
        """
        Arguments:
        reviews: a numpy array of reviews (tokenized and padded)
        targets: a numpy array of target labels

        """
        self.reviews = reviews
        self.targets = targets

    def __len__(self):
        # Return length of the dataset
        return len(self.reviews)

    def __getitem__(self, index):
        # Given an index, return the review and target of that index as tensors
        review = torch.tensor(self.reviews[index, :], dtype=torch.long)
        target = torch.tensor(self.targets[index], dtype=torch.float)

        return {'review': review, 'target': target}

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Load embedding matrix and dataset
embedding_matrix = np.load('./data/embedding/embedding_matrix.npy')

xtrain = np.load('./data/imdb/xtrain.npy')
xtest = np.load('./data/imdb/xtest.npy')

y_train = np.load('./data/imdb/y_train.npy')
y_test = np.load('./data/imdb/y_test.npy')

# Load datasets
train_dataset = IMDBDataset(reviews=xtrain, targets=y_train)
valid_dataset = IMDBDataset(reviews=xtest, targets=y_test)

train_data_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
valid_data_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=128, shuffle=False)

# TextCNN Model
class TextCNN(nn.Module):
    def __init__(self, embedding_matrix, num_classes, num_kernels, kernel_sizes):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), padding_idx=0)
        embed_dim = embedding_matrix.shape[1]  # Infer embed_dim from the embedding matrix
        
        # Create Conv1d layers for different kernel sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_kernels, kernel_size=k, padding=0) for k in kernel_sizes
        ])
        self.fc = nn.Linear(len(kernel_sizes) * num_kernels, num_classes)
        self.dropout = nn.Dropout(0.2)

    def init_weight(self):
        for conv in self.convs:
            torch.nn.init.normal_(conv.weight, std=0.1)
            torch.nn.init.constant_(conv.bias, 0.1)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.constant_(self.fc.bias, 0.1)


    
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        x = self.embedding(x).transpose(1, 2)  # Shape: (batch_size, embed_dim, seq_len)
        x = [F.relu(conv(x)) for conv in self.convs] # Convolution and ReLU activation
        # print('shape before pooling:\t',x[0].shape,'\t',x[1].shape,'\t',x[2].shape,'\t')
        x = [torch.max_pool1d(c, c.size(2)).squeeze(2) for c in x]  # Max pooling over time
        # print('shape after pooling:\t',x[0].shape,'\t',x[1].shape,'\t',x[2].shape,'\t')
        x = torch.cat(x, 1)  # Concatenate pooled features
        concat = x.clone().detach()
        x = self.dropout(x)
        return self.fc(x), concat

# Initialize Model

embed_dim = embedding_matrix.shape[1]  # Use the correct embedding dimension

model = TextCNN(embedding_matrix, num_classes, num_kernels, kernel_sizes).to(device)

# Loss and Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), eps=1e-07, lr=learning_rate, weight_decay=1e-4)

# Accuracy Calculation Function
def calculate_accuracy(outputs, labels):
    predicted = (torch.sigmoid(outputs) > 0.5).float()
    correct = (predicted == labels).sum().item()
    return correct / labels.size(0)

folder_path = f'./model/imdb/cnn/{k}/'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Training Function with Early Stopping based on Validation Loss
def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs, patience, k):
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train, total_train = 0, 0
        
        # Wrap the DataLoader with tqdm for a progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        
        for batch in progress_bar:
            texts = batch['review'].to(device)
            labels = batch['target'].to(device)

            out = model(texts)
            outputs = out[0].squeeze(1)
            concat = out[1]
            loss_CE = criterion(outputs, labels)

            GBM_loss = torch.sum(torch.abs(concat))
            GBM_norm = torch.norm(torch.abs(concat).clone().detach())
            
        
            loss = (1-k)*loss_CE + k*GBM_loss/GBM_norm
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            correct_train += calculate_accuracy(outputs, labels) * labels.size(0)
            total_train += labels.size(0)

            # Update the progress bar with the current loss
            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        
        # Validate after each epoch
        val_loss, val_accuracy = evaluate_model(model, valid_loader, criterion, k)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.4f}, '
              f'Training Accuracy: {train_accuracy:.2f}%, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

        # Early stopping based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
            torch.save(best_model_state, os.path.join(folder_path, f'new_concat_model_{k}_T_{num_epochs}.pth'))
        else:
            epochs_no_improve += 1
            print(f"No improvement in epoch {epoch+1}: Validation loss did not decrease.")

        if epochs_no_improve >= patience:
            print(f'Early stopping after {epoch+1} epochs. No improvement in validation loss for {patience} epochs.')
            break

# Evaluation Function for Validation with Loss Calculation
def evaluate_model(model, test_loader, criterion, k):
    model.eval()
    correct, total = 0, 0
    running_val_loss = 0.0
    
    progress_bar = tqdm(test_loader, desc="Validating", leave=False)
   
    with torch.no_grad():
        for batch in progress_bar:
            texts = batch['review'].to(device)
            labels = batch['target'].to(device)

            out = model(texts)
            outputs = out[0].squeeze(1)
            concat = out[1]

            GBM_loss = torch.sum(torch.abs(concat))
            GBM_norm = torch.norm(torch.abs(concat).clone().detach())
            loss_CE = criterion(outputs, labels)
            loss = (1-k)*loss_CE + k*GBM_loss/GBM_norm
            running_val_loss += loss.item()

            correct += calculate_accuracy(outputs, labels) * labels.size(0)
            total += labels.size(0)

    val_loss = running_val_loss / len(test_loader)
    val_accuracy = 100 * correct / total
    return val_loss, val_accuracy

# Run Training and Validation with Early Stopping based on Validation Loss
train_model(model, train_data_loader, valid_data_loader, criterion, optimizer, num_epochs=num_epochs, patience=5, k=k)
