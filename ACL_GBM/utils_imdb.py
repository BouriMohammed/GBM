import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


class IMDBDataset:
    def __init__(self, reviews, targets):
        """
        Argument:
        reviews: a numpy array
        targets: a vector array

        Return xtrain and ylabel in torch tensor datatype, stored in dictionary format
        """
        self.reviews = reviews
        self.target = targets

    def __len__(self):
        # return length of dataset
        return len(self.reviews)

    def __getitem__(self, index):
        # given an idex (item), return review and target of that index in torch tensor
        review = torch.tensor(self.reviews[index, :], dtype=torch.long)
        target = torch.tensor(self.target[index], dtype=torch.float)

        return {'review': review,
                'target': target}


class EmbeddingLoader:
    def __init__(self, embedding_path, d_model=300):
        self.embedding_path = embedding_path
        self.d_model = d_model
        self.embedding_dict = self.load_embedding()

    def load_embedding(self):
        embedding_dict = dict()
        with open(self.embedding_path, encoding="utf-8") as f:
            num_lines = sum(1 for _ in f)
            f.seek(0)  # Reset file pointer to the beginning
            for line in tqdm(f, total=num_lines, desc='Loading embeddings', unit=' lines'):
                values = line.strip().split(" ")
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embedding_dict[word] = coefs
        print('\t Loaded %s word vectors.' % len(embedding_dict))
        return embedding_dict

    def create_embedding_matrix(self, word_index):
        """
        This function creates the embedding matrix saved in a numpy array.
        :param word_index: a dictionary with word: index_value
        :return: a numpy array with embedding vectors for all known words
        """
        embedding_matrix = np.zeros((len(word_index)+1, self.d_model))
        emb_lower = {key.lower(): value for key,
                     value in self.embedding_dict.items()}

        for word, index in word_index.items():
            if word in self.embedding_dict and word != 'UNK':
                embedding_matrix[index] = self.embedding_dict[word]
            elif word not in self.embedding_dict and word in emb_lower and word != 'unk':
                embedding_matrix[index] = emb_lower[word]

        return embedding_matrix


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

        # Concatenate the hidden states of the last timestep from both directions
        hidden_state = torch.cat(
            (lstm_output[:, -1, :self.lstm.hidden_size], lstm_output[:, 0, self.lstm.hidden_size:]), dim=-1)

        out = self.relu(hidden_state)
        out = self.fc(self.dropout(out))

        return out

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
        x = self.dropout(x)
        return self.fc(x)