import torch
import torch.nn as nn

# Create a dataset class


class IMDBBERTDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_masks, targets):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, item):
        return {
            'input_ids': self.input_ids[item],
            'attention_mask': self.attention_masks[item],
            'targets': self.targets[item]
        }


class IMDBBERTDataset2(torch.utils.data.Dataset):
    # def __init__(self,  input_ids, attention_masks, embeddings, targets):
    def __init__(self, embeddings, targets):
        # self.input_ids = input_ids
        # self.attention_masks = attention_masks
        self.embeddings = embeddings
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, item):
        return {
            'embedding': self.embeddings[item],
            # 'input_ids': self.input_ids[item],
            # 'attention_mask': self.attention_masks[item],
            'targets': self.targets[item]
        }


class BiLSTM(nn.Module):
    def __init__(self, bert_model, hidden_size, num_layers, num_classes, dropout=0.2):
        super(BiLSTM, self).__init__()
        self.bert = bert_model

        self.lstm = nn.LSTM(input_size=self.bert.config.hidden_size, hidden_size=hidden_size,
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

    def forward(self, input_ids, attention_mask, train_mode=True):
        with torch.no_grad():
            bert_output = self.bert(
                input_ids=input_ids, attention_mask=attention_mask)
            embedded = bert_output.last_hidden_state

        # print(embedded.shape)
        lstm_output, state = self.lstm(embedded)

        # Concatenate the hidden states of the last timestep from both directions
        hidden_state = torch.cat(
            (lstm_output[:, -1, :self.lstm.hidden_size], lstm_output[:, 0, self.lstm.hidden_size:]), dim=-1)

        out = self.relu(hidden_state)
        out = self.fc(self.dropout(out))

        return out
