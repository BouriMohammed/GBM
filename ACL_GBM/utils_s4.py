import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import os
from s4.models.s4.s4 import S4Block as S4


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
            if word in self.embedding_dict and word != 'UNK' and word != '<UNK>':
                embedding_matrix[index] = self.embedding_dict[word]
            elif word not in self.embedding_dict and word in emb_lower and word != 'unk' and word != '<unk>':
                embedding_matrix[index] = emb_lower[word]

        return embedding_matrix


class S4Model(nn.Module):

    def __init__(
        self,
        embedding_matrix,
        d_input,
        d_output,
        d_model,
        n_layers,
        dropout=0.2,
        prenorm=False,

    ):
        super().__init__()

        self.prenorm = prenorm
        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix), padding_idx=0)

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4(d_model, dropout=dropout, transposed=True, lr=0.001)
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(nn.Dropout(dropout))

        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """
        x = self.embedding(x)  # (B, L, d_input) -> (B, L, d_model)
        x = self.encoder(x)
        # print('Embedding:\t', x.shape)
        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        # print('Embedding_transpose:\t', x.shape)

        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # print(layer)
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)
            # print('Im HERE')
            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)

        # Pooling: average pooling over the sequence length
        x = x.mean(dim=1)

        # Decode the outputs
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)

        return x



class Engine_gbm_discret_derivative:
    # def __init__(self, model, optimizer, scheduler, device):
    def __init__(self, model, optimizer, scheduler, k_a, k_b, device):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.ka = k_a
        self.kb = k_b
        self.scheduler = scheduler

    def extract_discret_matrices(self, A_real, A_imag, P, inv_dt, B):
        P_complex = torch.complex(
            P[:, :, :, 0].squeeze(0), P[:, :, :, 1].squeeze(0))
        P_conj = P_complex.conj()
        A_diag_blocks = torch.diag_embed(
            A_real + 1j * A_imag)  # [d_state, rank, rank]
        low_rank_correction = torch.einsum(
            'bi,bj->bij', P_complex, P_conj)  # Shape: [512, 32, 32]
        A = A_diag_blocks - low_rank_correction

        dt = 1/inv_dt

        B_complex = torch.complex(
            B[:, :, :, 0].squeeze(0), B[:, :, :, 1].squeeze(0))

        identity_matrix = torch.eye(32).expand(256, 32, 32).to('cuda')
        first_term = 2 * identity_matrix * inv_dt.view(256, 1, 1)

        A0 = first_term.detach().clone() + A.detach().clone()

        second_term = torch.linalg.inv(
            identity_matrix - 0.5*dt.view(256, 1, 1)*A)

        A1 = 0.5*dt.view(256, 1, 1)*second_term.detach().clone()

        B_bar = 2*torch.matmul(A1.detach().clone(),
                               B_complex.detach().clone().unsqueeze(-1)).squeeze(-1)

        A_bar = torch.matmul(A1.detach().clone(), A0.detach().clone())

        return A_bar, B_bar

    def extract_gbm(self, A_bar, B_bar, C, D):
        C_bar = torch.complex(C[:, :, :, 0].squeeze(0), C[:, :, :, 1].squeeze(0))
        x = torch.bmm(C_bar.unsqueeze(1), A_bar)
        u = torch.bmm(C_bar.unsqueeze(1), B_bar.unsqueeze(-1)).squeeze(-1).squeeze(-1) + D[0]
        return x,u
    
    # Training function

    def train(self, epoch, data_loader, num_epochs):
        self.model.train()
        train_loss = 0
        A_loss = 0
        B_loss = 0
        
        correct = 0
        total = 0
        for batch in tqdm(data_loader, desc=f'Training Progress {epoch + 1}/{num_epochs}', leave=False):
            inputs = batch['review'].to(self.device)
            # Add one dimension for BCEWithLogitsLoss
            targets = batch['target'].unsqueeze(1).to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            A_real = self.model.state_dict(
            )["s4_layers.0.layer.kernel.A_real"].detach().clone()
            A_imag = self.model.state_dict(
            )["s4_layers.0.layer.kernel.A_imag"].detach().clone()
            P = self.model.state_dict(
            )["s4_layers.0.layer.kernel.P"].detach().clone()
            B = self.model.state_dict(
            )["s4_layers.0.layer.kernel.B"].detach().clone()
            inv_dt = self.model.state_dict(
            )['s4_layers.0.layer.kernel.inv_dt'].detach().clone()
            C = self.model.state_dict(
            )["s4_layers.0.layer.kernel.C"].detach().clone()
            D = self.model.state_dict(
            )["s4_layers.0.layer.D"].detach().clone()

            A_bar, B_bar = self.extract_discret_matrices(
                A_real, A_imag, P, inv_dt, B)

            x, u = self.extract_gbm(A_bar, B_bar, C, D)
            
            modulus_sum_A = torch.sum(torch.abs(x))
            modulus_sum_B = torch.sum(torch.abs(u))
            

            loss = nn.BCEWithLogitsLoss()(outputs, targets)
            combined_loss = (1-self.ka-self.kb)*loss + self.ka * \
                (modulus_sum_A) + self.kb * (modulus_sum_B)

            combined_loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            train_loss += combined_loss.item()
            A_loss += modulus_sum_A
            B_loss += modulus_sum_B
            
            # Convert logits to probabilities and then to binary predictions (0 or 1)
            predicted = (torch.sigmoid(outputs) > 0.5).float()

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # After epoch, compute and display average loss and accuracy
        avg_loss = train_loss / len(data_loader)
        avg_A_loss = A_loss / len(data_loader)
        avg_B_loss = B_loss / len(data_loader)
        
        accuracy = 100. * correct / total
        print(
            f'Training for Epoch {epoch + 1}/{num_epochs} completed | Avg Loss: {avg_loss:.3f} | Avg A_loss: {avg_A_loss:.3f} | Avg B_loss: {avg_B_loss:.3f} | Accuracy: {accuracy:.3f}%')

    # Evaluation function

    def eval(self, epoch, data_loader, num_epochs, checkpoint=False):
        self.model.eval()
        eval_loss = 0
        A_loss = 0
        B_loss = 0
        
        correct = 0
        total = 0
        real_eg_loss = 0

        with torch.no_grad():
            A_real = self.model.state_dict(
            )["s4_layers.0.layer.kernel.A_real"].detach().clone()
            A_imag = self.model.state_dict(
            )["s4_layers.0.layer.kernel.A_imag"].detach().clone()
            P = self.model.state_dict(
            )["s4_layers.0.layer.kernel.P"].detach().clone()
            B = self.model.state_dict(
            )["s4_layers.0.layer.kernel.B"].detach().clone()
            inv_dt = self.model.state_dict(
            )['s4_layers.0.layer.kernel.inv_dt'].detach().clone()
            C = self.model.state_dict(
            )["s4_layers.0.layer.kernel.C"].detach().clone()
            D = self.model.state_dict(
            )["s4_layers.0.layer.D"].detach().clone()

            A_bar, B_bar = self.extract_discret_matrices(
                A_real, A_imag, P, inv_dt, B)
            
            x, u = self.extract_gbm(A_bar, B_bar, C, D)
            
            modulus_sum_A = torch.sum(torch.abs(x))
            modulus_sum_B = torch.sum(torch.abs(u))
            

            for batch in tqdm(data_loader, desc=f'Evaluating Progress {epoch + 1}/{num_epochs}', leave=False):
                inputs = batch['review'].to(self.device)
                targets = batch['target'].unsqueeze(1).to(self.device)

                outputs = self.model(inputs)
                loss = nn.BCEWithLogitsLoss()(outputs, targets)
                combined_loss = (1-self.ka-self.kb)*loss + self.ka * \
                    (modulus_sum_A) + self.kb * (modulus_sum_B)

                eval_loss += combined_loss.item()
                A_loss += modulus_sum_A
                B_loss += modulus_sum_B
               

                # Convert logits to binary predictions
                predicted = (torch.sigmoid(outputs) > 0.5).float()

                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            avg_loss = eval_loss / len(data_loader)
            avg_A_loss = A_loss / len(data_loader)
            avg_B_loss = B_loss / len(data_loader)
            
            accuracy = 100. * correct / total
            print(
                f'Evaluation for Epoch {epoch + 1}/{num_epochs} completed | Avg Loss: {avg_loss:.3f} | Avg A_loss: {avg_A_loss:.3f} | Avg B_loss: {avg_B_loss:.3f} | Accuracy: {accuracy:.3f}%')

            return avg_loss