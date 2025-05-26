import sys
sys.path.insert(0, './TextAttack')

import textattack
from textattack.models.wrappers import ModelWrapper
from textattack.attack_recipes import FasterGeneticAlgorithmJia2019, PWWSRen2019, PSOZang2020, InputReductionFeng2018, TextFoolerJin2019, TextBuggerLi2018, BAEGarg2019, CLARE2020, WordOrderSwap
from textattack import Attacker
import torch
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import gzip
import utils_s4 as utils
import time
import warnings
import argparse
import pickle



warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Training Attack script")
    parser.add_argument('--hidden_size', '-d', type=int, default=64)
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--sample', type=int, default=0)
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=1,
                        help='Num blocks of S4')
    parser.add_argument('--seed', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--portion', '-k', type=float, default=0)
    parser.add_argument('--attack', type=str, default='pwws')
    
    
    # parser.add_argument('--emb', '-e', type=str, default='glove')
    parser.add_argument('--task', type=str, default='imdb')
    parser.add_argument('--portion_egv', '-kegv', type=float, default=0.0)
    parser.add_argument('--portionc', '-kc', type=float, default=0.0)
    parser.add_argument('--portionb', '-kb', type=float, default=0.0)
    parser.add_argument('--portiona', '-ka', type=float, default=0.0)
    args = parser.parse_args()
    return args


args = parse_args()
MAX_LEN = args.max_len
sample = args.sample
d_output = args.num_classes
d_model = args.hidden_size
n_layers = args.num_layers
seed = args.seed
num_classes = args.num_classes
att = args.attack
task = args.task
lr = args.learning_rate
k = args.portion
k_egv = args.portion_egv
kc = args.portionc
kb = args.portionb
ka = args.portiona

attack_test_dataset_path = "./data/attack/imdb/imdb_attack_1000.csv"


def loading(d_output, d_model, n_layers):

    # Load the saved embedding matrix
    embedding_matrix = np.load('./data/embedding/embedding_matrix.npy')

    # Load the saved tokenizer
    tokenizer_path = './data/embedding/toc.pkl'

    with open(tokenizer_path, 'rb') as tokenizer_file:
        loaded_tokenizer = pickle.load(tokenizer_file)

    # Define the model architecture (you may need to adapt this based on your implementation)
    model = utils.S4Model(
        embedding_matrix=embedding_matrix,
        d_input=embedding_matrix.shape[1],
        d_output=d_output,
        d_model=d_model,
        n_layers=n_layers
    )

   
    model_path = f'./model/imdb/S4/{ka}_{kb}/model.pth.gz'
     
    with gzip.open(model_path, 'rb') as f:
        model_state_dict = torch.load(f)
    model.load_state_dict(model_state_dict)

    return model, loaded_tokenizer


class CustomTensorFlowModelWrapper(ModelWrapper):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device("cuda")
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _preprocess_text(self, text_input_list):
        # Tokenize and pad the input sentences
        x_input = self.tokenizer.texts_to_sequences(text_input_list)
        x_input = torch.tensor(tf.keras.preprocessing.sequence.pad_sequences(
            x_input, maxlen=300, padding='post', value=0), dtype=torch.long)
        return x_input

    def _get_probabilities(self, x_input):
        # Move the input data to the specified device (GPU)
        x_input = x_input.to(self.device)

        # Move the model to the specified device (GPU)
        model = self.model.to(self.device)

        # Set the model to evaluation mode
        model.eval()

        # Forward pass to get predictions
        with torch.no_grad():
            outputs = model(x_input)

        result = torch.sigmoid(outputs).cpu().detach().numpy()

        return result

    def __call__(self, text_input_list):
        x_input = self._preprocess_text(text_input_list)
        probabilities = self._get_probabilities(x_input)
        probabilities_tensor = torch.tensor(probabilities, dtype=torch.float32)

        # Concatenate probabilities with their complements
        complements = 1 - probabilities_tensor
        final_result = torch.cat((complements, probabilities_tensor), dim=1)

        return final_result.numpy()


def dataframe_to_list(df):
    return list(zip(df['review'], df['sentiment']))


model, loaded_tokenizer = loading(d_output, d_model, n_layers)
model_wrapper = CustomTensorFlowModelWrapper(model, loaded_tokenizer)

dataset = pd.read_csv(attack_test_dataset_path)
data = dataframe_to_list(dataset)
dataset = textattack.datasets.Dataset(data)

if sample != 0:
    print(f'Sample = {sample}')
    attack_args = textattack.AttackArgs(num_examples=sample)
else:
    print(f'Sample = ALL')
    attack_args = textattack.AttackArgs(num_examples=len(data))

if att == 'swap':
    print('swap')
    attack = WordOrderSwap.build(model_wrapper)
elif att == 'pwws':
    print('pwws')
    attack = PWWSRen2019.build(model_wrapper)
elif att == 'pso':
    print('pso')
    attack = PSOZang2020.build(model_wrapper)
elif att == 'remove':
    print('remove')
    attack = InputReductionFeng2018.build(model_wrapper)
elif att == 'fooler':
    print('fooler')
    attack = TextFoolerJin2019.build(model_wrapper)
elif att == 'bugger':
    print('bugger')
    attack = TextBuggerLi2018.build(model_wrapper)
elif att == 'clare':
    print('clare')
    attack = CLARE2020.build(model_wrapper)
elif att == 'ga':
    print('ga')
    attack = FasterGeneticAlgorithmJia2019.build(model_wrapper)

start = time.time()
print('\n-------------------- attack_test_dataset (1000 Random samples) --------------------\n\n')

attacker = Attacker(attack, dataset, attack_args)
aa = attacker.attack_dataset()
