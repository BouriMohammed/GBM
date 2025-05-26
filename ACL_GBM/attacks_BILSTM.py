import time
import pandas as pd
import argparse
import datasets
from OpenAttack import OpenAttack as oa
import ssl
import torch
import tensorflow as tf
import keras
# import tensorflow.keras.preprocessing.text
import numpy as np
import utils_imdb as utils
import pickle
import warnings
warnings.filterwarnings("ignore")
ssl._create_default_https_context = ssl._create_unverified_context
tf.config.optimizer.set_jit(False)


def parse_args():
    parser = argparse.ArgumentParser(description="Training Attack script")
    parser.add_argument('--hidden_size', '-d', type=int, default=64)
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--attack', type=str, default='pwws')
    parser.add_argument('--batch_size', '-b', type=int, default=64)
    parser.add_argument('--portion', '-k', type=float, default=0)
    parser.add_argument('--portiona', '-ka', type=float, default=0.0)
    parser.add_argument('--portionb', '-kb', type=float, default=0.0)
    parser.add_argument('--portiong', '-kg', type=float, default=0.0)
    parser.add_argument('--mode', '-m', type=str, default='bilstm')
    parser.add_argument('--emb', '-e', type=str, default='glove')
    parser.add_argument('--task', type=str, default='imdb')
    args = parser.parse_args()
    return args


path = './models/'

tf.config.experimental_compile = False
args = parse_args()
hidden_size = args.hidden_size
MAX_LEN = args.max_len
batch_size = args.batch_size
k = args.portion
ka = args.portiona
kb = args.portionb
kg = args.portiong
mode = args.mode
emb = args.emb
att = args.attack
task = args.task
num_classes = args.num_classes

attack_test_dataset_path = f"./data/attack/{task}"


def loading(hidden_size, batch_size, k):

    # Load the saved embedding matrix
    print('------ Robust Case ------\n')
    embedding_matrix = np.load(f'./model/{task}/{emb}/embedding_matrix.npy')

    # Load the saved tokenizer
    tokenizer_path = f'./model/{task}/{emb}/toc.pkl'

    with open(tokenizer_path, 'rb') as tokenizer_file:
        loaded_tokenizer = pickle.load(tokenizer_file)

    # Define the model architecture (you may need to adapt this based on your implementation)
    model = utils.BiLSTM(embedding_matrix, hidden_size=hidden_size,
                         num_layers=1, num_classes=num_classes)

    # Load the model state dictionary
    model_state_dict = torch.load(f'./model/{task}/{emb}/{ka}_{kb}/model.pth')
    model.load_state_dict(model_state_dict)

    return model, loaded_tokenizer


# configure access interface of the customized victim model by extending OpenAttack.Classifier.
class MyClassifier(oa.Classifier):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def get_pred(self, input_):
        return self.get_prob(input_).argmax(axis=1)

    def mymodel(self, sentences, model, tokenizer, device):
        # Tokenize and pad the input sentences
        x_input = tokenizer.texts_to_sequences(sentences)
        # x_input = tf.keras.preprocessing.sequence.pad_sequences(
        #     x_input, maxlen=MAX_LEN, padding='post', value = 0)
        x_input = torch.tensor(tf.keras.preprocessing.sequence.pad_sequences(
            x_input, maxlen=MAX_LEN, padding='post', value=0), dtype=torch.long)

        # Move the input data to the specified device (GPU)
        x_input = x_input.to(device)

        # Move the model to the specified device (GPU)
        model = model.to(device)

        # Set the model to evaluation mode
        model.eval()

        # Forward pass to get predictions
        with torch.no_grad():
            outputs = model(x_input)

        return torch.sigmoid(outputs).cpu().detach().numpy()
        # return torch.argmax(outputs, dim=1).cpu().detach().numpy()

    # access to the classification probability scores with respect to input sentences
    def get_prob(self, input_):
        ret = []
        device = torch.device("cuda")
        for sent in input_:
            res = self.mymodel([sent], self.model, self.tokenizer, device)
            prob = float(res[0][0])
            ret.append(np.array([1 - prob, prob]))
        return np.array(ret)


def dataset_mapping(x):
    return {
        "x": x["review"],
        "y": 1 if x["true_label"] == 1 else 0,
    }


def get_correct_prediction(path):
    # Load dataset and initialize classifier
    dataset = datasets.load_from_disk(path)
    cl = MyClassifier(model, loaded_tokenizer)

    # Get predictions
    reviews = cl.get_prob(dataset['text'])
    predicted_labels = (reviews[:, 1] >= 0.5).astype(int)

    # Calculate accuracy
    accuracy = (predicted_labels == dataset['label']).mean()
    print('Acc:', accuracy)

    # Create DataFrame with review, true_label, and pred_label
    df = pd.DataFrame({
        'review': dataset['text'],
        'true_label': dataset['label'],
        'pred_label': predicted_labels
    })

    # Filter DataFrame to include only correct predictions
    correct_df = df[df['true_label'] ==
                    df['pred_label']].reset_index(drop=True)

    # Select relevant columns and return as Dataset
    return datasets.Dataset.from_pandas(correct_df[['review', 'true_label']])


model, loaded_tokenizer = loading(hidden_size, batch_size, k)

start = time.time()
print('\n-------------------- attack_test_dataset (1000 Random samples) --------------------\n\n')
dataset = get_correct_prediction(
    attack_test_dataset_path).map(function=dataset_mapping)

# dataset = datasets.load_from_disk(attack_test_dataset_path).map(function=dataset_mapping)
# dataset = datasets.load_from_disk(attack_test_dataset_path).remove_columns('__index_level_0__').map(function=dataset_mapping).select(list(range(10)))
# choose the costomized classifier as the victim model
victim = MyClassifier(model, loaded_tokenizer)
# choose PWWS as the attacker and initialize it with default parameters

if att == 'ga':
    print('ga')
    attacker = oa.attackers.GeneticAttacker()
elif att == 'pwws':
    print('pwws')
    attacker = oa.attackers.PWWSAttacker()
elif att == 'pso':
    print('pso')
    attacker = oa.attackers.PSOAttacker()
elif att == 'bae':
    attacker = oa.attackers.BAEAttacker()
elif att == 'fooler':
    attacker = oa.attackers.TextFoolerAttacker()

# prepare for attacking
attack_eval = oa.AttackEval(attacker, victim)
# launch attacks and print attack results
print(attack_eval.eval(dataset, visualize=True))
end = time.time()

total_time = end - start
print("\n" + str(total_time))
