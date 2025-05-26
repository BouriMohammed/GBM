import torch
import numpy as np
import utils_imdb_bert as utils
import pickle
import warnings
warnings.filterwarnings("ignore")
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from OpenAttack import OpenAttack as oa
import datasets
import pandas as pd
import time
import argparse

from transformers import BertTokenizer, BertModel

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
    parser.add_argument('--task', type=str, default='imdb')
    args = parser.parse_args()
    return args

args = parse_args()
hidden_size = args.hidden_size
MAX_LEN = args.max_len
batch_size = args.batch_size
k = args.portion
ka = args.portiona
kb = args.portionb
kg = args.portiong
mode = args.mode
att = args.attack
task = args.task
num_classes = args.num_classes

path = './models/'
attack_test_dataset_path = f"./data/attack/{task}"

def loading(hidden_size, model):

    # Load the saved embedding matrix
    print('------ Robust Case ------\n')
    
    model = utils.BiLSTM(model, hidden_size=hidden_size, num_layers=1, num_classes=1)
        
    
    # Load the model state dictionary
    model_state_dict = torch.load(f'./model/{task}/bert/{ka}_{kb}/model.pth')
    model.load_state_dict(model_state_dict)
    

    return model

# Tokenize the dataset
def tokenize_data(data, tokenizer, max_len=512):
    input_ids = []
    attention_masks = []

    for sentence in data:
        encoded_dict = tokenizer.encode_plus(
            sentence, 
            add_special_tokens=True, 
            max_length=max_len, 
            padding='max_length', 
            truncation=True, 
            return_attention_mask=True, 
            return_tensors='pt'
        )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return input_ids, attention_masks

# configure access interface of the customized victim model by extending OpenAttack.Classifier.
class MyClassifier(oa.Classifier):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def get_pred(self, input_):
        return self.get_prob(input_).argmax(axis=1)

    def mymodel(self, sentences, model, tokenizer, device):
        # Tokenize and pad the input sentences

        xinf_ids, xinf_masks = tokenize_data(sentences, tokenizer)
        
        xinf_ids = xinf_ids.to(device)
        xinf_masks = xinf_masks.to(device)
    
        # Move the model to the specified device (GPU)
        model = model.to(device)
    
        # Set the model to evaluation mode
        model.eval()
    
        # Forward pass to get predictions
        with torch.no_grad():
            outputs = model(xinf_ids, xinf_masks, train_mode=False)
            
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
    correct_df = df[df['true_label'] == df['pred_label']].reset_index(drop=True)
    
    # Select relevant columns and return as Dataset
    return datasets.Dataset.from_pandas(correct_df[['review', 'true_label']])

loaded_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') 
bert_model = BertModel.from_pretrained('bert-base-uncased')

model = loading(hidden_size, bert_model)

start = time.time()
print('\n-------------------- attack_test_dataset (1000 Random samples) --------------------\n\n')
# load some examples of SST-2 for evaluation
data = get_correct_prediction(attack_test_dataset_path).map(function=dataset_mapping)
dataset = data.select(range(677,len(data)))

# choose the costomized classifier as the victim model
victim = MyClassifier(model, loaded_tokenizer)
# choose PWWS as the attacker and initialize it with default parameters

if att=='ga':
    print('ga')
    attacker = oa.attackers.GeneticAttacker()
elif att=='pwws':
    print('pwws')
    attacker = oa.attackers.PWWSAttacker()
elif att=='pso':
    print('pso')
    attacker = oa.attackers.PSOAttacker()
elif att=='bert':
    attacker = oa.attackers.BERTAttacker()
elif att=='fooler':
    attacker = oa.attackers.TextFoolerAttacker()
    
# prepare for attacking
attack_eval = oa.AttackEval(attacker, victim)
# launch attacks and print attack results 
print(attack_eval.eval(dataset, visualize=True))
end = time.time()

total_time = end - start
print("\n"+ str(total_time))