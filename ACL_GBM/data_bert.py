import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
from transformers import BertTokenizer, BertModel


data_train = pd.read_csv('./data/imdb/IMDB_train_data.csv')
data_test = pd.read_csv('./data/imdb/IMDB_test_data.csv')

train = data_train.copy()
test = data_test.copy()
data = pd.concat([train, test], ignore_index=True)

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


def get_bert_embeddings(sentences, model, tokenizer, device):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for sentence in tqdm(sentences):
            inputs = tokenizer(sentence, return_tensors='pt', padding=True,
                               truncation=True, max_length=512).to(device)
            outputs = model(**inputs)
            sentence_embedding = outputs.last_hidden_state.cpu().numpy()
            embeddings.append(sentence_embedding)
    return embeddings


def extract_x_interval(vectors):

    x_int_min = np.min(vectors[0][0], axis=0)
    x_int_max = np.max(vectors[0][0], axis=0)

    for i in tqdm(range(1, len(vectors))):
        min = np.min(vectors[i][0], axis=0)
        max = np.max(vectors[i][0], axis=0)

        x_int_min = np.minimum(x_int_min, min)
        x_int_max = np.maximum(x_int_max, max)

    return np.stack([x_int_min, x_int_max], axis=1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

bert_embeddings_data = get_bert_embeddings(
    data.review.values, bert_model, tokenizer, device)

x_interval = extract_x_interval(bert_embeddings_data)

np.save('./inputs/x_interval_bert.npy', x_interval)


xtrain_ids, xtrain_masks = tokenize_data(train.review.values, tokenizer)
xtest_ids, xtest_masks = tokenize_data(test.review.values, tokenizer)

np.save('./data/imdb/x_train_bert_id.npy', xtrain_ids.detach().numpy().copy())
np.save('./data/imdb/x_train_bert_masks.npy',
        xtrain_masks.detach().numpy().copy())

np.save('./data/imdb/x_test_bert_id.npy', xtest_ids.detach().numpy().copy())
np.save('./data/imdb/x_test_bert_masks.npy',
        xtest_masks.detach().numpy().copy())
