import pandas as pd
import tensorflow as tf
import utils_imdb as utils
import pickle
import numpy as np

data_train = pd.read_csv('./data/imdb/train.csv')
data_test = pd.read_csv('./data/imdb/test.csv')

train = data_train.copy()
test = data_test.copy()
data = pd.concat([train,test], ignore_index=True)

tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="<UNK>")
tokenizer.fit_on_texts(data.text.values.tolist())

# Adding padding token to the word index
tokenizer.word_index["<PAD>"] = 0
tokenizer.index_word[0] = "<PAD>"

xtrain = tokenizer.texts_to_sequences(train.text.values)
xtest = tokenizer.texts_to_sequences(test.text.values)

## With tensorflow
xtrain = tf.keras.preprocessing.sequence.pad_sequences(
    xtrain, maxlen=512, padding='post', value = 0)
xtest = tf.keras.preprocessing.sequence.pad_sequences(
    xtest, maxlen=512, padding='post', value = 0)

y_train = train.label.copy()
y_test = test.label.copy()

embed_path = './data/embedding/glove.840B.300d.txt'
embedding_loader = utils.EmbeddingLoader(
    embed_path, d_model=300)
embedding_matrix = embedding_loader.create_embedding_matrix(
    tokenizer.word_index)

x_interval = np.array([np.min(embedding_matrix, axis=0), np.max(embedding_matrix,axis=0)]).T
np.save('./inputs/x_interval.npy', x_interval)

with open('./data/embedding/toc.pkl', 'wb') as tokenizer_file:
    pickle.dump(tokenizer, tokenizer_file)

np.save('./data/embedding/embedding_matrix.npy',embedding_matrix)

np.save('./data/imdb/xtrain.npy', xtrain)
np.save('./data/imdb/xtest.npy', xtest)

np.save('./data/imdb/y_train.npy', y_train)
np.save('./data/imdb/y_test.npy', y_test)