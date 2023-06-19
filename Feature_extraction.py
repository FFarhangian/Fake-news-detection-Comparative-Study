from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from zeugma.embeddings import EmbeddingTransformer
import gensim
from numpy import savez_compressed
import scipy.sparse.linalg
from numpy import asarray
from scipy import sparse
from numpy import load
import joblib
import tensorflow_hub as hub
import tensorflow.compat.v1 as tf
import torch
import numpy as np
from transformers import BertModel, BertTokenizer
from transformers import DistilBertModel, DistilBertTokenizer
from transformers import AlbertModel, AlbertTokenizer
from transformers import BartModel, BartTokenizer
from transformers import RobertaModel, RobertaTokenizer
from transformers import ElectraModel, ElectraTokenizer
from transformers import XLNetModel, XLNetTokenizer
from allennlp.commands.elmo import ElmoEmbedder







# TF
TF = CountVectorizer(analyzer='word', lowercase=True, stop_words='english')
############df1
TF_df1_train = TF.fit_transform(df1_train)
TF_df1_test = TF.transform(df1_test)
############df2
TF_df2_train = TF.fit_transform(df2_train)
TF_df2_test = TF.transform(df2_test)
############df3
TF_df3_train = TF.fit_transform(df3_train)
TF_df3_test = TF.transform(df3_test)
############df4
TF_df4_train = TF.fit_transform(df4_train)
TF_df4_test = TF.transform(df4_test)

# Save matrix TF
sparse.save_npz("TF_df1_train.npz", TF_df1_train)
sparse.save_npz("TF_df2_train.npz", TF_df2_train)
sparse.save_npz("TF_df3_train.npz", TF_df3_train)
sparse.save_npz("TF_df4_train.npz", TF_df4_train)

sparse.save_npz("TF_df1_test.npz", TF_df1_test)
sparse.save_npz("TF_df2_test.npz", TF_df2_test)
sparse.save_npz("TF_df3_test.npz", TF_df3_test)
sparse.save_npz("TF_df4_test.npz", TF_df4_test)

# TF-IDF
TFIDF = TfidfVectorizer(analyzer='word', lowercase=True, use_idf=True, stop_words='english')
############df1
TFIDF_df1_train = TFIDF.fit_transform(df1_train)
TFIDF_df1_test = TFIDF.transform(df1_test)
############df2
TFIDF_df2_train = TFIDF.fit_transform(df2_train)
TFIDF_df2_test = TFIDF.transform(df2_test)
############df3
TFIDF_df3_train = TFIDF.fit_transform(df3_train)
TFIDF_df3_test = TFIDF.transform(df3_test)
############df4
TFIDF_df4_train = TFIDF.fit_transform(df4_train)
TFIDF_df4_test = TFIDF.transform(df4_test)

# Save matrix TFIDF
sparse.save_npz("TFIDF_df1_train.npz", TFIDF_df1_train)
sparse.save_npz("TFIDF_df2_train.npz", TFIDF_df2_train)
sparse.save_npz("TFIDF_df3_train.npz", TFIDF_df3_train)
sparse.save_npz("TFIDF_df4_train.npz", TFIDF_df4_train)

sparse.save_npz("TFIDF_df1_test.npz", TFIDF_df1_test)
sparse.save_npz("TFIDF_df2_test.npz", TFIDF_df2_test)
sparse.save_npz("TFIDF_df3_test.npz", TFIDF_df3_test)
sparse.save_npz("TFIDF_df4_test.npz", TFIDF_df4_test)

# Word2Vec
W2V = EmbeddingTransformer('word2vec')
############df1
W2V_df1_train = W2V.fit_transform(df1_train)
W2V_df1_test = W2V.transform(df1_test)
############df2
W2V_df2_train = W2V.fit_transform(df2_train)
W2V_df2_test = W2V.transform(df2_test)
############df3
W2V_df3_train = W2V.fit_transform(df3_train)
W2V_df3_test = W2V.transform(df3_test)
############df4
W2V_df4_train = W2V.fit_transform(df4_train)
W2V_df4_test = W2V.transform(df4_test)

# Save matrix W2V

savez_compressed('W2V_df1_train.npz', W2V_df1_train)
savez_compressed('W2V_df2_train.npz', W2V_df2_train)
savez_compressed('W2V_df3_train.npz', W2V_df3_train)
savez_compressed('W2V_df4_train.npz', W2V_df4_train)

savez_compressed('W2V_df1_test.npz', W2V_df1_test)
savez_compressed('W2V_df2_test.npz', W2V_df2_test)
savez_compressed('W2V_df3_test.npz', W2V_df3_test)
savez_compressed('W2V_df4_test.npz', W2V_df4_test)

# GloVe
Glove = EmbeddingTransformer('glove')
############df1
Glove_df1_train = Glove.fit_transform(df1_train)
Glove_df1_test = Glove.transform(df1_test)
############df2
Glove_df2_train = Glove.fit_transform(df2_train)
Glove_df2_test = Glove.transform(df2_test)
############df3
Glove_df3_train = Glove.fit_transform(df3_train)
Glove_df3_test = Glove.transform(df3_test)
############df4
Glove_df4_train = Glove.fit_transform(df4_train)
Glove_df4_test = Glove.transform(df4_test)

# Save matrix Glove
savez_compressed('Glove_df1_train.npz', Glove_df1_train)
savez_compressed('Glove_df2_train.npz', Glove_df2_train)
savez_compressed('Glove_df3_train.npz', Glove_df3_train)
savez_compressed('Glove_df4_train.npz', Glove_df4_train)

savez_compressed('Glove_df1_test.npz', Glove_df1_test)
savez_compressed('Glove_df2_test.npz', Glove_df2_test)
savez_compressed('Glove_df3_test.npz', Glove_df3_test)
savez_compressed('Glove_df4_test.npz', Glove_df4_test)

#Fasttext
Fasttext = EmbeddingTransformer('fasttext')
############df1
Fasttext_df1_train = Fasttext.fit_transform(df1_train)
Fasttext_df1_test = Fasttext.transform(df1_test)
############df2
Fasttext_df2_train = Fasttext.fit_transform(df2_train)
Fasttext_df2_test = Fasttext.transform(df2_test)
############df3
Fasttext_df3_train = Fasttext.fit_transform(df3_train)
Fasttext_df3_test = Fasttext.transform(df3_test)
############df4
Fasttext_df4_train = Fasttext.fit_transform(df4_train)
Fasttext_df4_test = Fasttext.transform(df4_test)

# Save matrix Fasttext
savez_compressed('Fasttext_df1_train.npz', Fasttext_df1_train)
savez_compressed('Fasttext_df2_train.npz', Fasttext_df2_train)
savez_compressed('Fasttext_df3_train.npz', Fasttext_df3_train)
savez_compressed('Fasttext_df4_train.npz', Fasttext_df4_train)

savez_compressed('Fasttext_df1_test.npz', Fasttext_df1_test)
savez_compressed('Fasttext_df2_test.npz', Fasttext_df2_test)
savez_compressed('Fasttext_df3_test.npz', Fasttext_df3_test)
savez_compressed('Fasttext_df4_test.npz', Fasttext_df4_test)

# ELMO
tf.disable_eager_execution()
  
# Load pre trained ELMo model
elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)
##############################################################

def elmo_vectors(z):
  embeddings = elmo(z, signature="default", as_dict=True)["elmo"]
  sess = tf.Session()
  sess.run(tf.initialize_all_variables())
  output = sess.run(tf.reduce_mean(embeddings,1))
  return output
  
########################################################
  
# ELMO-df1-Train
batch_num = 1000
ELMO_df1_train = np.zeros((df1_train.shape[0], 1024))

for i in range(0, len(df1_train.tolist()), batch_num):
    print(i)
    ELMO_df1_train[i:i + batch_num] = elmo_vectors(df1_train.tolist()[i:i + batch_num])

savez_compressed('ELMO_df1_train.npz', ELMO_df1_train)
print(ELMO_df1_train.shape)

# ELMO-df1-Test
batch_num = 1000
ELMO_df1_test = np.zeros((df1_test.shape[0], 1024))

for i in range(0, len(df1_test.tolist()), batch_num):
    print(i)
    ELMO_df1_test[i:i + batch_num] = elmo_vectors(df1_test.tolist()[i:i + batch_num])

savez_compressed('ELMO_df1_test.npz', ELMO_df1_test)
print(ELMO_df1_test.shape)
  
  
###########################################################
  
# ELMO-df2-Train
batch_num = 25
ELMO_df2_train = np.zeros((df2_train.shape[0], 1024))

for i in range(0, len(df2_train.tolist()), batch_num):
    print(i)
    ELMO_df2_train[i:i + batch_num] = elmo_vectors(df2_train.tolist()[i:i + batch_num])

savez_compressed('ELMO_df2_train.npz', ELMO_df2_train)
print(ELMO_df2_train.shape)

# ELMO-df2-Test
batch_num = 10
ELMO_df2_test = np.zeros((df2_test.shape[0], 1024))

for i in range(0, len(df2_test.tolist()), batch_num):
    print(i)
    ELMO_df2_test[i:i + batch_num] = elmo_vectors(df2_test.tolist()[i:i + batch_num])

savez_compressed('ELMO_df2_test.npz', ELMO_df2_test)
print(ELMO_df2_test.shape)

##############################################################
  
# ELMO-df3-Train
batch_num = 50
ELMO_df3_train = np.zeros((df3_train.shape[0], 1024))

for i in range(0, len(df3_train.tolist()), batch_num):
    print(i)
    ELMO_df3_train[i:i + batch_num] = elmo_vectors(df3_train.tolist()[i:i + batch_num])

savez_compressed('ELMO_df3_train.npz', ELMO_df3_train)
print(ELMO_df3_train.shape)

# ELMO-df2-Test
batch_num = 50
ELMO_df3_test = np.zeros((df3_test.shape[0], 1024))

for i in range(0, len(df3_test.tolist()), batch_num):
    print(i)
    ELMO_df3_test[i:i + batch_num] = elmo_vectors(df3_test.tolist()[i:i + batch_num])

savez_compressed('ELMO_df3_test.npz', ELMO_df3_test)
print(ELMO_df3_test.shape)

##############################################################

# ELMO-df4-Train
batch_num = 25
ELMO_df4_train = np.zeros((df4_train.shape[0], 1024))

for i in range(0, len(df4_train.tolist()), batch_num):
    print(i)
    ELMO_df4_train[i:i + batch_num] = elmo_vectors(df4_train.tolist()[i:i + batch_num])

savez_compressed('ELMO_df4_train.npz', ELMO_df4_train)
print(ELMO_df4_train.shape)

# ELMO-df4-Test
batch_num = 10
ELMO_df4_test = np.zeros((df4_test.shape[0], 1024))

for i in range(0, len(df4_test.tolist()), batch_num):
    print(i)
    ELMO_df4_test[i:i + batch_num] = elmo_vectors(df4_test.tolist()[i:i + batch_num])

savez_compressed('ELMO_df4_test.npz', ELMO_df4_test)
print(ELMO_df4_test.shape)

# ELMO another approach
# ELMO
from allennlp.commands.elmo import ElmoEmbedder
import numpy as np

# Load the ELMo model
elmo = ElmoEmbedder()

def extract_features(text):
    embeddings = elmo.embed_sentence(text)
    features = embeddings.mean(axis=0)  # Take the mean of the word embeddings
    return features

text_data_train = df1_train.tolist()
feature_matrix_train = []
for text in text_data_train:
    features = extract_features(text)
    feature_matrix_train.append(features)

feature_matrix_train = torch.cat(feature_matrix_train, dim=0)

np.savez_compressed('ELMO_df1_train.npz', feature_matrix_train.numpy())

text_data_test = df1_test.tolist()
feature_matrix_test = []
for text in text_data_test:
    features = extract_features(text)
    feature_matrix_test.append(features)

feature_matrix_test = torch.cat(feature_matrix_test, dim=0)

np.savez_compressed('ELMO_df1_test.npz', feature_matrix_test.numpy())

text_data_train = df2_train.tolist()
feature_matrix_train = []
for text in text_data_train:
    features = extract_features(text)
    feature_matrix_train.append(features)

feature_matrix_train = torch.cat(feature_matrix_train, dim=0)

np.savez_compressed('ELMO_df2_train.npz', feature_matrix_train.numpy())

text_data_test = df2_test.tolist()
feature_matrix_test = []
for text in text_data_test:
    features = extract_features(text)
    feature_matrix_test.append(features)

feature_matrix_test = torch.cat(feature_matrix_test, dim=0)

np.savez_compressed('ELMO_df2_test.npz', feature_matrix_test.numpy())

text_data_train = df3_train.tolist()
feature_matrix_train = []
for text in text_data_train:
    features = extract_features(text)
    feature_matrix_train.append(features)

feature_matrix_train = torch.cat(feature_matrix_train, dim=0)

np.savez_compressed('ELMO_df3_train.npz', feature_matrix_train.numpy())

text_data_test = df3_test.tolist()
feature_matrix_test = []
for text in text_data_test:
    features = extract_features(text)
    feature_matrix_test.append(features)

feature_matrix_test = torch.cat(feature_matrix_test, dim=0)

np.savez_compressed('ELMO_df3_test.npz', feature_matrix_test.numpy())

text_data_train = df4_train.tolist()
feature_matrix_train = []
for text in text_data_train:
    features = extract_features(text)
    feature_matrix_train.append(features)

feature_matrix_train = torch.cat(feature_matrix_train, dim=0)

np.savez_compressed('ELMO_df4_train.npz', feature_matrix_train.numpy())

text_data_test = df4_test.tolist()
feature_matrix_test = []
for text in text_data_test:
    features = extract_features(text)
    feature_matrix_test.append(features)

feature_matrix_test = torch.cat(feature_matrix_test, dim=0)

np.savez_compressed('ELMO_df4_test.npz', feature_matrix_test.numpy())

# BERT
model_name = "bert-base-uncased"
model = BertModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

def preprocess_text(text):
    encoded_input = tokenizer(text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    input_ids = encoded_input["input_ids"]
    attention_mask = encoded_input["attention_mask"]
    return input_ids, attention_mask

def extract_features(text):
    input_ids, attention_mask = preprocess_text(text)
    with torch.no_grad():
        model_output = model(input_ids, attention_mask)
    features = model_output.last_hidden_state[:, -4:, :].mean(dim=1)  # Concatenate last four layers and take their mean
    return features

text_data_train = df1_train.tolist()
feature_matrix_train = []
for text in text_data_train:
    features = extract_features(text)
    feature_matrix_train.append(features)

feature_matrix_train = torch.cat(feature_matrix_train, dim=0)

np.savez_compressed('BERT_df1_train.npz', feature_matrix_train.numpy())

text_data_test = df1_test.tolist()
feature_matrix_test = []
for text in text_data_test:
    features = extract_features(text)
    feature_matrix_test.append(features)

feature_matrix_test = torch.cat(feature_matrix_test, dim=0)

np.savez_compressed('BERT_df1_test.npz', feature_matrix_test.numpy())

text_data_train = df2_train.tolist()
feature_matrix_train = []
for text in text_data_train:
    features = extract_features(text)
    feature_matrix_train.append(features)

feature_matrix_train = torch.cat(feature_matrix_train, dim=0)

np.savez_compressed('BERT_df2_train.npz', feature_matrix_train.numpy())

text_data_test = df2_test.tolist()
feature_matrix_test = []
for text in text_data_test:
    features = extract_features(text)
    feature_matrix_test.append(features)

feature_matrix_test = torch.cat(feature_matrix_test, dim=0)

np.savez_compressed('BERT_df2_test.npz', feature_matrix_test.numpy())

text_data_train = df3_train.tolist()
feature_matrix_train = []
for text in text_data_train:
    features = extract_features(text)
    feature_matrix_train.append(features)

feature_matrix_train = torch.cat(feature_matrix_train, dim=0)

np.savez_compressed('BERT_df3_train.npz', feature_matrix_train.numpy())

text_data_test = df3_test.tolist()
feature_matrix_test = []
for text in text_data_test:
    features = extract_features(text)
    feature_matrix_test.append(features)

feature_matrix_test = torch.cat(feature_matrix_test, dim=0)

np.savez_compressed('BERT_df3_test.npz', feature_matrix_test.numpy())

text_data_train = df4_train.tolist()
feature_matrix_train = []
for text in text_data_train:
    features = extract_features(text)
    feature_matrix_train.append(features)

feature_matrix_train = torch.cat(feature_matrix_train, dim=0)

np.savez_compressed('BERT_df4_train.npz', feature_matrix_train.numpy())

text_data_test = df4_test.tolist()
feature_matrix_test = []
for text in text_data_test:
    features = extract_features(text)
    feature_matrix_test.append(features)

feature_matrix_test = torch.cat(feature_matrix_test, dim=0)

np.savez_compressed('BERT_df4_test.npz', feature_matrix_test.numpy())


# DistilBERT
model_name = "distilbert-base-uncased"
model = DistilBertModel.from_pretrained(model_name)
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

def preprocess_text(text):
    encoded_input = tokenizer(text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    input_ids = encoded_input["input_ids"]
    attention_mask = encoded_input["attention_mask"]
    return input_ids, attention_mask

def extract_features(text):
    input_ids, attention_mask = preprocess_text(text)
    with torch.no_grad():
        model_output = model(input_ids, attention_mask)
    features = model_output.last_hidden_state[:, -4:, :].mean(dim=1)  # Concatenate last four layers and take their mean
    return features

text_data_train = df1_train.tolist()
feature_matrix_train = []
for text in text_data_train:
    features = extract_features(text)
    feature_matrix_train.append(features)

feature_matrix_train = torch.cat(feature_matrix_train, dim=0)

np.savez_compressed('DistilBERT_df1_train.npz', feature_matrix_train.numpy())

text_data_test = df1_test.tolist()
feature_matrix_test = []
for text in text_data_test:
    features = extract_features(text)
    feature_matrix_test.append(features)

feature_matrix_test = torch.cat(feature_matrix_test, dim=0)

np.savez_compressed('DistilBERT_df1_test.npz', feature_matrix_test.numpy())

text_data_train = df2_train.tolist()
feature_matrix_train = []
for text in text_data_train:
    features = extract_features(text)
    feature_matrix_train.append(features)

feature_matrix_train = torch.cat(feature_matrix_train, dim=0)

np.savez_compressed('DistilBERT_df2_train.npz', feature_matrix_train.numpy())

text_data_test = df2_test.tolist()
feature_matrix_test = []
for text in text_data_test:
    features = extract_features(text)
    feature_matrix_test.append(features)

feature_matrix_test = torch.cat(feature_matrix_test, dim=0)

np.savez_compressed('DistilBERT_df2_test.npz', feature_matrix_test.numpy())

text_data_train = df3_train.tolist()
feature_matrix_train = []
for text in text_data_train:
    features = extract_features(text)
    feature_matrix_train.append(features)

feature_matrix_train = torch.cat(feature_matrix_train, dim=0)

np.savez_compressed('DistilBERT_df3_train.npz', feature_matrix_train.numpy())

text_data_test = df3_test.tolist()
feature_matrix_test = []
for text in text_data_test:
    features = extract_features(text)
    feature_matrix_test.append(features)

feature_matrix_test = torch.cat(feature_matrix_test, dim=0)

np.savez_compressed('DistilBERT_df3_test.npz', feature_matrix_test.numpy())

text_data_train = df4_train.tolist()
feature_matrix_train = []
for text in text_data_train:
    features = extract_features(text)
    feature_matrix_train.append(features)

feature_matrix_train = torch.cat(feature_matrix_train, dim=0)

np.savez_compressed('DistilBERT_df4_train.npz', feature_matrix_train.numpy())

text_data_test = df4_test.tolist()
feature_matrix_test = []
for text in text_data_test:
    features = extract_features(text)
    feature_matrix_test.append(features)

feature_matrix_test = torch.cat(feature_matrix_test, dim=0)

np.savez_compressed('DistilBERT_df4_test.npz', feature_matrix_test.numpy())


# ALBERT
model_name = "albert-base-v2"
model = AlbertModel.from_pretrained(model_name)
tokenizer = AlbertTokenizer.from_pretrained(model_name)

def preprocess_text(text):
    encoded_input = tokenizer(text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    input_ids = encoded_input["input_ids"]
    attention_mask = encoded_input["attention_mask"]
    return input_ids, attention_mask

def extract_features(text):
    input_ids, attention_mask = preprocess_text(text)
    with torch.no_grad():
        model_output = model(input_ids, attention_mask)
    features = model_output.last_hidden_state[:, -4:, :].mean(dim=1)  # Concatenate last four layers and take their mean
    return features

text_data_train = df1_train.tolist()
feature_matrix_train = []
for text in text_data_train:
    features = extract_features(text)
    feature_matrix_train.append(features)

feature_matrix_train = torch.cat(feature_matrix_train, dim=0)

np.savez_compressed('ALBERT_df1_train.npz', feature_matrix_train.numpy())

text_data_test = df1_test.tolist()
feature_matrix_test = []
for text in text_data_test:
    features = extract_features(text)
    feature_matrix_test.append(features)

feature_matrix_test = torch.cat(feature_matrix_test, dim=0)

np.savez_compressed('ALBERT_df1_test.npz', feature_matrix_test.numpy())

text_data_train = df2_train.tolist()
feature_matrix_train = []
for text in text_data_train:
    features = extract_features(text)
    feature_matrix_train.append(features)

feature_matrix_train = torch.cat(feature_matrix_train, dim=0)

np.savez_compressed('ALBERT_df2_train.npz', feature_matrix_train.numpy())

text_data_test = df2_test.tolist()
feature_matrix_test = []
for text in text_data_test:
    features = extract_features(text)
    feature_matrix_test.append(features)

feature_matrix_test = torch.cat(feature_matrix_test, dim=0)

np.savez_compressed('ALBERT_df2_test.npz', feature_matrix_test.numpy())

text_data_train = df3_train.tolist()
feature_matrix_train = []
for text in text_data_train:
    features = extract_features(text)
    feature_matrix_train.append(features)

feature_matrix_train = torch.cat(feature_matrix_train, dim=0)

np.savez_compressed('ALBERT_df3_train.npz', feature_matrix_train.numpy())

text_data_test = df3_test.tolist()
feature_matrix_test = []
for text in text_data_test:
    features = extract_features(text)
    feature_matrix_test.append(features)

feature_matrix_test = torch.cat(feature_matrix_test, dim=0)

np.savez_compressed('ALBERT_df3_test.npz', feature_matrix_test.numpy())

text_data_train = df4_train.tolist()
feature_matrix_train = []
for text in text_data_train:
    features = extract_features(text)
    feature_matrix_train.append(features)

feature_matrix_train = torch.cat(feature_matrix_train, dim=0)

np.savez_compressed('ALBERT_df4_train.npz', feature_matrix_train.numpy())

text_data_test = df4_test.tolist()
feature_matrix_test = []
for text in text_data_test:
    features = extract_features(text)
    feature_matrix_test.append(features)

feature_matrix_test = torch.cat(feature_matrix_test, dim=0)

np.savez_compressed('ALBERT_df4_test.npz', feature_matrix_test.numpy())

# BART
model_name = "facebook/bart-base"
model = BartModel.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

def preprocess_text(text):
    encoded_input = tokenizer(text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    input_ids = encoded_input["input_ids"]
    attention_mask = encoded_input["attention_mask"]
    return input_ids, attention_mask

def extract_features(text):
    input_ids, attention_mask = preprocess_text(text)
    with torch.no_grad():
        model_output = model(input_ids, attention_mask)
    features = model_output.last_hidden_state[:, -4:, :].mean(dim=1)  # Concatenate last four layers and take their mean
    return features

text_data_train = df1_train.tolist()
feature_matrix_train = []
for text in text_data_train:
    features = extract_features(text)
    feature_matrix_train.append(features)

feature_matrix_train = torch.cat(feature_matrix_train, dim=0)

np.savez_compressed('BART_df1_train.npz', feature_matrix_train.numpy())

text_data_test = df1_test.tolist()
feature_matrix_test = []
for text in text_data_test:
    features = extract_features(text)
    feature_matrix_test.append(features)

feature_matrix_test = torch.cat(feature_matrix_test, dim=0)

np.savez_compressed('BART_df1_test.npz', feature_matrix_test.numpy())

text_data_train = df2_train.tolist()
feature_matrix_train = []
for text in text_data_train:
    features = extract_features(text)
    feature_matrix_train.append(features)

feature_matrix_train = torch.cat(feature_matrix_train, dim=0)

np.savez_compressed('BART_df2_train.npz', feature_matrix_train.numpy())

text_data_test = df2_test.tolist()
feature_matrix_test = []
for text in text_data_test:
    features = extract_features(text)
    feature_matrix_test.append(features)

feature_matrix_test = torch.cat(feature_matrix_test, dim=0)

np.savez_compressed('BART_df2_test.npz', feature_matrix_test.numpy())

text_data_train = df3_train.tolist()
feature_matrix_train = []
for text in text_data_train:
    features = extract_features(text)
    feature_matrix_train.append(features)

feature_matrix_train = torch.cat(feature_matrix_train, dim=0)

np.savez_compressed('BART_df3_train.npz', feature_matrix_train.numpy())

text_data_test = df3_test.tolist()
feature_matrix_test = []
for text in text_data_test:
    features = extract_features(text)
    feature_matrix_test.append(features)

feature_matrix_test = torch.cat(feature_matrix_test, dim=0)

np.savez_compressed('BART_df3_test.npz', feature_matrix_test.numpy())

text_data_train = df4_train.tolist()
feature_matrix_train = []
for text in text_data_train:
    features = extract_features(text)
    feature_matrix_train.append(features)

feature_matrix_train = torch.cat(feature_matrix_train, dim=0)

np.savez_compressed('BART_df4_train.npz', feature_matrix_train.numpy())

text_data_test = df4_test.tolist()
feature_matrix_test = []
for text in text_data_test:
    features = extract_features(text)
    feature_matrix_test.append(features)

feature_matrix_test = torch.cat(feature_matrix_test, dim=0)

np.savez_compressed('BART_df4_test.npz', feature_matrix_test.numpy())

# RoBERTa
model_name = "roberta-base"
model = RobertaModel.from_pretrained(model_name)
tokenizer = RobertaTokenizer.from_pretrained(model_name)

def preprocess_text(text):
    encoded_input = tokenizer(text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    input_ids = encoded_input["input_ids"]
    attention_mask = encoded_input["attention_mask"]
    return input_ids, attention_mask

def extract_features(text):
    input_ids, attention_mask = preprocess_text(text)
    with torch.no_grad():
        model_output = model(input_ids, attention_mask)
    features = model_output.last_hidden_state[:, -4:, :].mean(dim=1)  # Concatenate last four layers and take their mean
    return features

text_data_train = df1_train.tolist()
feature_matrix_train = []
for text in text_data_train:
    features = extract_features(text)
    feature_matrix_train.append(features)

feature_matrix_train = torch.cat(feature_matrix_train, dim=0)

np.savez_compressed('RoBERTa_df1_train.npz', feature_matrix_train.numpy())

text_data_test = df1_test.tolist()
feature_matrix_test = []
for text in text_data_test:
    features = extract_features(text)
    feature_matrix_test.append(features)

feature_matrix_test = torch.cat(feature_matrix_test, dim=0)

np.savez_compressed('RoBERTa_df1_test.npz', feature_matrix_test.numpy())

text_data_train = df2_train.tolist()
feature_matrix_train = []
for text in text_data_train:
    features = extract_features(text)
    feature_matrix_train.append(features)

feature_matrix_train = torch.cat(feature_matrix_train, dim=0)

np.savez_compressed('RoBERTa_df2_train.npz', feature_matrix_train.numpy())

text_data_test = df2_test.tolist()
feature_matrix_test = []
for text in text_data_test:
    features = extract_features(text)
    feature_matrix_test.append(features)

feature_matrix_test = torch.cat(feature_matrix_test, dim=0)

np.savez_compressed('RoBERTa_df2_test.npz', feature_matrix_test.numpy())

text_data_train = df3_train.tolist()
feature_matrix_train = []
for text in text_data_train:
    features = extract_features(text)
    feature_matrix_train.append(features)

feature_matrix_train = torch.cat(feature_matrix_train, dim=0)

np.savez_compressed('RoBERTa_df3_train.npz', feature_matrix_train.numpy())

text_data_test = df3_test.tolist()
feature_matrix_test = []
for text in text_data_test:
    features = extract_features(text)
    feature_matrix_test.append(features)

feature_matrix_test = torch.cat(feature_matrix_test, dim=0)

np.savez_compressed('RoBERTa_df3_test.npz', feature_matrix_test.numpy())

text_data_train = df4_train.tolist()
feature_matrix_train = []
for text in text_data_train:
    features = extract_features(text)
    feature_matrix_train.append(features)

feature_matrix_train = torch.cat(feature_matrix_train, dim=0)

np.savez_compressed('RoBERTa_df4_train.npz', feature_matrix_train.numpy())

text_data_test = df4_test.tolist()
feature_matrix_test = []
for text in text_data_test:
    features = extract_features(text)
    feature_matrix_test.append(features)

feature_matrix_test = torch.cat(feature_matrix_test, dim=0)

np.savez_compressed('RoBERTa_df4_test.npz', feature_matrix_test.numpy())

# ELECTRA
model_name = "google/electra-base-discriminator"
model = ElectraModel.from_pretrained(model_name)
tokenizer = ElectraTokenizer.from_pretrained(model_name)

def preprocess_text(text):
    encoded_input = tokenizer(text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    input_ids = encoded_input["input_ids"]
    attention_mask = encoded_input["attention_mask"]
    return input_ids, attention_mask

def extract_features(text):
    input_ids, attention_mask = preprocess_text(text)
    with torch.no_grad():
        model_output = model(input_ids, attention_mask)
    features = model_output.last_hidden_state[:, -4:, :].mean(dim=1)  # Concatenate last four layers and take their mean
    return features

text_data_train = df1_train.tolist()
feature_matrix_train = []
for text in text_data_train:
    features = extract_features(text)
    feature_matrix_train.append(features)

feature_matrix_train = torch.cat(feature_matrix_train, dim=0)

np.savez_compressed('ELECTRA_df1_train.npz', feature_matrix_train.numpy())

text_data_test = df1_test.tolist()
feature_matrix_test = []
for text in text_data_test:
    features = extract_features(text)
    feature_matrix_test.append(features)

feature_matrix_test = torch.cat(feature_matrix_test, dim=0)

np.savez_compressed('ELECTRA_df1_test.npz', feature_matrix_test.numpy())

text_data_train = df2_train.tolist()
feature_matrix_train = []
for text in text_data_train:
    features = extract_features(text)
    feature_matrix_train.append(features)

feature_matrix_train = torch.cat(feature_matrix_train, dim=0)

np.savez_compressed('ELECTRA_df2_train.npz', feature_matrix_train.numpy())

text_data_test = df2_test.tolist()
feature_matrix_test = []
for text in text_data_test:
    features = extract_features(text)
    feature_matrix_test.append(features)

feature_matrix_test = torch.cat(feature_matrix_test, dim=0)

np.savez_compressed('ELECTRA_df2_test.npz', feature_matrix_test.numpy())

text_data_train = df3_train.tolist()
feature_matrix_train = []
for text in text_data_train:
    features = extract_features(text)
    feature_matrix_train.append(features)

feature_matrix_train = torch.cat(feature_matrix_train, dim=0)

np.savez_compressed('ELECTRA_df3_train.npz', feature_matrix_train.numpy())

text_data_test = df3_test.tolist()
feature_matrix_test = []
for text in text_data_test:
    features = extract_features(text)
    feature_matrix_test.append(features)

feature_matrix_test = torch.cat(feature_matrix_test, dim=0)

np.savez_compressed('ELECTRA_df3_test.npz', feature_matrix_test.numpy())

text_data_train = df4_train.tolist()
feature_matrix_train = []
for text in text_data_train:
    features = extract_features(text)
    feature_matrix_train.append(features)

feature_matrix_train = torch.cat(feature_matrix_train, dim=0)

np.savez_compressed('ELECTRA_df4_train.npz', feature_matrix_train.numpy())

text_data_test = df4_test.tolist()
feature_matrix_test = []
for text in text_data_test:
    features = extract_features(text)
    feature_matrix_test.append(features)

feature_matrix_test = torch.cat(feature_matrix_test, dim=0)

np.savez_compressed('ELECTRA_df4_test.npz', feature_matrix_test.numpy())

# XLnet
model_name = "xlnet-base-cased"
model = XLNetModel.from_pretrained(model_name)
tokenizer = XLNetTokenizer.from_pretrained(model_name)

def preprocess_text(text):
    encoded_input = tokenizer(text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    input_ids = encoded_input["input_ids"]
    attention_mask = encoded_input["attention_mask"]
    return input_ids, attention_mask

def extract_features(text):
    input_ids, attention_mask = preprocess_text(text)
    with torch.no_grad():
        model_output = model(input_ids, attention_mask)
    features = model_output.last_hidden_state[:, -4:, :].mean(dim=1)  # Concatenate last four layers and take their mean
    return features

text_data_train = df1_train.tolist()
feature_matrix_train = []
for text in text_data_train:
    features = extract_features(text)
    feature_matrix_train.append(features)

feature_matrix_train = torch.cat(feature_matrix_train, dim=0)

np.savez_compressed('XLNET_df1_train.npz', feature_matrix_train.numpy())

text_data_test = df1_test.tolist()
feature_matrix_test = []
for text in text_data_test:
    features = extract_features(text)
    feature_matrix_test.append(features)

feature_matrix_test = torch.cat(feature_matrix_test, dim=0)

np.savez_compressed('XLNET_df1_test.npz', feature_matrix_test.numpy())

text_data_train = df2_train.tolist()
feature_matrix_train = []
for text in text_data_train:
    features = extract_features(text)
    feature_matrix_train.append(features)

feature_matrix_train = torch.cat(feature_matrix_train, dim=0)

np.savez_compressed('XLNET_df2_train.npz', feature_matrix_train.numpy())

text_data_test = df2_test.tolist()
feature_matrix_test = []
for text in text_data_test:
    features = extract_features(text)
    feature_matrix_test.append(features)

feature_matrix_test = torch.cat(feature_matrix_test, dim=0)

np.savez_compressed('XLNET_df2_test.npz', feature_matrix_test.numpy())

text_data_train = df3_train.tolist()
feature_matrix_train = []
for text in text_data_train:
    features = extract_features(text)
    feature_matrix_train.append(features)

feature_matrix_train = torch.cat(feature_matrix_train, dim=0)

np.savez_compressed('XLNET_df3_train.npz', feature_matrix_train.numpy())

text_data_test = df3_test.tolist()
feature_matrix_test = []
for text in text_data_test:
    features = extract_features(text)
    feature_matrix_test.append(features)

feature_matrix_test = torch.cat(feature_matrix_test, dim=0)

np.savez_compressed('XLNET_df3_test.npz', feature_matrix_test.numpy())

text_data_train = df4_train.tolist()
feature_matrix_train = []
for text in text_data_train:
    features = extract_features(text)
    feature_matrix_train.append(features)

feature_matrix_train = torch.cat(feature_matrix_train, dim=0)

np.savez_compressed('XLNET_df4_train.npz', feature_matrix_train.numpy())

text_data_test = df4_test.tolist()
feature_matrix_test = []
for text in text_data_test:
    features = extract_features(text)
    feature_matrix_test.append(features)

feature_matrix_test = torch.cat(feature_matrix_test, dim=0)

np.savez_compressed('XLNET_df4_test.npz', feature_matrix_test.numpy())

















