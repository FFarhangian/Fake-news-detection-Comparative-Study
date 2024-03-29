import tensorflow_hub as hub
import tensorflow.compat.v1 as tf
from numpy import savez_compressed

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
