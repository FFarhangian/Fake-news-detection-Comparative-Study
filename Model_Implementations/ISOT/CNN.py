########################################## Word2Vec ##################################################
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Bidirectional
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import joblib
import timeit

MAX_NB_WORDS = 20000
MAX_SEQUENCE_LENGTH = 300

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(df2_train)
seq_train = tokenizer.texts_to_sequences(df2_train)
seq_test = tokenizer.texts_to_sequences(df2_test)
data_train = pad_sequences(seq_train.tolist(), maxlen=MAX_SEQUENCE_LENGTH)
data_test = pad_sequences(seq_test.tolist(), maxlen=MAX_SEQUENCE_LENGTH)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

# Load word embeddings
word_vectors = KeyedVectors.load_word2vec_format('/content/gdrive/MyDrive/GoogleNews-vectors-negative300.bin', binary=True)
vocabulary_size = min(len(word_index) + 1, MAX_NB_WORDS)
embedding_matrix = np.zeros((vocabulary_size, 300))
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    try:
        embedding_vector = word_vectors[word]
        embedding_matrix[i] = embedding_vector
    except KeyError:
        embedding_matrix[i] = np.random.normal(0, np.sqrt(0.25), 300)

# Define hyperparameters
activations = ['sigmoid', 'softmax']
batch_sizes = [64, 128, 512]
num_epochs = [5, 20, 100]
learning_rate = 0.001
dropout_rate = 0.2

# Perform 5-fold cross-validation
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

best_accuracy = 0.0
best_model = None

for activation in activations:
    for batch_size in batch_sizes:
        for epochs in num_epochs:
            fold = 1
            print(f"Activation: {activation}, Batch Size: {batch_size}, Epochs: {epochs}")
            accuracies = []

            for train_index, val_index in kf.split(data_train):
                # Split the data into training and validation sets
                train_data, val_data = data_train[train_index], data_train[val_index]
                train_labels, val_labels = df2_train_class[train_index], df2_train_class[val_index]

                # Create the model
                model = Sequential()
                model.add(Embedding(MAX_NB_WORDS, 300, input_length=MAX_SEQUENCE_LENGTH))
                model.add(Conv1D(128, 5, activation='relu'))
                model.add(Conv1D(32, 3, activation='relu'))
                model.add(GlobalMaxPooling1D())
                model.add(Dense(180, activation='relu'))
                model.add(Dense(180, activation='relu'))
                model.add(Dropout(dropout_rate))

                if activation == 'sigmoid':
                    model.add(Dense(1, activation=activation))
                else:
                    model.add(Dense(num_classes, activation=activation))

                # Compile the model
                model.compile(loss='binary_crossentropy' if activation == 'sigmoid' else 'categorical_crossentropy',
                              optimizer='adam', metrics=['accuracy'])

                # Train the model
                model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, verbose=0)

                # Evaluate on the validation set
                val_pred = model.predict(val_data)
                val_pred_classes = np.argmax(val_pred, axis=1) if activation != 'sigmoid' else np.where(val_pred > 0.5, 1, 0)
                val_labels_classes = np.argmax(val_labels, axis=1) if activation != 'sigmoid' else val_labels.reshape(-1)
                accuracy = accuracy_score(val_labels_classes, val_pred_classes)
                accuracies.append(accuracy)

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model

                fold += 1

            mean_accuracy = np.mean(accuracies)
            print(f"Mean Validation Accuracy: {mean_accuracy:.2f}")

# Train the best model on the full training set
best_model.fit(data_train, df2_train_class, epochs=num_epochs, batch_size=batch_size, verbose=0)

# Predict on the test set
df2_Word2Vec_CNN_pred = best_model.predict(data_test)
df2_Word2Vec_CNN_pred_classes = np.argmax(df2_Word2Vec_CNN_pred, axis=1) if activations != 'sigmoid' else np.where(df2_Word2Vec_CNN_pred > 0.5, 1, 0)

# Evaluate performance on the test set
test_accuracy = accuracy_score(df2_test_class, df2_Word2Vec_CNN_pred_classes)
print("Test Accuracy: %.2f%%" % (test_accuracy * 100))

# Save predictions and model
np.savetxt("df2_Word2Vec_CNN_pred.csv", df2_Word2Vec_CNN_pred_classes, delimiter=",")
joblib.dump(best_model, 'df2_Word2Vec_CNN.sav')

########################################## GloVe ##################################################
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Bidirectional
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import joblib
import timeit

# Load GloVe
!wget http://nlp.stanford.edu/data/glove.6B.zip
!unzip glove*.zip

# Load GloVe embeddings
embeddings_index = {}
with open('/content/glove.6B.300d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# Tokenize and pad sequences
MAX_NB_WORDS = 20000
MAX_SEQUENCE_LENGTH = 300
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(df2_train)
seq_train = tokenizer.texts_to_sequences(df2_train)
seq_test = tokenizer.texts_to_sequences(df2_test)
data_train = pad_sequences(seq_train, maxlen=MAX_SEQUENCE_LENGTH)
data_test = pad_sequences(seq_test, maxlen=MAX_SEQUENCE_LENGTH)

# Create embedding matrix
embedding_matrix = np.zeros((MAX_NB_WORDS, 300))
for word, index in tokenizer.word_index.items():
    if index >= MAX_NB_WORDS:
        break
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

# Define hyperparameters
activations = ['sigmoid', 'softmax']
batch_sizes = [64, 128, 512]
num_epochs = [5, 20, 100]
learning_rate = 0.001
dropout_rate = 0.2

# Perform 5-fold cross-validation
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

best_accuracy = 0.0
best_model = None

for activation in activations:
    for batch_size in batch_sizes:
        for epochs in num_epochs:
            fold = 1
            print(f"Activation: {activation}, Batch Size: {batch_size}, Epochs: {epochs}")
            accuracies = []

            for train_index, val_index in kf.split(data_train):
                # Split the data into training and validation sets
                train_data, val_data = data_train[train_index], data_train[val_index]
                train_labels, val_labels = df2_train_class[train_index], df2_train_class[val_index]

                # Create the model
                model = Sequential()
                model.add(Embedding(MAX_NB_WORDS, 300, input_length=MAX_SEQUENCE_LENGTH))
                model.add(Conv1D(128, 5, activation='relu'))
                model.add(Conv1D(32, 3, activation='relu'))
                model.add(GlobalMaxPooling1D())
                model.add(Dense(180, activation='relu'))
                model.add(Dense(180, activation='relu'))
                model.add(Dropout(dropout_rate))

                if activation == 'sigmoid':
                    model.add(Dense(1, activation=activation))
                else:
                    model.add(Dense(num_classes, activation=activation))

                # Compile the model
                model.compile(loss='binary_crossentropy' if activation == 'sigmoid' else 'categorical_crossentropy',
                              optimizer='adam', metrics=['accuracy'])

                # Train the model
                model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, verbose=0)

                # Evaluate on the validation set
                val_pred = model.predict(val_data)
                val_pred_classes = np.argmax(val_pred, axis=1) if activation != 'sigmoid' else np.where(val_pred > 0.5, 1, 0)
                val_labels_classes = np.argmax(val_labels, axis=1) if activation != 'sigmoid' else val_labels.reshape(-1)
                accuracy = accuracy_score(val_labels_classes, val_pred_classes)
                accuracies.append(accuracy)

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model

                fold += 1

            mean_accuracy = np.mean(accuracies)
            print(f"Mean Validation Accuracy: {mean_accuracy:.2f}")

# Train the best model on the full training set
best_model.fit(data_train, df2_train_class, epochs=num_epochs, batch_size=batch_size, verbose=0)

# Predict on the test set
df2_Glove_CNN_pred = best_model.predict(data_test)
df2_Glove_CNN_pred_classes = np.argmax(df2_Glove_CNN_pred, axis=1) if activations != 'sigmoid' else np.where(df2_Glove_CNN_pred > 0.5, 1, 0)

# Evaluate performance on the test set
test_accuracy = accuracy_score(df2_test_class, df2_Glove_CNN_pred_classes)
print("Test Accuracy: %.2f%%" % (test_accuracy * 100))

# Save predictions and model
np.savetxt("df2_Glove_CNN_pred.csv", df2_Glove_CNN_pred_classes, delimiter=",")
joblib.dump(best_model, 'df2_Glove_CNN.sav')

########################################## FastText ##################################################
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Bidirectional
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import joblib
import timeit

#Load Fasttext
!wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip
!unzip wiki-news-300d-1M.vec.zip

# Load Fasttext embeddings
embeddings_index = {}
with open('/content/wiki-news-300d-1M.vec', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# Tokenize and pad sequences
MAX_NB_WORDS = 20000
MAX_SEQUENCE_LENGTH = 300
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(df2_train)
seq_train = tokenizer.texts_to_sequences(df2_train)
seq_test = tokenizer.texts_to_sequences(df2_test)
data_train = pad_sequences(seq_train, maxlen=MAX_SEQUENCE_LENGTH)
data_test = pad_sequences(seq_test, maxlen=MAX_SEQUENCE_LENGTH)

# Create embedding matrix
embedding_matrix = np.zeros((MAX_NB_WORDS, 300))
for word, index in tokenizer.word_index.items():
    if index >= MAX_NB_WORDS:
        break
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

# Define hyperparameters
activations = ['sigmoid', 'softmax']
batch_sizes = [64, 128, 512]
num_epochs = [5, 20, 100]
learning_rate = 0.001
dropout_rate = 0.2

# Perform 5-fold cross-validation
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

best_accuracy = 0.0
best_model = None

for activation in activations:
    for batch_size in batch_sizes:
        for epochs in num_epochs:
            fold = 1
            print(f"Activation: {activation}, Batch Size: {batch_size}, Epochs: {epochs}")
            accuracies = []

            for train_index, val_index in kf.split(data_train):
                # Split the data into training and validation sets
                train_data, val_data = data_train[train_index], data_train[val_index]
                train_labels, val_labels = df2_train_class[train_index], df2_train_class[val_index]

                # Create the model
                model = Sequential()
                model.add(Embedding(MAX_NB_WORDS, 300, input_length=MAX_SEQUENCE_LENGTH))
                model.add(Conv1D(128, 5, activation='relu'))
                model.add(Conv1D(32, 3, activation='relu'))
                model.add(GlobalMaxPooling1D())
                model.add(Dense(180, activation='relu'))
                model.add(Dense(180, activation='relu'))
                model.add(Dropout(dropout_rate))

                if activation == 'sigmoid':
                    model.add(Dense(1, activation=activation))
                else:
                    model.add(Dense(num_classes, activation=activation))

                # Compile the model
                model.compile(loss='binary_crossentropy' if activation == 'sigmoid' else 'categorical_crossentropy',
                              optimizer='adam', metrics=['accuracy'])

                # Train the model
                model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, verbose=0)

                # Evaluate on the validation set
                val_pred = model.predict(val_data)
                val_pred_classes = np.argmax(val_pred, axis=1) if activation != 'sigmoid' else np.where(val_pred > 0.5, 1, 0)
                val_labels_classes = np.argmax(val_labels, axis=1) if activation != 'sigmoid' else val_labels.reshape(-1)
                accuracy = accuracy_score(val_labels_classes, val_pred_classes)
                accuracies.append(accuracy)

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model

                fold += 1

            mean_accuracy = np.mean(accuracies)
            print(f"Mean Validation Accuracy: {mean_accuracy:.2f}")

# Train the best model on the full training set
best_model.fit(data_train, df2_train_class, epochs=num_epochs, batch_size=batch_size, verbose=0)

# Predict on the test set
df2_Fasttext_CNN_pred = best_model.predict(data_test)
df2_Fasttext_CNN_pred_classes = np.argmax(df2_Fasttext_CNN_pred, axis=1) if activations != 'sigmoid' else np.where(df2_Fasttext_CNN_pred > 0.5, 1, 0)

# Evaluate performance on the test set
test_accuracy = accuracy_score(df2_test_class, df2_Fasttext_CNN_pred_classes)
print("Test Accuracy: %.2f%%" % (test_accuracy * 100))

# Save predictions and model
np.savetxt("df2_Fasttext_CNN_pred.csv", df2_Fasttext_CNN_pred_classes, delimiter=",")
joblib.dump(best_model, 'df2_Fasttext_CNN.sav')


#################################################### ELMO ##########################################

