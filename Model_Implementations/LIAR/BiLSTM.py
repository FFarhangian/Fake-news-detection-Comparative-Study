############################################# Word2Vec ####################################################


############################################# GloVe ####################################################
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Bidirectional
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import joblib
import timeit

# Tokenize and pad sequences
MAX_NB_WORDS = 20000
MAX_SEQUENCE_LENGTH = 300
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(df1_train)
seq_train = tokenizer.texts_to_sequences(df1_train)
seq_test = tokenizer.texts_to_sequences(df1_test)
data_train = pad_sequences(seq_train, maxlen=MAX_SEQUENCE_LENGTH)
data_test = pad_sequences(seq_test, maxlen=MAX_SEQUENCE_LENGTH)

# Define hyperparameters
activations = 'softmax'
batch_sizes = [64, 128, 512]
num_epochs = [5, 20, 100]
learning_rate = 0.001
hidden_size = 128
dropout_rate = 0.2

# Perform 5-fold cross-validation
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

best_accuracy = 0.0
best_model = None

for batch_size in batch_sizes:
    for epochs in num_epochs:
        fold = 1
        print(f"Batch Size: {batch_size}, Epochs: {epochs}")
        accuracies = []

        for train_index, val_index in kf.split(data_train):
            # Split the data into training and validation sets
            train_data, val_data = data_train[train_index], data_train[val_index]
            train_labels, val_labels = df1_train_class[train_index], df1_train_class[val_index]

            # Create the model
            model = Sequential()
            model.add(Embedding(MAX_NB_WORDS, 300, input_length=MAX_SEQUENCE_LENGTH, name="embeddinglayer",
                                weights=[embedding_matrix], trainable=False))
            model.add(Bidirectional(LSTM(hidden_size, dropout=dropout_rate, recurrent_dropout=dropout_rate)))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(num_classes, activation=activations))

            # Compile the model
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            # Train the model
            model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, verbose=0)

            # Evaluate on the validation set
            val_pred = model.predict(val_data)
            val_pred_classes = np.argmax(val_pred, axis=1)
            val_labels_classes = np.argmax(val_labels, axis=1)
            accuracy = accuracy_score(val_labels_classes, val_pred_classes)
            accuracies.append(accuracy)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model

            fold += 1

        mean_accuracy = np.mean(accuracies)
        print(f"Mean Validation Accuracy: {mean_accuracy:.2f}")

# Train the best model on the full training set
best_model.fit(data_train, df1_train_class, epochs=num_epochs, batch_size=batch_size, verbose=0)

# Predict on the test set
df1_Glove_BiLSTM_pred = best_model.predict(data_test)
df1_Glove_BiLSTM_pred_classes = np.argmax(df1_Glove_BiLSTM_pred, axis=1)

# Evaluate performance on the test set
test_accuracy = accuracy_score(df1_test_class, df1_Glove_BiLSTM_pred_classes)
print("Test Accuracy: %.2f%%" % (test_accuracy * 100))

# Save predictions and model
np.savetxt("df1_Glove_BiLSTM_pred.csv", df1_Glove_BiLSTM_pred_classes, delimiter=",")
joblib.dump(best_model, 'df1_Glove_BiLSTM.sav')

############################################# FastText ####################################################


############################################# ELMO ####################################################
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.externals import joblib
import pandas as pd
import time

# Load the ELMO features
feature_matrix_train = np.load('ELMO_df1_train.npz')['arr_0']
feature_matrix_test = np.load('ELMO_df1_test.npz')['arr_0']
labels_train = df1_train_class.to_numpy().flatten()
labels_test = df1_test_class.to_numpy().flatten()

# Convert the numpy arrays to PyTorch tensors
features_train = torch.tensor(feature_matrix_train, dtype=torch.float32)
features_test = torch.tensor(feature_matrix_test, dtype=torch.float32)
labels_train = torch.tensor(labels_train, dtype=torch.long)
labels_test = torch.tensor(labels_test, dtype=torch.long)

# Define the BiLSTM model
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate=0.2, activation='relu'):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_size * 2, 32)
        self.fc2 = nn.Linear(32, num_classes)
        self.activation = self.get_activation(activation)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), self.hidden_size).to(x.device)  # Initialize hidden state for bidirectional LSTM
        c0 = torch.zeros(2, x.size(0), self.hidden_size).to(x.device)  # Initialize cell state for bidirectional LSTM
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])  # Take the last time step output and apply dropout
        out = self.fc1(out)
        out = self.activation(out)
        out = self.fc2(out)
        return out

    def get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        else:
            raise ValueError("Invalid activation function.")

# Set the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the hyperparameters
batch_sizes = [64, 128, 512]
num_epochs_list = [5, 20, 100]
activations = ['sigmoid', 'relu']
learning_rate = 0.001
hidden_size = 128
dropout_rate = 0.2

# Perform 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

best_accuracy = 0.0  # Variable to store the best validation accuracy
best_model = None  # Variable to store the best model

for activation in activations:
    for batch_size in batch_sizes:
        for num_epochs in num_epochs_list:
            fold = 1
            print(f"Activation: {activation}, Batch Size: {batch_size}, Epochs: {num_epochs}")
            for train_index, val_index in kf.split(features_train):
                # Split the data into training and validation sets for hyperparameter tuning and find the best model
                train_features, val_features = features_train[train_index], features_train[val_index]
                train_labels, val_labels = labels_train[train_index], labels_train[val_index]

                # Create DataLoaders for training and validation datasets
                train_dataset = TensorDataset(train_features, train_labels)
                val_dataset = TensorDataset(val_features, val_labels)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

                # Initialize the BiLSTM model
                input_size = features_train.size(1)
                num_classes = torch.max(labels_train) + 1
                model = BiLSTM(input_size, hidden_size, num_classes, dropout_rate, activation).to(device)

                # Define the loss function and optimizer
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)

                # Train the BiLSTM model
                model.train()
                start_time = time.time()
                for epoch in range(num_epochs):
                    for inputs, labels in train_loader:
                        inputs = inputs.unsqueeze(1).to(device)  # Add an extra dimension for the LSTM input
                        labels = labels.to(device)
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                    print(f"Fold {fold}, Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
                end_time = time.time()
                train_time = end_time - start_time
                print(f"Training Time: {train_time:.2f} seconds")

                # Test the model
                model.eval()
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for inputs, labels in val_loader:
                        inputs = inputs.unsqueeze(1).to(device)
                        labels = labels.to(device)
                        outputs = model(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    accuracy = correct / total
                    print(f"Fold {fold}, Validation Accuracy: {accuracy:.2%}")
                    fold += 1

                    # Check if the current model is the best based on validation accuracy
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_model = model.state_dict().copy()

# Load the best model
model.load_state_dict(best_model)

# Save the best model
joblib.dump(model, 'df1_ELMO_BiLSTM.sav')

# Evaluate the best model on the test set
model.eval()
test_dataset = TensorDataset(features_test, labels_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
with torch.no_grad():
    correct = 0
    total = 0
    predictions = []
    start_time = time.time()
    for inputs, labels in test_loader:
        inputs = inputs.unsqueeze(1).to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        predictions.extend(predicted.cpu().numpy())
    end_time = time.time()
    test_time = end_time - start_time
    test_accuracy = correct / total
    print(f"Test Accuracy: {test_accuracy:.2%}")
    print(f"Test Time: {test_time:.2f} seconds")

    # Save the predictions as a CSV file
    predictions_df = pd.DataFrame({'True Label': labels_test.cpu().numpy(), 'Predicted Label': predictions})
    predictions_df.to_csv('predictions.csv', index=False)

    # Generate classification report for the test data
    classification_rep = classification_report(labels_test.cpu().numpy(), predictions)
    print("Classification Report:")
    print(classification_rep)


############################################# BERT ####################################################
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.externals import joblib
import pandas as pd
import time

# Load the BERT features
feature_matrix_train = np.load('BERT_df1_train.npz')['arr_0']
feature_matrix_test = np.load('BERT_df1_test.npz')['arr_0']
labels_train = df1_train_class.to_numpy().flatten()
labels_test = df1_test_class.to_numpy().flatten()

# Convert the numpy arrays to PyTorch tensors
features_train = torch.tensor(feature_matrix_train, dtype=torch.float32)
features_test = torch.tensor(feature_matrix_test, dtype=torch.float32)
labels_train = torch.tensor(labels_train, dtype=torch.long)
labels_test = torch.tensor(labels_test, dtype=torch.long)

# Define the BiLSTM model
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate=0.2, activation='relu'):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_size * 2, 32)
        self.fc2 = nn.Linear(32, num_classes)
        self.activation = self.get_activation(activation)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), self.hidden_size).to(x.device)  # Initialize hidden state for bidirectional LSTM
        c0 = torch.zeros(2, x.size(0), self.hidden_size).to(x.device)  # Initialize cell state for bidirectional LSTM
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])  # Take the last time step output and apply dropout
        out = self.fc1(out)
        out = self.activation(out)
        out = self.fc2(out)
        return out

    def get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        else:
            raise ValueError("Invalid activation function.")

# Set the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the hyperparameters
batch_sizes = [64, 128, 512]
num_epochs_list = [5, 20, 100]
activations = ['sigmoid', 'relu']
learning_rate = 0.001
hidden_size = 128
dropout_rate = 0.2

# Perform 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

best_accuracy = 0.0  # Variable to store the best validation accuracy
best_model = None  # Variable to store the best model

for activation in activations:
    for batch_size in batch_sizes:
        for num_epochs in num_epochs_list:
            fold = 1
            print(f"Activation: {activation}, Batch Size: {batch_size}, Epochs: {num_epochs}")
            for train_index, val_index in kf.split(features_train):
                # Split the data into training and validation sets for hyperparameter tuning and find the best model
                train_features, val_features = features_train[train_index], features_train[val_index]
                train_labels, val_labels = labels_train[train_index], labels_train[val_index]

                # Create DataLoaders for training and validation datasets
                train_dataset = TensorDataset(train_features, train_labels)
                val_dataset = TensorDataset(val_features, val_labels)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

                # Initialize the BiLSTM model
                input_size = features_train.size(1)
                num_classes = torch.max(labels_train) + 1
                model = BiLSTM(input_size, hidden_size, num_classes, dropout_rate, activation).to(device)

                # Define the loss function and optimizer
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)

                # Train the BiLSTM model
                model.train()
                start_time = time.time()
                for epoch in range(num_epochs):
                    for inputs, labels in train_loader:
                        inputs = inputs.unsqueeze(1).to(device)  # Add an extra dimension for the LSTM input
                        labels = labels.to(device)
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                    print(f"Fold {fold}, Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
                end_time = time.time()
                train_time = end_time - start_time
                print(f"Training Time: {train_time:.2f} seconds")

                # Test the model
                model.eval()
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for inputs, labels in val_loader:
                        inputs = inputs.unsqueeze(1).to(device)
                        labels = labels.to(device)
                        outputs = model(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    accuracy = correct / total
                    print(f"Fold {fold}, Validation Accuracy: {accuracy:.2%}")
                    fold += 1

                    # Check if the current model is the best based on validation accuracy
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_model = model.state_dict().copy()

# Load the best model
model.load_state_dict(best_model)

# Save the best model
joblib.dump(model, 'df1_BERT_BiLSTM.sav')

# Evaluate the best model on the test set
model.eval()
test_dataset = TensorDataset(features_test, labels_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
with torch.no_grad():
    correct = 0
    total = 0
    predictions = []
    start_time = time.time()
    for inputs, labels in test_loader:
        inputs = inputs.unsqueeze(1).to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        predictions.extend(predicted.cpu().numpy())
    end_time = time.time()
    test_time = end_time - start_time
    test_accuracy = correct / total
    print(f"Test Accuracy: {test_accuracy:.2%}")
    print(f"Test Time: {test_time:.2f} seconds")

    # Save the predictions as a CSV file
    predictions_df = pd.DataFrame({'True Label': labels_test.cpu().numpy(), 'Predicted Label': predictions})
    predictions_df.to_csv('predictions.csv', index=False)

    # Generate classification report for the test data
    classification_rep = classification_report(labels_test.cpu().numpy(), predictions)
    print("Classification Report:")
    print(classification_rep)


############################################# DistilBERT ####################################################
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.externals import joblib
import pandas as pd
import time

# Load the DistilBERT features
feature_matrix_train = np.load('DistilBERT_df1_train.npz')['arr_0']
feature_matrix_test = np.load('DistilBERT_df1_test.npz')['arr_0']
labels_train = df1_train_class.to_numpy().flatten()
labels_test = df1_test_class.to_numpy().flatten()

# Convert the numpy arrays to PyTorch tensors
features_train = torch.tensor(feature_matrix_train, dtype=torch.float32)
features_test = torch.tensor(feature_matrix_test, dtype=torch.float32)
labels_train = torch.tensor(labels_train, dtype=torch.long)
labels_test = torch.tensor(labels_test, dtype=torch.long)

# Define the BiLSTM model
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate=0.2, activation='relu'):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_size * 2, 32)
        self.fc2 = nn.Linear(32, num_classes)
        self.activation = self.get_activation(activation)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), self.hidden_size).to(x.device)  # Initialize hidden state for bidirectional LSTM
        c0 = torch.zeros(2, x.size(0), self.hidden_size).to(x.device)  # Initialize cell state for bidirectional LSTM
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])  # Take the last time step output and apply dropout
        out = self.fc1(out)
        out = self.activation(out)
        out = self.fc2(out)
        return out

    def get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        else:
            raise ValueError("Invalid activation function.")

# Set the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the hyperparameters
batch_sizes = [64, 128, 512]
num_epochs_list = [5, 20, 100]
activations = ['sigmoid', 'relu']
learning_rate = 0.001
hidden_size = 128
dropout_rate = 0.2

# Perform 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

best_accuracy = 0.0  # Variable to store the best validation accuracy
best_model = None  # Variable to store the best model

for activation in activations:
    for batch_size in batch_sizes:
        for num_epochs in num_epochs_list:
            fold = 1
            print(f"Activation: {activation}, Batch Size: {batch_size}, Epochs: {num_epochs}")
            for train_index, val_index in kf.split(features_train):
                # Split the data into training and validation sets for hyperparameter tuning and find the best model
                train_features, val_features = features_train[train_index], features_train[val_index]
                train_labels, val_labels = labels_train[train_index], labels_train[val_index]

                # Create DataLoaders for training and validation datasets
                train_dataset = TensorDataset(train_features, train_labels)
                val_dataset = TensorDataset(val_features, val_labels)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

                # Initialize the BiLSTM model
                input_size = features_train.size(1)
                num_classes = torch.max(labels_train) + 1
                model = BiLSTM(input_size, hidden_size, num_classes, dropout_rate, activation).to(device)

                # Define the loss function and optimizer
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)

                # Train the BiLSTM model
                model.train()
                start_time = time.time()
                for epoch in range(num_epochs):
                    for inputs, labels in train_loader:
                        inputs = inputs.unsqueeze(1).to(device)  # Add an extra dimension for the LSTM input
                        labels = labels.to(device)
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                    print(f"Fold {fold}, Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
                end_time = time.time()
                train_time = end_time - start_time
                print(f"Training Time: {train_time:.2f} seconds")

                # Test the model
                model.eval()
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for inputs, labels in val_loader:
                        inputs = inputs.unsqueeze(1).to(device)
                        labels = labels.to(device)
                        outputs = model(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    accuracy = correct / total
                    print(f"Fold {fold}, Validation Accuracy: {accuracy:.2%}")
                    fold += 1

                    # Check if the current model is the best based on validation accuracy
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_model = model.state_dict().copy()

# Load the best model
model.load_state_dict(best_model)

# Save the best model
joblib.dump(model, 'df1_DistilBERT_BiLSTM.sav')

# Evaluate the best model on the test set
model.eval()
test_dataset = TensorDataset(features_test, labels_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
with torch.no_grad():
    correct = 0
    total = 0
    predictions = []
    start_time = time.time()
    for inputs, labels in test_loader:
        inputs = inputs.unsqueeze(1).to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        predictions.extend(predicted.cpu().numpy())
    end_time = time.time()
    test_time = end_time - start_time
    test_accuracy = correct / total
    print(f"Test Accuracy: {test_accuracy:.2%}")
    print(f"Test Time: {test_time:.2f} seconds")

    # Save the predictions as a CSV file
    predictions_df = pd.DataFrame({'True Label': labels_test.cpu().numpy(), 'Predicted Label': predictions})
    predictions_df.to_csv('predictions.csv', index=False)

    # Generate classification report for the test data
    classification_rep = classification_report(labels_test.cpu().numpy(), predictions)
    print("Classification Report:")
    print(classification_rep)


############################################# BART ####################################################
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.externals import joblib
import pandas as pd
import time

# Load the BART features
feature_matrix_train = np.load('BART_df1_train.npz')['arr_0']
feature_matrix_test = np.load('BART_df1_test.npz')['arr_0']
labels_train = df1_train_class.to_numpy().flatten()
labels_test = df1_test_class.to_numpy().flatten()

# Convert the numpy arrays to PyTorch tensors
features_train = torch.tensor(feature_matrix_train, dtype=torch.float32)
features_test = torch.tensor(feature_matrix_test, dtype=torch.float32)
labels_train = torch.tensor(labels_train, dtype=torch.long)
labels_test = torch.tensor(labels_test, dtype=torch.long)

# Define the BiLSTM model
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate=0.2, activation='relu'):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_size * 2, 32)
        self.fc2 = nn.Linear(32, num_classes)
        self.activation = self.get_activation(activation)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), self.hidden_size).to(x.device)  # Initialize hidden state for bidirectional LSTM
        c0 = torch.zeros(2, x.size(0), self.hidden_size).to(x.device)  # Initialize cell state for bidirectional LSTM
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])  # Take the last time step output and apply dropout
        out = self.fc1(out)
        out = self.activation(out)
        out = self.fc2(out)
        return out

    def get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        else:
            raise ValueError("Invalid activation function.")

# Set the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the hyperparameters
batch_sizes = [64, 128, 512]
num_epochs_list = [5, 20, 100]
activations = ['sigmoid', 'relu']
learning_rate = 0.001
hidden_size = 128
dropout_rate = 0.2

# Perform 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

best_accuracy = 0.0  # Variable to store the best validation accuracy
best_model = None  # Variable to store the best model

for activation in activations:
    for batch_size in batch_sizes:
        for num_epochs in num_epochs_list:
            fold = 1
            print(f"Activation: {activation}, Batch Size: {batch_size}, Epochs: {num_epochs}")
            for train_index, val_index in kf.split(features_train):
                # Split the data into training and validation sets for hyperparameter tuning and find the best model
                train_features, val_features = features_train[train_index], features_train[val_index]
                train_labels, val_labels = labels_train[train_index], labels_train[val_index]

                # Create DataLoaders for training and validation datasets
                train_dataset = TensorDataset(train_features, train_labels)
                val_dataset = TensorDataset(val_features, val_labels)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

                # Initialize the BiLSTM model
                input_size = features_train.size(1)
                num_classes = torch.max(labels_train) + 1
                model = BiLSTM(input_size, hidden_size, num_classes, dropout_rate, activation).to(device)

                # Define the loss function and optimizer
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)

                # Train the BiLSTM model
                model.train()
                start_time = time.time()
                for epoch in range(num_epochs):
                    for inputs, labels in train_loader:
                        inputs = inputs.unsqueeze(1).to(device)  # Add an extra dimension for the LSTM input
                        labels = labels.to(device)
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                    print(f"Fold {fold}, Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
                end_time = time.time()
                train_time = end_time - start_time
                print(f"Training Time: {train_time:.2f} seconds")

                # Test the model
                model.eval()
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for inputs, labels in val_loader:
                        inputs = inputs.unsqueeze(1).to(device)
                        labels = labels.to(device)
                        outputs = model(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    accuracy = correct / total
                    print(f"Fold {fold}, Validation Accuracy: {accuracy:.2%}")
                    fold += 1

                    # Check if the current model is the best based on validation accuracy
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_model = model.state_dict().copy()

# Load the best model
model.load_state_dict(best_model)

# Save the best model
joblib.dump(model, 'df1_BART_BiLSTM.sav')

# Evaluate the best model on the test set
model.eval()
test_dataset = TensorDataset(features_test, labels_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
with torch.no_grad():
    correct = 0
    total = 0
    predictions = []
    start_time = time.time()
    for inputs, labels in test_loader:
        inputs = inputs.unsqueeze(1).to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        predictions.extend(predicted.cpu().numpy())
    end_time = time.time()
    test_time = end_time - start_time
    test_accuracy = correct / total
    print(f"Test Accuracy: {test_accuracy:.2%}")
    print(f"Test Time: {test_time:.2f} seconds")

    # Save the predictions as a CSV file
    predictions_df = pd.DataFrame({'True Label': labels_test.cpu().numpy(), 'Predicted Label': predictions})
    predictions_df.to_csv('predictions.csv', index=False)

    # Generate classification report for the test data
    classification_rep = classification_report(labels_test.cpu().numpy(), predictions)
    print("Classification Report:")
    print(classification_rep)


############################################# ALBERT ####################################################
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.externals import joblib
import pandas as pd
import time

# Load the ALBERT features
feature_matrix_train = np.load('ALBERT_df1_train.npz')['arr_0']
feature_matrix_test = np.load('ALBERT_df1_test.npz')['arr_0']
labels_train = df1_train_class.to_numpy().flatten()
labels_test = df1_test_class.to_numpy().flatten()

# Convert the numpy arrays to PyTorch tensors
features_train = torch.tensor(feature_matrix_train, dtype=torch.float32)
features_test = torch.tensor(feature_matrix_test, dtype=torch.float32)
labels_train = torch.tensor(labels_train, dtype=torch.long)
labels_test = torch.tensor(labels_test, dtype=torch.long)

# Define the BiLSTM model
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate=0.2, activation='relu'):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_size * 2, 32)
        self.fc2 = nn.Linear(32, num_classes)
        self.activation = self.get_activation(activation)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), self.hidden_size).to(x.device)  # Initialize hidden state for bidirectional LSTM
        c0 = torch.zeros(2, x.size(0), self.hidden_size).to(x.device)  # Initialize cell state for bidirectional LSTM
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])  # Take the last time step output and apply dropout
        out = self.fc1(out)
        out = self.activation(out)
        out = self.fc2(out)
        return out

    def get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        else:
            raise ValueError("Invalid activation function.")

# Set the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the hyperparameters
batch_sizes = [64, 128, 512]
num_epochs_list = [5, 20, 100]
activations = ['sigmoid', 'relu']
learning_rate = 0.001
hidden_size = 128
dropout_rate = 0.2

# Perform 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

best_accuracy = 0.0  # Variable to store the best validation accuracy
best_model = None  # Variable to store the best model

for activation in activations:
    for batch_size in batch_sizes:
        for num_epochs in num_epochs_list:
            fold = 1
            print(f"Activation: {activation}, Batch Size: {batch_size}, Epochs: {num_epochs}")
            for train_index, val_index in kf.split(features_train):
                # Split the data into training and validation sets for hyperparameter tuning and find the best model
                train_features, val_features = features_train[train_index], features_train[val_index]
                train_labels, val_labels = labels_train[train_index], labels_train[val_index]

                # Create DataLoaders for training and validation datasets
                train_dataset = TensorDataset(train_features, train_labels)
                val_dataset = TensorDataset(val_features, val_labels)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

                # Initialize the BiLSTM model
                input_size = features_train.size(1)
                num_classes = torch.max(labels_train) + 1
                model = BiLSTM(input_size, hidden_size, num_classes, dropout_rate, activation).to(device)

                # Define the loss function and optimizer
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)

                # Train the BiLSTM model
                model.train()
                start_time = time.time()
                for epoch in range(num_epochs):
                    for inputs, labels in train_loader:
                        inputs = inputs.unsqueeze(1).to(device)  # Add an extra dimension for the LSTM input
                        labels = labels.to(device)
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                    print(f"Fold {fold}, Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
                end_time = time.time()
                train_time = end_time - start_time
                print(f"Training Time: {train_time:.2f} seconds")

                # Test the model
                model.eval()
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for inputs, labels in val_loader:
                        inputs = inputs.unsqueeze(1).to(device)
                        labels = labels.to(device)
                        outputs = model(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    accuracy = correct / total
                    print(f"Fold {fold}, Validation Accuracy: {accuracy:.2%}")
                    fold += 1

                    # Check if the current model is the best based on validation accuracy
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_model = model.state_dict().copy()

# Load the best model
model.load_state_dict(best_model)

# Save the best model
joblib.dump(model, 'df1_ALBERT_BiLSTM.sav')

# Evaluate the best model on the test set
model.eval()
test_dataset = TensorDataset(features_test, labels_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
with torch.no_grad():
    correct = 0
    total = 0
    predictions = []
    start_time = time.time()
    for inputs, labels in test_loader:
        inputs = inputs.unsqueeze(1).to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        predictions.extend(predicted.cpu().numpy())
    end_time = time.time()
    test_time = end_time - start_time
    test_accuracy = correct / total
    print(f"Test Accuracy: {test_accuracy:.2%}")
    print(f"Test Time: {test_time:.2f} seconds")

    # Save the predictions as a CSV file
    predictions_df = pd.DataFrame({'True Label': labels_test.cpu().numpy(), 'Predicted Label': predictions})
    predictions_df.to_csv('predictions.csv', index=False)

    # Generate classification report for the test data
    classification_rep = classification_report(labels_test.cpu().numpy(), predictions)
    print("Classification Report:")
    print(classification_rep)


############################################# RoBERTa ####################################################
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.externals import joblib
import pandas as pd
import time

# Load the RoBERTa features
feature_matrix_train = np.load('RoBERTa_df1_train.npz')['arr_0']
feature_matrix_test = np.load('RoBERTa_df1_test.npz')['arr_0']
labels_train = df1_train_class.to_numpy().flatten()
labels_test = df1_test_class.to_numpy().flatten()

# Convert the numpy arrays to PyTorch tensors
features_train = torch.tensor(feature_matrix_train, dtype=torch.float32)
features_test = torch.tensor(feature_matrix_test, dtype=torch.float32)
labels_train = torch.tensor(labels_train, dtype=torch.long)
labels_test = torch.tensor(labels_test, dtype=torch.long)

# Define the BiLSTM model
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate=0.2, activation='relu'):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_size * 2, 32)
        self.fc2 = nn.Linear(32, num_classes)
        self.activation = self.get_activation(activation)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), self.hidden_size).to(x.device)  # Initialize hidden state for bidirectional LSTM
        c0 = torch.zeros(2, x.size(0), self.hidden_size).to(x.device)  # Initialize cell state for bidirectional LSTM
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])  # Take the last time step output and apply dropout
        out = self.fc1(out)
        out = self.activation(out)
        out = self.fc2(out)
        return out

    def get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        else:
            raise ValueError("Invalid activation function.")

# Set the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the hyperparameters
batch_sizes = [64, 128, 512]
num_epochs_list = [5, 20, 100]
activations = ['sigmoid', 'relu']
learning_rate = 0.001
hidden_size = 128
dropout_rate = 0.2

# Perform 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

best_accuracy = 0.0  # Variable to store the best validation accuracy
best_model = None  # Variable to store the best model

for activation in activations:
    for batch_size in batch_sizes:
        for num_epochs in num_epochs_list:
            fold = 1
            print(f"Activation: {activation}, Batch Size: {batch_size}, Epochs: {num_epochs}")
            for train_index, val_index in kf.split(features_train):
                # Split the data into training and validation sets for hyperparameter tuning and find the best model
                train_features, val_features = features_train[train_index], features_train[val_index]
                train_labels, val_labels = labels_train[train_index], labels_train[val_index]

                # Create DataLoaders for training and validation datasets
                train_dataset = TensorDataset(train_features, train_labels)
                val_dataset = TensorDataset(val_features, val_labels)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

                # Initialize the BiLSTM model
                input_size = features_train.size(1)
                num_classes = torch.max(labels_train) + 1
                model = BiLSTM(input_size, hidden_size, num_classes, dropout_rate, activation).to(device)

                # Define the loss function and optimizer
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)

                # Train the BiLSTM model
                model.train()
                start_time = time.time()
                for epoch in range(num_epochs):
                    for inputs, labels in train_loader:
                        inputs = inputs.unsqueeze(1).to(device)  # Add an extra dimension for the LSTM input
                        labels = labels.to(device)
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                    print(f"Fold {fold}, Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
                end_time = time.time()
                train_time = end_time - start_time
                print(f"Training Time: {train_time:.2f} seconds")

                # Test the model
                model.eval()
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for inputs, labels in val_loader:
                        inputs = inputs.unsqueeze(1).to(device)
                        labels = labels.to(device)
                        outputs = model(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    accuracy = correct / total
                    print(f"Fold {fold}, Validation Accuracy: {accuracy:.2%}")
                    fold += 1

                    # Check if the current model is the best based on validation accuracy
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_model = model.state_dict().copy()

# Load the best model
model.load_state_dict(best_model)

# Save the best model
joblib.dump(model, 'df1_RoBERTa_BiLSTM.sav')

# Evaluate the best model on the test set
model.eval()
test_dataset = TensorDataset(features_test, labels_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
with torch.no_grad():
    correct = 0
    total = 0
    predictions = []
    start_time = time.time()
    for inputs, labels in test_loader:
        inputs = inputs.unsqueeze(1).to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        predictions.extend(predicted.cpu().numpy())
    end_time = time.time()
    test_time = end_time - start_time
    test_accuracy = correct / total
    print(f"Test Accuracy: {test_accuracy:.2%}")
    print(f"Test Time: {test_time:.2f} seconds")

    # Save the predictions as a CSV file
    predictions_df = pd.DataFrame({'True Label': labels_test.cpu().numpy(), 'Predicted Label': predictions})
    predictions_df.to_csv('predictions.csv', index=False)

    # Generate classification report for the test data
    classification_rep = classification_report(labels_test.cpu().numpy(), predictions)
    print("Classification Report:")
    print(classification_rep)


############################################# ELECTRA ####################################################
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.externals import joblib
import pandas as pd
import time

# Load the ELECTRA features
feature_matrix_train = np.load('ELECTRA_df1_train.npz')['arr_0']
feature_matrix_test = np.load('ELECTRA_df1_test.npz')['arr_0']
labels_train = df1_train_class.to_numpy().flatten()
labels_test = df1_test_class.to_numpy().flatten()

# Convert the numpy arrays to PyTorch tensors
features_train = torch.tensor(feature_matrix_train, dtype=torch.float32)
features_test = torch.tensor(feature_matrix_test, dtype=torch.float32)
labels_train = torch.tensor(labels_train, dtype=torch.long)
labels_test = torch.tensor(labels_test, dtype=torch.long)

# Define the BiLSTM model
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate=0.2, activation='relu'):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_size * 2, 32)
        self.fc2 = nn.Linear(32, num_classes)
        self.activation = self.get_activation(activation)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), self.hidden_size).to(x.device)  # Initialize hidden state for bidirectional LSTM
        c0 = torch.zeros(2, x.size(0), self.hidden_size).to(x.device)  # Initialize cell state for bidirectional LSTM
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])  # Take the last time step output and apply dropout
        out = self.fc1(out)
        out = self.activation(out)
        out = self.fc2(out)
        return out

    def get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        else:
            raise ValueError("Invalid activation function.")

# Set the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the hyperparameters
batch_sizes = [64, 128, 512]
num_epochs_list = [5, 20, 100]
activations = ['sigmoid', 'relu']
learning_rate = 0.001
hidden_size = 128
dropout_rate = 0.2

# Perform 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

best_accuracy = 0.0  # Variable to store the best validation accuracy
best_model = None  # Variable to store the best model

for activation in activations:
    for batch_size in batch_sizes:
        for num_epochs in num_epochs_list:
            fold = 1
            print(f"Activation: {activation}, Batch Size: {batch_size}, Epochs: {num_epochs}")
            for train_index, val_index in kf.split(features_train):
                # Split the data into training and validation sets for hyperparameter tuning and find the best model
                train_features, val_features = features_train[train_index], features_train[val_index]
                train_labels, val_labels = labels_train[train_index], labels_train[val_index]

                # Create DataLoaders for training and validation datasets
                train_dataset = TensorDataset(train_features, train_labels)
                val_dataset = TensorDataset(val_features, val_labels)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

                # Initialize the BiLSTM model
                input_size = features_train.size(1)
                num_classes = torch.max(labels_train) + 1
                model = BiLSTM(input_size, hidden_size, num_classes, dropout_rate, activation).to(device)

                # Define the loss function and optimizer
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)

                # Train the BiLSTM model
                model.train()
                start_time = time.time()
                for epoch in range(num_epochs):
                    for inputs, labels in train_loader:
                        inputs = inputs.unsqueeze(1).to(device)  # Add an extra dimension for the LSTM input
                        labels = labels.to(device)
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                    print(f"Fold {fold}, Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
                end_time = time.time()
                train_time = end_time - start_time
                print(f"Training Time: {train_time:.2f} seconds")

                # Test the model
                model.eval()
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for inputs, labels in val_loader:
                        inputs = inputs.unsqueeze(1).to(device)
                        labels = labels.to(device)
                        outputs = model(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    accuracy = correct / total
                    print(f"Fold {fold}, Validation Accuracy: {accuracy:.2%}")
                    fold += 1

                    # Check if the current model is the best based on validation accuracy
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_model = model.state_dict().copy()

# Load the best model
model.load_state_dict(best_model)

# Save the best model
joblib.dump(model, 'df1_ELECTRA_BiLSTM.sav')

# Evaluate the best model on the test set
model.eval()
test_dataset = TensorDataset(features_test, labels_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
with torch.no_grad():
    correct = 0
    total = 0
    predictions = []
    start_time = time.time()
    for inputs, labels in test_loader:
        inputs = inputs.unsqueeze(1).to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        predictions.extend(predicted.cpu().numpy())
    end_time = time.time()
    test_time = end_time - start_time
    test_accuracy = correct / total
    print(f"Test Accuracy: {test_accuracy:.2%}")
    print(f"Test Time: {test_time:.2f} seconds")

    # Save the predictions as a CSV file
    predictions_df = pd.DataFrame({'True Label': labels_test.cpu().numpy(), 'Predicted Label': predictions})
    predictions_df.to_csv('predictions.csv', index=False)

    # Generate classification report for the test data
    classification_rep = classification_report(labels_test.cpu().numpy(), predictions)
    print("Classification Report:")
    print(classification_rep)


############################################# XLNET ####################################################
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.externals import joblib
import pandas as pd
import time

# Load the XLNET features
feature_matrix_train = np.load('XLNET_df1_train.npz')['arr_0']
feature_matrix_test = np.load('XLNET_df1_test.npz')['arr_0']
labels_train = df1_train_class.to_numpy().flatten()
labels_test = df1_test_class.to_numpy().flatten()

# Convert the numpy arrays to PyTorch tensors
features_train = torch.tensor(feature_matrix_train, dtype=torch.float32)
features_test = torch.tensor(feature_matrix_test, dtype=torch.float32)
labels_train = torch.tensor(labels_train, dtype=torch.long)
labels_test = torch.tensor(labels_test, dtype=torch.long)

# Define the BiLSTM model
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate=0.2, activation='relu'):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_size * 2, 32)
        self.fc2 = nn.Linear(32, num_classes)
        self.activation = self.get_activation(activation)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), self.hidden_size).to(x.device)  # Initialize hidden state for bidirectional LSTM
        c0 = torch.zeros(2, x.size(0), self.hidden_size).to(x.device)  # Initialize cell state for bidirectional LSTM
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])  # Take the last time step output and apply dropout
        out = self.fc1(out)
        out = self.activation(out)
        out = self.fc2(out)
        return out

    def get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        else:
            raise ValueError("Invalid activation function.")

# Set the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the hyperparameters
batch_sizes = [64, 128, 512]
num_epochs_list = [5, 20, 100]
activations = ['sigmoid', 'relu']
learning_rate = 0.001
hidden_size = 128
dropout_rate = 0.2

# Perform 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

best_accuracy = 0.0  # Variable to store the best validation accuracy
best_model = None  # Variable to store the best model

for activation in activations:
    for batch_size in batch_sizes:
        for num_epochs in num_epochs_list:
            fold = 1
            print(f"Activation: {activation}, Batch Size: {batch_size}, Epochs: {num_epochs}")
            for train_index, val_index in kf.split(features_train):
                # Split the data into training and validation sets for hyperparameter tuning and find the best model
                train_features, val_features = features_train[train_index], features_train[val_index]
                train_labels, val_labels = labels_train[train_index], labels_train[val_index]

                # Create DataLoaders for training and validation datasets
                train_dataset = TensorDataset(train_features, train_labels)
                val_dataset = TensorDataset(val_features, val_labels)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

                # Initialize the BiLSTM model
                input_size = features_train.size(1)
                num_classes = torch.max(labels_train) + 1
                model = BiLSTM(input_size, hidden_size, num_classes, dropout_rate, activation).to(device)

                # Define the loss function and optimizer
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)

                # Train the BiLSTM model
                model.train()
                start_time = time.time()
                for epoch in range(num_epochs):
                    for inputs, labels in train_loader:
                        inputs = inputs.unsqueeze(1).to(device)  # Add an extra dimension for the LSTM input
                        labels = labels.to(device)
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                    print(f"Fold {fold}, Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
                end_time = time.time()
                train_time = end_time - start_time
                print(f"Training Time: {train_time:.2f} seconds")

                # Test the model
                model.eval()
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for inputs, labels in val_loader:
                        inputs = inputs.unsqueeze(1).to(device)
                        labels = labels.to(device)
                        outputs = model(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    accuracy = correct / total
                    print(f"Fold {fold}, Validation Accuracy: {accuracy:.2%}")
                    fold += 1

                    # Check if the current model is the best based on validation accuracy
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_model = model.state_dict().copy()

# Load the best model
model.load_state_dict(best_model)

# Save the best model
joblib.dump(model, 'df1_XLNET_BiLSTM.sav')

# Evaluate the best model on the test set
model.eval()
test_dataset = TensorDataset(features_test, labels_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
with torch.no_grad():
    correct = 0
    total = 0
    predictions = []
    start_time = time.time()
    for inputs, labels in test_loader:
        inputs = inputs.unsqueeze(1).to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        predictions.extend(predicted.cpu().numpy())
    end_time = time.time()
    test_time = end_time - start_time
    test_accuracy = correct / total
    print(f"Test Accuracy: {test_accuracy:.2%}")
    print(f"Test Time: {test_time:.2f} seconds")

    # Save the predictions as a CSV file
    predictions_df = pd.DataFrame({'True Label': labels_test.cpu().numpy(), 'Predicted Label': predictions})
    predictions_df.to_csv('predictions.csv', index=False)

    # Generate classification report for the test data
    classification_rep = classification_report(labels_test.cpu().numpy(), predictions)
    print("Classification Report:")
    print(classification_rep)
