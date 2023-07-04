import time
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from allennlp.modules.elmo import Elmo, batch_to_ids
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

# Set the seed for reproducibility
torch.manual_seed(2023)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the ELMO model
elmo_model_name = "elmo"  # Use the default ELMO model
elmo = Elmo(elmo_model_name, options_file="elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json",
            weight_file="elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5", num_output_representations=1)

# Set the maximum sequence length
max_seq_len = 512


# Tokenize and encode sequences
def tokenize_sequences(texts):
    tokens = [text.split() for text in texts]
    return batch_to_ids(tokens)


# Define the architecture for binary classification
class ELMO_Arch(nn.Module):
    def __init__(self, elmo):
        super(ELMO_Arch, self).__init__()
        self.elmo = elmo
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id):
        embeddings = self.elmo(sent_id)['elmo_representations'][0]
        x = torch.mean(embeddings, dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


# Define the training loop
def train(model, train_dataloader):
    model.train()
    total_loss = 0
    total_preds = []
    for step, batch in enumerate(train_dataloader):
        batch = batch.to(device)
        labels = batch[1]
        model.zero_grad()
        preds = model(batch[0])
        loss = criterion(preds, labels)
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        preds = preds.detach().cpu().numpy()
        total_preds.append(preds)
    avg_loss = total_loss / len(train_dataloader)
    total_preds = np.concatenate(total_preds, axis=0)
    return avg_loss, total_preds


df2_tokens_train = tokenize_sequences(df2_train.tolist())
df2_tokens_test = tokenize_sequences(df2_test.tolist())

# Convert the encodings to PyTorch tensors
df2_train_seq = torch.tensor(df2_tokens_train['token_ids'])
df2_train_labels = torch.tensor(df2_train_class.tolist())

df2_test_seq = torch.tensor(df2_tokens_test['token_ids'])
df2_test_labels = torch.tensor(df2_test_class.tolist())

# Create data loader
dataset = TensorDataset(df2_train_seq, df2_train_labels)

# Set up cross-validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)

# Define lists to store performance metrics
cv_reports = []
final_reports = []

# Perform cross-validation
for fold, (train_idx, val_idx) in enumerate(skf.split(df2_train_seq, df2_train_labels)):
    print(f"Fold: {fold + 1}/{n_splits}")

    # Split data into train and validation sets
    train_data = dataset[train_idx]
    val_data = dataset[val_idx]

    # Create data loaders for train and validation sets
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=32)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=32)

    # Create the model architecture
    model = ELMO_Arch(elmo)
    model = model.to(device)

    # Compute class weights for imbalanced dataset
    class_weights = compute_class_weight('balanced', np.unique(train_data.tensors[1]), train_data.tensors[1])
    class_weights = torch.FloatTensor(class_weights).to(device)

    # Define the loss function and optimizer
    criterion = nn.NLLLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Train the model
    start_time = time.time()
    best_val_loss = float('inf')
    for epoch in range(10):
        avg_loss, _ = train(model, train_dataloader)
        print(f"Epoch {epoch + 1}/10: Average Loss = {avg_loss:.4f}")
        # Evaluate on the validation set
        model.eval()
        val_preds = []
        with torch.no_grad():
            for batch in val_dataloader:
                batch = batch.to(device)
                pred = model(batch)
                val_preds.extend(torch.argmax(pred, axis=1).cpu().detach().numpy())

        # Calculate performance metrics on the validation set
        val_report = classification_report(val_data.tensors[1].tolist(), val_preds)
        print(f"Validation Report:\n{val_report}\n")
        cv_reports.append(val_report)

        # Save the best model based on validation loss
        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            torch.save(model.state_dict(), f"best_model_fold{fold + 1}.pth")

    end_time = time.time()
    train_time = end_time - start_time

    # Load the best model for the fold
    model.load_state_dict(torch.load(f"best_model_fold{fold + 1}.pth"))

    # Predict on the test set
    test_dataset = TensorDataset(df2_test_seq, df2_test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=32)
    model.eval()
    test_preds = []
    with torch.no_grad():
        for batch in test_dataloader:
            batch = batch.to(device)
            pred = model(batch)
            test_preds.extend(torch.argmax(pred, axis=1).cpu().detach().numpy())

    # Calculate performance metrics on the test set
    test_report = classification_report(df2_test_labels.tolist(), test_preds)
    print(f"Test Report:\n{test_report}\n")
    final_reports.append(test_report)

# Print final evaluation reports
print("Final Evaluation Reports:")
for i, report in enumerate(final_reports):
    print(f"Fold {i + 1}:")
    print(report)
    print()

np.savetxt("df2_ELMO_preds.csv", test_preds, delimiter=",")

