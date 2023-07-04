import time
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from transformers import AutoModel, AutoTokenizer, AdamW
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

# Define the RoBERTa model and tokenizer
albert_model_name = "albert-base-v2"
albert = AutoModel.from_pretrained(albert_model_name)
tokenizer = AutoTokenizer.from_pretrained(albert_model_name)

# Set the maximum sequence length
max_seq_len = 512


# Tokenize and encode sequences
def tokenize_sequences(texts):
    return tokenizer.batch_encode_plus(
        texts,
        max_length=max_seq_len,
        pad_to_max_length=True,
        truncation=True,
        return_token_type_ids=False
    )


# Define the architecture for binary classification
class ALBERT_Arch(nn.Module):
    def __init__(self, albert):
        super(ALBERT_Arch, self).__init__()
        self.albert = albert
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        _, cls_hs = self.albert(sent_id, attention_mask=mask)
        x = self.fc1(cls_hs)
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
        batch = [r.to(device) for r in batch]
        sent_id, mask, labels = batch
        model.zero_grad()
        preds = model(sent_id, mask)
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


df3_tokens_train = tokenize_sequences(df3_train.tolist())
df3_tokens_test = tokenize_sequences(df3_test.tolist())

# Convert the encodings to PyTorch tensors
df3_train_seq = torch.tensor(df3_tokens_train['input_ids'])
df3_train_mask = torch.tensor(df3_tokens_train['attention_mask'])
df3_train_labels = torch.tensor(df3_train_class.tolist())

df3_test_seq = torch.tensor(df3_tokens_test['input_ids'])
df3_test_mask = torch.tensor(df3_tokens_test['attention_mask'])
df3_test_labels = torch.tensor(df3_test_class.tolist())

# Create data loader
dataset = TensorDataset(df3_train_seq, df3_train_mask, df3_train_labels)

# Set up cross-validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)

# Define lists to store performance metrics
cv_reports = []
final_reports = []

# Perform cross-validation
for fold, (train_idx, val_idx) in enumerate(skf.split(df3_train_seq, df3_train_labels)):
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
    model = ALBERT_Arch(albert)
    model = model.to(device)

    # Freeze all the parameters except the last four layers
    named_params = list(model.named_parameters())
    layers_to_freeze = ['encoder.layer.8.', 'encoder.layer.9.', 'encoder.layer.10.', 'encoder.layer.11.']
    for name, param in named_params:
        if not any(layer in name for layer in layers_to_freeze):
            param.requires_grad = False

    # Compute class weights for imbalanced dataset
    class_weights = compute_class_weight('balanced', np.unique(train_data.tensors[2]), train_data.tensors[2])
    class_weights = torch.FloatTensor(class_weights).to(device)

    # Define the loss function and optimizer
    criterion = nn.NLLLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=1e-3)

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
            for step, batch in enumerate(val_dataloader):
                batch = [r.to(device) for r in batch]
                sent_id, mask, _ = batch
                pred = model(sent_id, mask)
                val_preds.extend(torch.argmax(pred, axis=1).cpu().detach().numpy())

        # Calculate performance metrics on the validation set
        val_report = classification_report(val_data.tensors[2].tolist(), val_preds)
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
    test_dataset = TensorDataset(df3_test_seq, df3_test_mask, df3_test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=32)
    model.eval()
    test_preds = []
    with torch.no_grad():
        for batch in test_dataloader:
            batch = [r.to(device) for r in batch]
            sent_id, mask, _ = batch
            pred = model(sent_id, mask)
            test_preds.extend(torch.argmax(pred, axis=1).cpu().detach().numpy())

    # Calculate performance metrics on the test set
    test_report = classification_report(df3_test_labels.tolist(), test_preds)
    print(f"Test Report:\n{test_report}\n")
    final_reports.append(test_report)

# Print final evaluation reports
print("Final Evaluation Reports:")
for i, report in enumerate(final_reports):
    print(f"Fold {i + 1}:")
    print(report)
    print()

np.savetxt("df3_ALBERT_preds.csv", test_preds, delimiter=",")

