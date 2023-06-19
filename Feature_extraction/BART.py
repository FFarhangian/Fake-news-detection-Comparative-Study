from numpy import savez_compressed
from numpy import asarray
from numpy import load
import joblib
import torch
import numpy as np
from transformers import BartModel, BartTokenizer



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
