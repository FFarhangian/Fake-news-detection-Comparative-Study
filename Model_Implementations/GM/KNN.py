import joblib
import timeit
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, KFold, PredefinedSplit, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import tensorflow as tf
import whatlies
from whatlies.language import HFTransformersLanguage
from transformers import AutoModel, BertTokenizerFast, RobertaTokenizerFast

warnings.filterwarnings("ignore")
device = torch.device("cuda")

from transformers import pipeline
from sklearn.pipeline import Pipeline
import datasets
import sklearn
from numpy import savez_compressed, asarray
from scipy import sparse
from numpy import load
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import mean, std
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, KFold, PredefinedSplit
from sklearn.KNN import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb


# Before using transformers make sure that the data convert to list
df4_X_train = df4_train["Text"].tolist()
df4_y_train = df4_train["Label"].tolist()
df4_X_test = df4_test["Text"].tolist()
df4_y_test = df4_test["Label"].tolist()

#######################################TF
#Training and gridsearchcv
start_train = timeit.default_timer()
gridcv_KNN.fit(TF_df4_train, df4_train_class)
stop_train = timeit.default_timer()

#Prediction
start_predict = timeit.default_timer()
TF_df4_pred = gridcv_KNN.predict(TF_df4_test)
stop_predict = timeit.default_timer()

#Training Results
cv_results = gridcv_KNN.cv_results_['split0_test_score'],
gridcv_KNN.cv_results_['split1_test_score'],
gridcv_KNN.cv_results_['split2_test_score'],
gridcv_KNN.cv_results_['split3_test_score'],
gridcv_KNN.cv_results_['split4_test_score']
mean_score = np.mean(cv_results)
std_score = np.std(cv_results)

# Print the mean and standard deviation of each fold's performance
print("Mean cross-validation score: {:.2f}".format(mean_score))
print("Standard deviation of cross-validation score: {:.2f}".format(std_score))

# Print the best hyperparameters found by GridSearchCV
print("Best hyperparameters: {}".format(gridcv_KNN.best_params_))
print("Best score: {}".format(gridcv_KNN.best_score_))

#Prediction Results
print(classification_report(df4_test_class, TF_df4_pred))
print("Accuracy: ", accuracy_score(df4_test_class, TF_df4_pred))
print("f1_score: ", f1_score(df4_test_class, TF_df4_pred, average='micro'))
print("precision_score: ", precision_score(df4_test_class, TF_df4_pred, average='micro'))
print("recall_score: ", recall_score(df4_test_class, TF_df4_pred, average='micro'))

#Computational Cost
print('Train Time: ', stop_train - start_train)
print('Prediction Time : ', stop_predict - start_predict)
print('Inference Time: ', (stop_predict - start_predict)/TF_df4_test.shape[0])

#Confusion matrix
cm = confusion_matrix(df4_test_class, TF_df4_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

#Saved model and preds
np.savetxt("df4_KNN_TF_pred.csv", TF_df4_pred, delimiter=",")
joblib.dump(gridcv_KNN, 'df4_TF_KNN.sav')

#######################################TFIDF
#Training and gridsearchcv
start_train = timeit.default_timer()
gridcv_KNN.fit(TFIDF_df4_train, df4_train_class)
stop_train = timeit.default_timer()

#Prediction
start_predict = timeit.default_timer()
TFIDF_df4_pred = gridcv_KNN.predict(TFIDF_df4_test)
stop_predict = timeit.default_timer()

#Training Results
cv_results = gridcv_KNN.cv_results_['split0_test_score'],
gridcv_KNN.cv_results_['split1_test_score'],
gridcv_KNN.cv_results_['split2_test_score'],
gridcv_KNN.cv_results_['split3_test_score'],
gridcv_KNN.cv_results_['split4_test_score']
mean_score = np.mean(cv_results)
std_score = np.std(cv_results)

# Print the mean and standard deviation of each fold's performance
print("Mean cross-validation score: {:.2f}".format(mean_score))
print("Standard deviation of cross-validation score: {:.2f}".format(std_score))

# Print the best hyperparameters found by GridSearchCV
print("Best hyperparameters: {}".format(gridcv_KNN.best_params_))
print("Best score: {}".format(gridcv_KNN.best_score_))

#Prediction Results
print(classification_report(df4_test_class, TFIDF_df4_pred))
print("Accuracy: ", accuracy_score(df4_test_class, TFIDF_df4_pred))
print("f1_score: ", f1_score(df4_test_class, TFIDF_df4_pred, average='micro'))
print("precision_score: ", precision_score(df4_test_class, TFIDF_df4_pred, average='micro'))
print("recall_score: ", recall_score(df4_test_class, TFIDF_df4_pred, average='micro'))

#Computational Cost
print('Train Time: ', stop_train - start_train)
print('Prediction Time : ', stop_predict - start_predict)
print('Inference Time: ', (stop_predict - start_predict)/TFIDF_df4_test.shape[0])

#Confusion matrix
cm = confusion_matrix(df4_test_class, TFIDF_df4_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

#Saved model and preds
np.savetxt("df4_KNN_TFIDF_pred.csv", TFIDF_df4_pred, delimiter=",")
joblib.dump(gridcv_KNN, 'df4_TFIDF_KNN.sav')

#######################################W2V
#Training and gridsearchcv
start_train = timeit.default_timer()
gridcv_KNN.fit(W2V_df4_train, df4_train_class)
stop_train = timeit.default_timer()

#Prediction
start_predict = timeit.default_timer()
W2V_df4_pred = gridcv_KNN.predict(W2V_df4_test)
stop_predict = timeit.default_timer()

#Training Results
cv_results = gridcv_KNN.cv_results_['split0_test_score'],
gridcv_KNN.cv_results_['split1_test_score'],
gridcv_KNN.cv_results_['split2_test_score'],
gridcv_KNN.cv_results_['split3_test_score'],
gridcv_KNN.cv_results_['split4_test_score']
mean_score = np.mean(cv_results)
std_score = np.std(cv_results)

# Print the mean and standard deviation of each fold's performance
print("Mean cross-validation score: {:.2f}".format(mean_score))
print("Standard deviation of cross-validation score: {:.2f}".format(std_score))

# Print the best hyperparameters found by GridSearchCV
print("Best hyperparameters: {}".format(gridcv_KNN.best_params_))
print("Best score: {}".format(gridcv_KNN.best_score_))

#Prediction Results
print(classification_report(df4_test_class, W2V_df4_pred))
print("Accuracy: ", accuracy_score(df4_test_class, W2V_df4_pred))
print("f1_score: ", f1_score(df4_test_class, W2V_df4_pred, average='micro'))
print("precision_score: ", precision_score(df4_test_class, W2V_df4_pred, average='micro'))
print("recall_score: ", recall_score(df4_test_class, W2V_df4_pred, average='micro'))

#Computational Cost
print('Train Time: ', stop_train - start_train)
print('Prediction Time : ', stop_predict - start_predict)
print('Inference Time: ', (stop_predict - start_predict)/W2V_df4_test.shape[0])

#Confusion matrix
cm = confusion_matrix(df4_test_class, W2V_df4_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

#Saved model and preds
np.savetxt("df4_KNN_W2V_pred.csv", W2V_df4_pred, delimiter=",")
joblib.dump(gridcv_KNN, 'df4_W2V_KNN.sav')

#######################################Glove
#Training and gridsearchcv
start_train = timeit.default_timer()
gridcv_KNN.fit(Glove_df4_train, df4_train_class)
stop_train = timeit.default_timer()

#Prediction
start_predict = timeit.default_timer()
Glove_df4_pred = gridcv_KNN.predict(Glove_df4_test)
stop_predict = timeit.default_timer()

#Training Results
cv_results = gridcv_KNN.cv_results_['split0_test_score'],
gridcv_KNN.cv_results_['split1_test_score'],
gridcv_KNN.cv_results_['split2_test_score'],
gridcv_KNN.cv_results_['split3_test_score'],
gridcv_KNN.cv_results_['split4_test_score']
mean_score = np.mean(cv_results)
std_score = np.std(cv_results)

# Print the mean and standard deviation of each fold's performance
print("Mean cross-validation score: {:.2f}".format(mean_score))
print("Standard deviation of cross-validation score: {:.2f}".format(std_score))

# Print the best hyperparameters found by GridSearchCV
print("Best hyperparameters: {}".format(gridcv_KNN.best_params_))
print("Best score: {}".format(gridcv_KNN.best_score_))

#Prediction Results
print(classification_report(df4_test_class, Glove_df4_pred))
print("Accuracy: ", accuracy_score(df4_test_class, Glove_df4_pred))
print("f1_score: ", f1_score(df4_test_class, Glove_df4_pred, average='micro'))
print("precision_score: ", precision_score(df4_test_class, Glove_df4_pred, average='micro'))
print("recall_score: ", recall_score(df4_test_class, Glove_df4_pred, average='micro'))

#Computational Cost
print('Train Time: ', stop_train - start_train)
print('Prediction Time : ', stop_predict - start_predict)
print('Inference Time: ', (stop_predict - start_predict)/Glove_df4_test.shape[0])

#Confusion matrix
cm = confusion_matrix(df4_test_class, Glove_df4_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

#Saved model and preds
np.savetxt("df4_KNN_Glove_pred.csv", Glove_df4_pred, delimiter=",")
joblib.dump(gridcv_KNN, 'df4_Glove_KNN.sav')

#######################################Fasttext
#Training and gridsearchcv
start_train = timeit.default_timer()
gridcv_KNN.fit(Fasttext_df4_train, df4_train_class)
stop_train = timeit.default_timer()

#Prediction
start_predict = timeit.default_timer()
Fasttext_df4_pred = gridcv_KNN.predict(Fasttext_df4_test)
stop_predict = timeit.default_timer()

#Training Results
cv_results = gridcv_KNN.cv_results_['split0_test_score'],
gridcv_KNN.cv_results_['split1_test_score'],
gridcv_KNN.cv_results_['split2_test_score'],
gridcv_KNN.cv_results_['split3_test_score'],
gridcv_KNN.cv_results_['split4_test_score']
mean_score = np.mean(cv_results)
std_score = np.std(cv_results)

# Print the mean and standard deviation of each fold's performance
print("Mean cross-validation score: {:.2f}".format(mean_score))
print("Standard deviation of cross-validation score: {:.2f}".format(std_score))

# Print the best hyperparameters found by GridSearchCV
print("Best hyperparameters: {}".format(gridcv_KNN.best_params_))
print("Best score: {}".format(gridcv_KNN.best_score_))

#Prediction Results
print(classification_report(df4_test_class, Fasttext_df4_pred))
print("Accuracy: ", accuracy_score(df4_test_class, Fasttext_df4_pred))
print("f1_score: ", f1_score(df4_test_class, Fasttext_df4_pred, average='micro'))
print("precision_score: ", precision_score(df4_test_class, Fasttext_df4_pred, average='micro'))
print("recall_score: ", recall_score(df4_test_class, Fasttext_df4_pred, average='micro'))

#Computational Cost
print('Train Time: ', stop_train - start_train)
print('Prediction Time : ', stop_predict - start_predict)
print('Inference Time: ', (stop_predict - start_predict)/Fasttext_df4_test.shape[0])

#Confusion matrix
cm = confusion_matrix(df4_test_class, Fasttext_df4_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

#Saved model and preds
np.savetxt("df4_KNN_Fasttext_pred.csv", Fasttext_df4_pred, delimiter=",")
joblib.dump(gridcv_KNN, 'df4_Fasttext_KNN.sav')

#######################################ELMO

#Training and gridsearchcv
start_train = timeit.default_timer()
gridcv_KNN.fit(ELMO_df4_train, df4_train_class)
stop_train = timeit.default_timer()

#Prediction
start_predict = timeit.default_timer()
ELMO_df4_pred = gridcv_KNN.predict(ELMO_df4_test)
stop_predict = timeit.default_timer()

#Training Results
cv_results = gridcv_KNN.cv_results_['split0_test_score'],
gridcv_KNN.cv_results_['split1_test_score'],
gridcv_KNN.cv_results_['split2_test_score'],
gridcv_KNN.cv_results_['split3_test_score'],
gridcv_KNN.cv_results_['split4_test_score']
mean_score = np.mean(cv_results)
std_score = np.std(cv_results)

# Print the mean and standard deviation of each fold's performance
print("Mean cross-validation score: {:.2f}".format(mean_score))
print("Standard deviation of cross-validation score: {:.2f}".format(std_score))

# Print the best hyperparameters found by GridSearchCV
print("Best hyperparameters: {}".format(gridcv_KNN.best_params_))
print("Best score: {}".format(gridcv_KNN.best_score_))

#Prediction Results
print(classification_report(df4_test_class, ELMO_df4_pred))
print("Accuracy: ", accuracy_score(df4_test_class, ELMO_df4_pred))
print("f1_score: ", f1_score(df4_test_class, ELMO_df4_pred, average='micro'))
print("precision_score: ", precision_score(df4_test_class, ELMO_df4_pred, average='micro'))
print("recall_score: ", recall_score(df4_test_class, ELMO_df4_pred, average='micro'))

#Computational Cost
print('Train Time: ', stop_train - start_train)
print('Prediction Time : ', stop_predict - start_predict)
print('Inference Time: ', (stop_predict - start_predict)/ELMO_df4_test.shape[0])

#Confusion matrix
cm = confusion_matrix(df4_test_class, ELMO_df4_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

#Saved model and preds
np.savetxt("df4_ELMO_KNN_pred.csv", ELMO_df4_pred, delimiter=",")
joblib.dump(gridcv_KNN, 'df4_ELMO_KNN.sav')

#######################################BERT

#Training and gridsearchcv
start_train = timeit.default_timer()
df4_BERT_KNN = Pipeline([
    ("embedding", HFTransformersLanguage("bert-base-uncased")),
    ("model", KNeighborsClassifier())
])
gridcv_KNN = GridSearchCV(df4_BERT_KNN, param_grid = grid_KNN, scoring='f1_micro', cv= 5, n_jobs = -1, verbose = 2, refit = True)
gridcv_KNN.fit(df4_X_train, df4_y_train)
stop_train = timeit.default_timer()

# Prediction
start_predict = timeit.default_timer()
df4_BERT_KNN_pred = gridcv_KNN.predict(df4_X_test)
stop_predict = timeit.default_timer()

# Training Results
cv_results = gridcv_KNN.cv_results_['split0_test_score'],
gridcv_KNN.cv_results_['split1_test_score'],
gridcv_KNN.cv_results_['split2_test_score'],
gridcv_KNN.cv_results_['split3_test_score'],
gridcv_KNN.cv_results_['split4_test_score']
mean_score = np.mean(cv_results)
std_score = np.std(cv_results)

# Print the mean and standard deviation of each fold's performance
print("Mean cross-validation score: {:.2f}".format(mean_score))
print("Standard deviation of cross-validation score: {:.2f}".format(std_score))

# Print the best hyperparameters found by GridSearchCV
print("Best hyperparameters: {}".format(gridcv_KNN.best_params_))
print("Best score: {}".format(gridcv_KNN.best_score_))

# Prediction Results
print(classification_report(df4_y_test, df4_BERT_KNN_pred))
print("Accuracy: ", accuracy_score(df4_y_test, df4_BERT_KNN_pred))
print("f1_score: ", f1_score(df4_y_test, df4_BERT_KNN_pred, average='micro'))
print("precision_score: ", precision_score(df4_y_test, df4_BERT_KNN_pred, average='micro'))
print("recall_score: ", recall_score(df4_y_test, df4_BERT_KNN_pred, average='micro'))

# Computational Cost
print('Train Time: ', stop_train - start_train, 'Seconds')
print('Prediction Time : ', stop_predict - start_predict, 'Seconds')
print('Inference Time: ', (stop_predict - start_predict) * 1000 / len(df4_X_test), 'Milliseconds')

# Confusion matrix
cm = confusion_matrix(df4_y_test, df4_BERT_KNN_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Saved model and preds
np.savetxt("df4_BERT_KNN_pred.csv", df4_BERT_KNN_pred, delimiter=",")
joblib.dump(gridcv_KNN, 'df4_BERT_KNN.sav')

#######################################DistilBERT

#Training and gridsearchcv
start_train = timeit.default_timer()
df4_DistilBERT_KNN = Pipeline([
    ("embedding", HFTransformersLanguage("distilbert-base-uncased")),
    ("model", KNeighborsClassifier())
])
gridcv_KNN = GridSearchCV(df4_DistilBERT_KNN, param_grid = grid_KNN, scoring='f1_micro', cv= 5, n_jobs = -1, verbose = 2, refit = True)
gridcv_KNN.fit(df4_X_train, df4_y_train)
stop_train = timeit.default_timer()

# Prediction
start_predict = timeit.default_timer()
df4_DistilBERT_KNN_pred = gridcv_KNN.predict(df4_X_test)
stop_predict = timeit.default_timer()

# Training Results
cv_results = gridcv_KNN.cv_results_['split0_test_score'],
gridcv_KNN.cv_results_['split1_test_score'],
gridcv_KNN.cv_results_['split2_test_score'],
gridcv_KNN.cv_results_['split3_test_score'],
gridcv_KNN.cv_results_['split4_test_score']
mean_score = np.mean(cv_results)
std_score = np.std(cv_results)

# Print the mean and standard deviation of each fold's performance
print("Mean cross-validation score: {:.2f}".format(mean_score))
print("Standard deviation of cross-validation score: {:.2f}".format(std_score))

# Print the best hyperparameters found by GridSearchCV
print("Best hyperparameters: {}".format(gridcv_KNN.best_params_))
print("Best score: {}".format(gridcv_KNN.best_score_))

# Prediction Results
print(classification_report(df4_y_test, df4_DistilBERT_KNN_pred))
print("Accuracy: ", accuracy_score(df4_y_test, df4_DistilBERT_KNN_pred))
print("f1_score: ", f1_score(df4_y_test, df4_DistilBERT_KNN_pred, average='micro'))
print("precision_score: ", precision_score(df4_y_test, df4_DistilBERT_KNN_pred, average='micro'))
print("recall_score: ", recall_score(df4_y_test, df4_DistilBERT_KNN_pred, average='micro'))

# Computational Cost
print('Train Time: ', stop_train - start_train, 'Seconds')
print('Prediction Time : ', stop_predict - start_predict, 'Seconds')
print('Inference Time: ', (stop_predict - start_predict) * 1000 / len(df4_X_test), 'Milliseconds')

# Confusion matrix
cm = confusion_matrix(df4_y_test, df4_DistilBERT_KNN_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Saved model and preds
np.savetxt("df4_DistilBERT_KNN_pred.csv", df4_DistilBERT_KNN_pred, delimiter=",")
joblib.dump(gridcv_KNN, 'df4_DistilBERT_KNN.sav')

#######################################ALBERT

#Training and gridsearchcv
start_train = timeit.default_timer()
df4_ALBERT_KNN = Pipeline([
    ("embedding", HFTransformersLanguage("albert-base-v2")),
    ("model", KNeighborsClassifier())
])
gridcv_KNN = GridSearchCV(df4_ALBERT_KNN, param_grid = grid_KNN, scoring='f1_micro', cv= 5, n_jobs = -1, verbose = 2, refit = True)
gridcv_KNN.fit(df4_X_train, df4_y_train)
stop_train = timeit.default_timer()

# Prediction
start_predict = timeit.default_timer()
df4_ALBERT_KNN_pred = gridcv_KNN.predict(df4_X_test)
stop_predict = timeit.default_timer()

# Training Results
cv_results = gridcv_KNN.cv_results_['split0_test_score'],
gridcv_KNN.cv_results_['split1_test_score'],
gridcv_KNN.cv_results_['split2_test_score'],
gridcv_KNN.cv_results_['split3_test_score'],
gridcv_KNN.cv_results_['split4_test_score']
mean_score = np.mean(cv_results)
std_score = np.std(cv_results)

# Print the mean and standard deviation of each fold's performance
print("Mean cross-validation score: {:.2f}".format(mean_score))
print("Standard deviation of cross-validation score: {:.2f}".format(std_score))

# Print the best hyperparameters found by GridSearchCV
print("Best hyperparameters: {}".format(gridcv_KNN.best_params_))
print("Best score: {}".format(gridcv_KNN.best_score_))

# Prediction Results
print(classification_report(df4_y_test, df4_ALBERT_KNN_pred))
print("Accuracy: ", accuracy_score(df4_y_test, df4_ALBERT_KNN_pred))
print("f1_score: ", f1_score(df4_y_test, df4_ALBERT_KNN_pred, average='micro'))
print("precision_score: ", precision_score(df4_y_test, df4_ALBERT_KNN_pred, average='micro'))
print("recall_score: ", recall_score(df4_y_test, df4_ALBERT_KNN_pred, average='micro'))

# Computational Cost
print('Train Time: ', stop_train - start_train, 'Seconds')
print('Prediction Time : ', stop_predict - start_predict, 'Seconds')
print('Inference Time: ', (stop_predict - start_predict) * 1000 / len(df4_X_test), 'Milliseconds')

# Confusion matrix
cm = confusion_matrix(df4_y_test, df4_ALBERT_KNN_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Saved model and preds
np.savetxt("df4_ALBERT_KNN_pred.csv", df4_ALBERT_KNN_pred, delimiter=",")
joblib.dump(gridcv_KNN, 'df4_ALBERT_KNN.sav')

#######################################BART

#Training and gridsearchcv
start_train = timeit.default_timer()
df4_BART_KNN = Pipeline([
    ("embedding", HFTransformersLanguage("facebook/bart-base")),
    ("model", KNeighborsClassifier())
])
gridcv_KNN = GridSearchCV(df4_BART_KNN, param_grid = grid_KNN, scoring='f1_micro', cv= 5, n_jobs = -1, verbose = 2, refit = True)
gridcv_KNN.fit(df4_X_train, df4_y_train)
stop_train = timeit.default_timer()

# Prediction
start_predict = timeit.default_timer()
df4_BART_KNN_pred = gridcv_KNN.predict(df4_X_test)
stop_predict = timeit.default_timer()

# Training Results
cv_results = gridcv_KNN.cv_results_['split0_test_score'],
gridcv_KNN.cv_results_['split1_test_score'],
gridcv_KNN.cv_results_['split2_test_score'],
gridcv_KNN.cv_results_['split3_test_score'],
gridcv_KNN.cv_results_['split4_test_score']
mean_score = np.mean(cv_results)
std_score = np.std(cv_results)

# Print the mean and standard deviation of each fold's performance
print("Mean cross-validation score: {:.2f}".format(mean_score))
print("Standard deviation of cross-validation score: {:.2f}".format(std_score))

# Print the best hyperparameters found by GridSearchCV
print("Best hyperparameters: {}".format(gridcv_KNN.best_params_))
print("Best score: {}".format(gridcv_KNN.best_score_))

# Prediction Results
print(classification_report(df4_y_test, df4_BART_KNN_pred))
print("Accuracy: ", accuracy_score(df4_y_test, df4_BART_KNN_pred))
print("f1_score: ", f1_score(df4_y_test, df4_BART_KNN_pred, average='micro'))
print("precision_score: ", precision_score(df4_y_test, df4_BART_KNN_pred, average='micro'))
print("recall_score: ", recall_score(df4_y_test, df4_BART_KNN_pred, average='micro'))

# Computational Cost
print('Train Time: ', stop_train - start_train, 'Seconds')
print('Prediction Time : ', stop_predict - start_predict, 'Seconds')
print('Inference Time: ', (stop_predict - start_predict) * 1000 / len(df4_X_test), 'Milliseconds')

# Confusion matrix
cm = confusion_matrix(df4_y_test, df4_BART_KNN_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Saved model and preds
np.savetxt("df4_BART_KNN_pred.csv", df4_BART_KNN_pred, delimiter=",")
joblib.dump(gridcv_KNN, 'df4_BART_KNN.sav')

#######################################RoBERTa

#Training and gridsearchcv
start_train = timeit.default_timer()
df4_RoBERTa_KNN = Pipeline([
    ("embedding", HFTransformersLanguage("roberta-base")),
    ("model", KNeighborsClassifier())
])
gridcv_KNN = GridSearchCV(df4_RoBERTa_KNN, param_grid = grid_KNN, scoring='f1_micro', cv= 5, n_jobs = -1, verbose = 2, refit = True)
gridcv_KNN.fit(df4_X_train, df4_y_train)
stop_train = timeit.default_timer()

# Prediction
start_predict = timeit.default_timer()
df4_RoBERTa_KNN_pred = gridcv_KNN.predict(df4_X_test)
stop_predict = timeit.default_timer()

# Training Results
cv_results = gridcv_KNN.cv_results_['split0_test_score'],
gridcv_KNN.cv_results_['split1_test_score'],
gridcv_KNN.cv_results_['split2_test_score'],
gridcv_KNN.cv_results_['split3_test_score'],
gridcv_KNN.cv_results_['split4_test_score']
mean_score = np.mean(cv_results)
std_score = np.std(cv_results)

# Print the mean and standard deviation of each fold's performance
print("Mean cross-validation score: {:.2f}".format(mean_score))
print("Standard deviation of cross-validation score: {:.2f}".format(std_score))

# Print the best hyperparameters found by GridSearchCV
print("Best hyperparameters: {}".format(gridcv_KNN.best_params_))
print("Best score: {}".format(gridcv_KNN.best_score_))

# Prediction Results
print(classification_report(df4_y_test, df4_RoBERTa_KNN_pred))
print("Accuracy: ", accuracy_score(df4_y_test, df4_RoBERTa_KNN_pred))
print("f1_score: ", f1_score(df4_y_test, df4_RoBERTa_KNN_pred, average='micro'))
print("precision_score: ", precision_score(df4_y_test, df4_RoBERTa_KNN_pred, average='micro'))
print("recall_score: ", recall_score(df4_y_test, df4_RoBERTa_KNN_pred, average='micro'))

# Computational Cost
print('Train Time: ', stop_train - start_train, 'Seconds')
print('Prediction Time : ', stop_predict - start_predict, 'Seconds')
print('Inference Time: ', (stop_predict - start_predict) * 1000 / len(df4_X_test), 'Milliseconds')

# Confusion matrix
cm = confusion_matrix(df4_y_test, df4_RoBERTa_KNN_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Saved model and preds
np.savetxt("df4_RoBERTa_KNN_pred.csv", df4_RoBERTa_KNN_pred, delimiter=",")
joblib.dump(gridcv_KNN, 'df4_RoBERTa_KNN.sav')

#######################################ELECTRA

#Training and gridsearchcv
start_train = timeit.default_timer()
df4_ELECTRA_KNN = Pipeline([
    ("embedding", HFTransformersLanguage("google/electra-small-discriminator")),
    ("model", KNeighborsClassifier())
])
gridcv_KNN = GridSearchCV(df4_ELECTRA_KNN, param_grid = grid_KNN, scoring='f1_micro', cv= 5, n_jobs = -1, verbose = 2, refit = True)
gridcv_KNN.fit(df4_X_train, df4_y_train)
stop_train = timeit.default_timer()

# Prediction
start_predict = timeit.default_timer()
df4_ELECTRA_KNN_pred = gridcv_KNN.predict(df4_X_test)
stop_predict = timeit.default_timer()

# Training Results
cv_results = gridcv_KNN.cv_results_['split0_test_score'],
gridcv_KNN.cv_results_['split1_test_score'],
gridcv_KNN.cv_results_['split2_test_score'],
gridcv_KNN.cv_results_['split3_test_score'],
gridcv_KNN.cv_results_['split4_test_score']
mean_score = np.mean(cv_results)
std_score = np.std(cv_results)

# Print the mean and standard deviation of each fold's performance
print("Mean cross-validation score: {:.2f}".format(mean_score))
print("Standard deviation of cross-validation score: {:.2f}".format(std_score))

# Print the best hyperparameters found by GridSearchCV
print("Best hyperparameters: {}".format(gridcv_KNN.best_params_))
print("Best score: {}".format(gridcv_KNN.best_score_))

# Prediction Results
print(classification_report(df4_y_test, df4_ELECTRA_KNN_pred))
print("Accuracy: ", accuracy_score(df4_y_test, df4_ELECTRA_KNN_pred))
print("f1_score: ", f1_score(df4_y_test, df4_ELECTRA_KNN_pred, average='micro'))
print("precision_score: ", precision_score(df4_y_test, df4_ELECTRA_KNN_pred, average='micro'))
print("recall_score: ", recall_score(df4_y_test, df4_ELECTRA_KNN_pred, average='micro'))

# Computational Cost
print('Train Time: ', stop_train - start_train, 'Seconds')
print('Prediction Time : ', stop_predict - start_predict, 'Seconds')
print('Inference Time: ', (stop_predict - start_predict) * 1000 / len(df4_X_test), 'Milliseconds')

# Confusion matrix
cm = confusion_matrix(df4_y_test, df4_ELECTRA_KNN_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Saved model and preds
np.savetxt("df4_ELECTRA_KNN_pred.csv", df4_ELECTRA_KNN_pred, delimiter=",")
joblib.dump(gridcv_KNN, 'df4_ELECTRA_KNN.sav')

#######################################XLNet

#Training and gridsearchcv
start_train = timeit.default_timer()
df4_XLNET_KNN = Pipeline([
    ("embedding", HFTransformersLanguage("xlnet-base-cased")),
    ("model", KNeighborsClassifier())
])
gridcv_KNN = GridSearchCV(df4_XLNET_KNN, param_grid = grid_KNN, scoring='f1_micro', cv= 5, n_jobs = -1, verbose = 2, refit = True)
gridcv_KNN.fit(df4_X_train, df4_y_train)
stop_train = timeit.default_timer()

# Prediction
start_predict = timeit.default_timer()
df4_XLNET_KNN_pred = gridcv_KNN.predict(df4_X_test)
stop_predict = timeit.default_timer()

# Training Results
cv_results = gridcv_KNN.cv_results_['split0_test_score'],
gridcv_KNN.cv_results_['split1_test_score'],
gridcv_KNN.cv_results_['split2_test_score'],
gridcv_KNN.cv_results_['split3_test_score'],
gridcv_KNN.cv_results_['split4_test_score']
mean_score = np.mean(cv_results)
std_score = np.std(cv_results)

# Print the mean and standard deviation of each fold's performance
print("Mean cross-validation score: {:.2f}".format(mean_score))
print("Standard deviation of cross-validation score: {:.2f}".format(std_score))

# Print the best hyperparameters found by GridSearchCV
print("Best hyperparameters: {}".format(gridcv_KNN.best_params_))
print("Best score: {}".format(gridcv_KNN.best_score_))

# Prediction Results
print(classification_report(df4_y_test, df4_XLNET_KNN_pred))
print("Accuracy: ", accuracy_score(df4_y_test, df4_XLNET_KNN_pred))
print("f1_score: ", f1_score(df4_y_test, df4_XLNET_KNN_pred, average='micro'))
print("precision_score: ", precision_score(df4_y_test, df4_XLNET_KNN_pred, average='micro'))
print("recall_score: ", recall_score(df4_y_test, df4_XLNET_KNN_pred, average='micro'))

# Computational Cost
print('Train Time: ', stop_train - start_train, 'Seconds')
print('Prediction Time : ', stop_predict - start_predict, 'Seconds')
print('Inference Time: ', (stop_predict - start_predict) * 1000 / len(df4_X_test), 'Milliseconds')

# Confusion matrix
cm = confusion_matrix(df4_y_test, df4_XLNET_KNN_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Saved model and preds
np.savetxt("df4_XLNET_KNN_pred.csv", df4_XLNET_KNN_pred, delimiter=",")
joblib.dump(gridcv_KNN, 'df4_XLNET_KNN.sav')

