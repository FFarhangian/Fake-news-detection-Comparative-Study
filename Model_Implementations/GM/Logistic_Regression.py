import joblib
import timeit
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, KFold, PredefinedSplit, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
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
from sklearn.svm import SVC
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
gridcv_LR.fit(TF_df4_train, df4_train_class)
stop_train = timeit.default_timer()

#Prediction
start_predict = timeit.default_timer()
TF_df4_pred = gridcv_LR.predict(TF_df4_test)
stop_predict = timeit.default_timer()

#Training Results
cv_results = gridcv_LR.cv_results_['split0_test_score'],
gridcv_LR.cv_results_['split1_test_score'],
gridcv_LR.cv_results_['split2_test_score'],
gridcv_LR.cv_results_['split3_test_score'],
gridcv_LR.cv_results_['split4_test_score']
mean_score = np.mean(cv_results)
std_score = np.std(cv_results)

# Print the mean and standard deviation of each fold's performance
print("Mean cross-validation score: {:.2f}".format(mean_score))
print("Standard deviation of cross-validation score: {:.2f}".format(std_score))

# Print the best hyperparameters found by GridSearchCV
print("Best hyperparameters: {}".format(gridcv_LR.best_params_))
print("Best score: {}".format(gridcv_LR.best_score_))

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
np.savetxt("df4_LR_TF_pred.csv", TF_df4_pred, delimiter=",")
joblib.dump(gridcv_LR, 'df4_TF_LR.sav')

#######################################TFIDF
#Training and gridsearchcv
start_train = timeit.default_timer()
gridcv_LR.fit(TFIDF_df4_train, df4_train_class)
stop_train = timeit.default_timer()

#Prediction
start_predict = timeit.default_timer()
TFIDF_df4_pred = gridcv_LR.predict(TFIDF_df4_test)
stop_predict = timeit.default_timer()

#Training Results
cv_results = gridcv_LR.cv_results_['split0_test_score'],
gridcv_LR.cv_results_['split1_test_score'],
gridcv_LR.cv_results_['split2_test_score'],
gridcv_LR.cv_results_['split3_test_score'],
gridcv_LR.cv_results_['split4_test_score']
mean_score = np.mean(cv_results)
std_score = np.std(cv_results)

# Print the mean and standard deviation of each fold's performance
print("Mean cross-validation score: {:.2f}".format(mean_score))
print("Standard deviation of cross-validation score: {:.2f}".format(std_score))

# Print the best hyperparameters found by GridSearchCV
print("Best hyperparameters: {}".format(gridcv_LR.best_params_))
print("Best score: {}".format(gridcv_LR.best_score_))

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
np.savetxt("df4_LR_TFIDF_pred.csv", TFIDF_df4_pred, delimiter=",")
joblib.dump(gridcv_LR, 'df4_TFIDF_LR.sav')

#######################################W2V
#Training and gridsearchcv
start_train = timeit.default_timer()
gridcv_LR.fit(W2V_df4_train, df4_train_class)
stop_train = timeit.default_timer()

#Prediction
start_predict = timeit.default_timer()
W2V_df4_pred = gridcv_LR.predict(W2V_df4_test)
stop_predict = timeit.default_timer()

#Training Results
cv_results = gridcv_LR.cv_results_['split0_test_score'],
gridcv_LR.cv_results_['split1_test_score'],
gridcv_LR.cv_results_['split2_test_score'],
gridcv_LR.cv_results_['split3_test_score'],
gridcv_LR.cv_results_['split4_test_score']
mean_score = np.mean(cv_results)
std_score = np.std(cv_results)

# Print the mean and standard deviation of each fold's performance
print("Mean cross-validation score: {:.2f}".format(mean_score))
print("Standard deviation of cross-validation score: {:.2f}".format(std_score))

# Print the best hyperparameters found by GridSearchCV
print("Best hyperparameters: {}".format(gridcv_LR.best_params_))
print("Best score: {}".format(gridcv_LR.best_score_))

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
np.savetxt("df4_LR_W2V_pred.csv", W2V_df4_pred, delimiter=",")
joblib.dump(gridcv_LR, 'df4_W2V_LR.sav')

#######################################Glove
#Training and gridsearchcv
start_train = timeit.default_timer()
gridcv_LR.fit(Glove_df4_train, df4_train_class)
stop_train = timeit.default_timer()

#Prediction
start_predict = timeit.default_timer()
Glove_df4_pred = gridcv_LR.predict(Glove_df4_test)
stop_predict = timeit.default_timer()

#Training Results
cv_results = gridcv_LR.cv_results_['split0_test_score'],
gridcv_LR.cv_results_['split1_test_score'],
gridcv_LR.cv_results_['split2_test_score'],
gridcv_LR.cv_results_['split3_test_score'],
gridcv_LR.cv_results_['split4_test_score']
mean_score = np.mean(cv_results)
std_score = np.std(cv_results)

# Print the mean and standard deviation of each fold's performance
print("Mean cross-validation score: {:.2f}".format(mean_score))
print("Standard deviation of cross-validation score: {:.2f}".format(std_score))

# Print the best hyperparameters found by GridSearchCV
print("Best hyperparameters: {}".format(gridcv_LR.best_params_))
print("Best score: {}".format(gridcv_LR.best_score_))

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
np.savetxt("df4_LR_Glove_pred.csv", Glove_df4_pred, delimiter=",")
joblib.dump(gridcv_LR, 'df4_Glove_LR.sav')

#######################################Fasttext
#Training and gridsearchcv
start_train = timeit.default_timer()
gridcv_LR.fit(Fasttext_df4_train, df4_train_class)
stop_train = timeit.default_timer()

#Prediction
start_predict = timeit.default_timer()
Fasttext_df4_pred = gridcv_LR.predict(Fasttext_df4_test)
stop_predict = timeit.default_timer()

#Training Results
cv_results = gridcv_LR.cv_results_['split0_test_score'],
gridcv_LR.cv_results_['split1_test_score'],
gridcv_LR.cv_results_['split2_test_score'],
gridcv_LR.cv_results_['split3_test_score'],
gridcv_LR.cv_results_['split4_test_score']
mean_score = np.mean(cv_results)
std_score = np.std(cv_results)

# Print the mean and standard deviation of each fold's performance
print("Mean cross-validation score: {:.2f}".format(mean_score))
print("Standard deviation of cross-validation score: {:.2f}".format(std_score))

# Print the best hyperparameters found by GridSearchCV
print("Best hyperparameters: {}".format(gridcv_LR.best_params_))
print("Best score: {}".format(gridcv_LR.best_score_))

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
np.savetxt("df4_LR_Fasttext_pred.csv", Fasttext_df4_pred, delimiter=",")
joblib.dump(gridcv_LR, 'df4_Fasttext_LR.sav')

#######################################ELMO

#Training and gridsearchcv
start_train = timeit.default_timer()
gridcv_LR.fit(ELMO_df4_train, df4_train_class)
stop_train = timeit.default_timer()

#Prediction
start_predict = timeit.default_timer()
ELMO_df4_pred = gridcv_LR.predict(ELMO_df4_test)
stop_predict = timeit.default_timer()

#Training Results
cv_results = gridcv_LR.cv_results_['split0_test_score'],
gridcv_LR.cv_results_['split1_test_score'],
gridcv_LR.cv_results_['split2_test_score'],
gridcv_LR.cv_results_['split3_test_score'],
gridcv_LR.cv_results_['split4_test_score']
mean_score = np.mean(cv_results)
std_score = np.std(cv_results)

# Print the mean and standard deviation of each fold's performance
print("Mean cross-validation score: {:.2f}".format(mean_score))
print("Standard deviation of cross-validation score: {:.2f}".format(std_score))

# Print the best hyperparameters found by GridSearchCV
print("Best hyperparameters: {}".format(gridcv_LR.best_params_))
print("Best score: {}".format(gridcv_LR.best_score_))

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
np.savetxt("df4_ELMO_LR_pred.csv", ELMO_df4_pred, delimiter=",")
joblib.dump(gridcv_LR, 'df4_ELMO_LR.sav')

#######################################BERT

#Training and gridsearchcv
start_train = timeit.default_timer()
df4_BERT_LR = Pipeline([
    ("embedding", HFTransformersLanguage("bert-base-uncased")),
    ("model", LogisticRegression())
])
gridcv_LR = GridSearchCV(df4_BERT_LR, param_grid = grid_LR, scoring='f1_micro', cv= 5, n_jobs = -1, verbose = 2, refit = True)
gridcv_LR.fit(df4_X_train, df4_y_train)
stop_train = timeit.default_timer()

# Prediction
start_predict = timeit.default_timer()
df4_BERT_LR_pred = gridcv_LR.predict(df4_X_test)
stop_predict = timeit.default_timer()

# Training Results
cv_results = gridcv_LR.cv_results_['split0_test_score'],
gridcv_LR.cv_results_['split1_test_score'],
gridcv_LR.cv_results_['split2_test_score'],
gridcv_LR.cv_results_['split3_test_score'],
gridcv_LR.cv_results_['split4_test_score']
mean_score = np.mean(cv_results)
std_score = np.std(cv_results)

# Print the mean and standard deviation of each fold's performance
print("Mean cross-validation score: {:.2f}".format(mean_score))
print("Standard deviation of cross-validation score: {:.2f}".format(std_score))

# Print the best hyperparameters found by GridSearchCV
print("Best hyperparameters: {}".format(gridcv_LR.best_params_))
print("Best score: {}".format(gridcv_LR.best_score_))

# Prediction Results
print(classification_report(df4_y_test, df4_BERT_LR_pred))
print("Accuracy: ", accuracy_score(df4_y_test, df4_BERT_LR_pred))
print("f1_score: ", f1_score(df4_y_test, df4_BERT_LR_pred, average='micro'))
print("precision_score: ", precision_score(df4_y_test, df4_BERT_LR_pred, average='micro'))
print("recall_score: ", recall_score(df4_y_test, df4_BERT_LR_pred, average='micro'))

# Computational Cost
print('Train Time: ', stop_train - start_train, 'Seconds')
print('Prediction Time : ', stop_predict - start_predict, 'Seconds')
print('Inference Time: ', (stop_predict - start_predict) * 1000 / len(df4_X_test), 'Milliseconds')

# Confusion matrix
cm = confusion_matrix(df4_y_test, df4_BERT_LR_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Saved model and preds
np.savetxt("df4_BERT_LR_pred.csv", df4_BERT_LR_pred, delimiter=",")
joblib.dump(gridcv_LR, 'df4_BERT_LR.sav')

#######################################DistilBERT

#Training and gridsearchcv
start_train = timeit.default_timer()
df4_DistilBERT_LR = Pipeline([
    ("embedding", HFTransformersLanguage("distilbert-base-uncased")),
    ("model", LogisticRegression())
])
gridcv_LR = GridSearchCV(df4_DistilBERT_LR, param_grid = grid_LR, scoring='f1_micro', cv= 5, n_jobs = -1, verbose = 2, refit = True)
gridcv_LR.fit(df4_X_train, df4_y_train)
stop_train = timeit.default_timer()

# Prediction
start_predict = timeit.default_timer()
df4_DistilBERT_LR_pred = gridcv_LR.predict(df4_X_test)
stop_predict = timeit.default_timer()

# Training Results
cv_results = gridcv_LR.cv_results_['split0_test_score'],
gridcv_LR.cv_results_['split1_test_score'],
gridcv_LR.cv_results_['split2_test_score'],
gridcv_LR.cv_results_['split3_test_score'],
gridcv_LR.cv_results_['split4_test_score']
mean_score = np.mean(cv_results)
std_score = np.std(cv_results)

# Print the mean and standard deviation of each fold's performance
print("Mean cross-validation score: {:.2f}".format(mean_score))
print("Standard deviation of cross-validation score: {:.2f}".format(std_score))

# Print the best hyperparameters found by GridSearchCV
print("Best hyperparameters: {}".format(gridcv_LR.best_params_))
print("Best score: {}".format(gridcv_LR.best_score_))

# Prediction Results
print(classification_report(df4_y_test, df4_DistilBERT_LR_pred))
print("Accuracy: ", accuracy_score(df4_y_test, df4_DistilBERT_LR_pred))
print("f1_score: ", f1_score(df4_y_test, df4_DistilBERT_LR_pred, average='micro'))
print("precision_score: ", precision_score(df4_y_test, df4_DistilBERT_LR_pred, average='micro'))
print("recall_score: ", recall_score(df4_y_test, df4_DistilBERT_LR_pred, average='micro'))

# Computational Cost
print('Train Time: ', stop_train - start_train, 'Seconds')
print('Prediction Time : ', stop_predict - start_predict, 'Seconds')
print('Inference Time: ', (stop_predict - start_predict) * 1000 / len(df4_X_test), 'Milliseconds')

# Confusion matrix
cm = confusion_matrix(df4_y_test, df4_DistilBERT_LR_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Saved model and preds
np.savetxt("df4_DistilBERT_LR_pred.csv", df4_DistilBERT_LR_pred, delimiter=",")
joblib.dump(gridcv_LR, 'df4_DistilBERT_LR.sav')

#######################################ALBERT

#Training and gridsearchcv
start_train = timeit.default_timer()
df4_ALBERT_LR = Pipeline([
    ("embedding", HFTransformersLanguage("albert-base-v2")),
    ("model", LogisticRegression())
])
gridcv_LR = GridSearchCV(df4_ALBERT_LR, param_grid = grid_LR, scoring='f1_micro', cv= 5, n_jobs = -1, verbose = 2, refit = True)
gridcv_LR.fit(df4_X_train, df4_y_train)
stop_train = timeit.default_timer()

# Prediction
start_predict = timeit.default_timer()
df4_ALBERT_LR_pred = gridcv_LR.predict(df4_X_test)
stop_predict = timeit.default_timer()

# Training Results
cv_results = gridcv_LR.cv_results_['split0_test_score'],
gridcv_LR.cv_results_['split1_test_score'],
gridcv_LR.cv_results_['split2_test_score'],
gridcv_LR.cv_results_['split3_test_score'],
gridcv_LR.cv_results_['split4_test_score']
mean_score = np.mean(cv_results)
std_score = np.std(cv_results)

# Print the mean and standard deviation of each fold's performance
print("Mean cross-validation score: {:.2f}".format(mean_score))
print("Standard deviation of cross-validation score: {:.2f}".format(std_score))

# Print the best hyperparameters found by GridSearchCV
print("Best hyperparameters: {}".format(gridcv_LR.best_params_))
print("Best score: {}".format(gridcv_LR.best_score_))

# Prediction Results
print(classification_report(df4_y_test, df4_ALBERT_LR_pred))
print("Accuracy: ", accuracy_score(df4_y_test, df4_ALBERT_LR_pred))
print("f1_score: ", f1_score(df4_y_test, df4_ALBERT_LR_pred, average='micro'))
print("precision_score: ", precision_score(df4_y_test, df4_ALBERT_LR_pred, average='micro'))
print("recall_score: ", recall_score(df4_y_test, df4_ALBERT_LR_pred, average='micro'))

# Computational Cost
print('Train Time: ', stop_train - start_train, 'Seconds')
print('Prediction Time : ', stop_predict - start_predict, 'Seconds')
print('Inference Time: ', (stop_predict - start_predict) * 1000 / len(df4_X_test), 'Milliseconds')

# Confusion matrix
cm = confusion_matrix(df4_y_test, df4_ALBERT_LR_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Saved model and preds
np.savetxt("df4_ALBERT_LR_pred.csv", df4_ALBERT_LR_pred, delimiter=",")
joblib.dump(gridcv_LR, 'df4_ALBERT_LR.sav')

#######################################BART

#Training and gridsearchcv
start_train = timeit.default_timer()
df4_BART_LR = Pipeline([
    ("embedding", HFTransformersLanguage("facebook/bart-base")),
    ("model", LogisticRegression())
])
gridcv_LR = GridSearchCV(df4_BART_LR, param_grid = grid_LR, scoring='f1_micro', cv= 5, n_jobs = -1, verbose = 2, refit = True)
gridcv_LR.fit(df4_X_train, df4_y_train)
stop_train = timeit.default_timer()

# Prediction
start_predict = timeit.default_timer()
df4_BART_LR_pred = gridcv_LR.predict(df4_X_test)
stop_predict = timeit.default_timer()

# Training Results
cv_results = gridcv_LR.cv_results_['split0_test_score'],
gridcv_LR.cv_results_['split1_test_score'],
gridcv_LR.cv_results_['split2_test_score'],
gridcv_LR.cv_results_['split3_test_score'],
gridcv_LR.cv_results_['split4_test_score']
mean_score = np.mean(cv_results)
std_score = np.std(cv_results)

# Print the mean and standard deviation of each fold's performance
print("Mean cross-validation score: {:.2f}".format(mean_score))
print("Standard deviation of cross-validation score: {:.2f}".format(std_score))

# Print the best hyperparameters found by GridSearchCV
print("Best hyperparameters: {}".format(gridcv_LR.best_params_))
print("Best score: {}".format(gridcv_LR.best_score_))

# Prediction Results
print(classification_report(df4_y_test, df4_BART_LR_pred))
print("Accuracy: ", accuracy_score(df4_y_test, df4_BART_LR_pred))
print("f1_score: ", f1_score(df4_y_test, df4_BART_LR_pred, average='micro'))
print("precision_score: ", precision_score(df4_y_test, df4_BART_LR_pred, average='micro'))
print("recall_score: ", recall_score(df4_y_test, df4_BART_LR_pred, average='micro'))

# Computational Cost
print('Train Time: ', stop_train - start_train, 'Seconds')
print('Prediction Time : ', stop_predict - start_predict, 'Seconds')
print('Inference Time: ', (stop_predict - start_predict) * 1000 / len(df4_X_test), 'Milliseconds')

# Confusion matrix
cm = confusion_matrix(df4_y_test, df4_BART_LR_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Saved model and preds
np.savetxt("df4_BART_LR_pred.csv", df4_BART_LR_pred, delimiter=",")
joblib.dump(gridcv_LR, 'df4_BART_LR.sav')

#######################################RoBERTa

#Training and gridsearchcv
start_train = timeit.default_timer()
df4_RoBERTa_LR = Pipeline([
    ("embedding", HFTransformersLanguage("roberta-base")),
    ("model", LogisticRegression())
])
gridcv_LR = GridSearchCV(df4_RoBERTa_LR, param_grid = grid_LR, scoring='f1_micro', cv= 5, n_jobs = -1, verbose = 2, refit = True)
gridcv_LR.fit(df4_X_train, df4_y_train)
stop_train = timeit.default_timer()

# Prediction
start_predict = timeit.default_timer()
df4_RoBERTa_LR_pred = gridcv_LR.predict(df4_X_test)
stop_predict = timeit.default_timer()

# Training Results
cv_results = gridcv_LR.cv_results_['split0_test_score'],
gridcv_LR.cv_results_['split1_test_score'],
gridcv_LR.cv_results_['split2_test_score'],
gridcv_LR.cv_results_['split3_test_score'],
gridcv_LR.cv_results_['split4_test_score']
mean_score = np.mean(cv_results)
std_score = np.std(cv_results)

# Print the mean and standard deviation of each fold's performance
print("Mean cross-validation score: {:.2f}".format(mean_score))
print("Standard deviation of cross-validation score: {:.2f}".format(std_score))

# Print the best hyperparameters found by GridSearchCV
print("Best hyperparameters: {}".format(gridcv_LR.best_params_))
print("Best score: {}".format(gridcv_LR.best_score_))

# Prediction Results
print(classification_report(df4_y_test, df4_RoBERTa_LR_pred))
print("Accuracy: ", accuracy_score(df4_y_test, df4_RoBERTa_LR_pred))
print("f1_score: ", f1_score(df4_y_test, df4_RoBERTa_LR_pred, average='micro'))
print("precision_score: ", precision_score(df4_y_test, df4_RoBERTa_LR_pred, average='micro'))
print("recall_score: ", recall_score(df4_y_test, df4_RoBERTa_LR_pred, average='micro'))

# Computational Cost
print('Train Time: ', stop_train - start_train, 'Seconds')
print('Prediction Time : ', stop_predict - start_predict, 'Seconds')
print('Inference Time: ', (stop_predict - start_predict) * 1000 / len(df4_X_test), 'Milliseconds')

# Confusion matrix
cm = confusion_matrix(df4_y_test, df4_RoBERTa_LR_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Saved model and preds
np.savetxt("df4_RoBERTa_LR_pred.csv", df4_RoBERTa_LR_pred, delimiter=",")
joblib.dump(gridcv_LR, 'df4_RoBERTa_LR.sav')

#######################################ELECTRA

#Training and gridsearchcv
start_train = timeit.default_timer()
df4_ELECTRA_LR = Pipeline([
    ("embedding", HFTransformersLanguage("google/electra-small-discriminator")),
    ("model", LogisticRegression())
])
gridcv_LR = GridSearchCV(df4_ELECTRA_LR, param_grid = grid_LR, scoring='f1_micro', cv= 5, n_jobs = -1, verbose = 2, refit = True)
gridcv_LR.fit(df4_X_train, df4_y_train)
stop_train = timeit.default_timer()

# Prediction
start_predict = timeit.default_timer()
df4_ELECTRA_LR_pred = gridcv_LR.predict(df4_X_test)
stop_predict = timeit.default_timer()

# Training Results
cv_results = gridcv_LR.cv_results_['split0_test_score'],
gridcv_LR.cv_results_['split1_test_score'],
gridcv_LR.cv_results_['split2_test_score'],
gridcv_LR.cv_results_['split3_test_score'],
gridcv_LR.cv_results_['split4_test_score']
mean_score = np.mean(cv_results)
std_score = np.std(cv_results)

# Print the mean and standard deviation of each fold's performance
print("Mean cross-validation score: {:.2f}".format(mean_score))
print("Standard deviation of cross-validation score: {:.2f}".format(std_score))

# Print the best hyperparameters found by GridSearchCV
print("Best hyperparameters: {}".format(gridcv_LR.best_params_))
print("Best score: {}".format(gridcv_LR.best_score_))

# Prediction Results
print(classification_report(df4_y_test, df4_ELECTRA_LR_pred))
print("Accuracy: ", accuracy_score(df4_y_test, df4_ELECTRA_LR_pred))
print("f1_score: ", f1_score(df4_y_test, df4_ELECTRA_LR_pred, average='micro'))
print("precision_score: ", precision_score(df4_y_test, df4_ELECTRA_LR_pred, average='micro'))
print("recall_score: ", recall_score(df4_y_test, df4_ELECTRA_LR_pred, average='micro'))

# Computational Cost
print('Train Time: ', stop_train - start_train, 'Seconds')
print('Prediction Time : ', stop_predict - start_predict, 'Seconds')
print('Inference Time: ', (stop_predict - start_predict) * 1000 / len(df4_X_test), 'Milliseconds')

# Confusion matrix
cm = confusion_matrix(df4_y_test, df4_ELECTRA_LR_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Saved model and preds
np.savetxt("df4_ELECTRA_LR_pred.csv", df4_ELECTRA_LR_pred, delimiter=",")
joblib.dump(gridcv_LR, 'df4_ELECTRA_LR.sav')

#######################################XLNet

#Training and gridsearchcv
start_train = timeit.default_timer()
df4_XLNET_LR = Pipeline([
    ("embedding", HFTransformersLanguage("xlnet-base-cased")),
    ("model", LogisticRegression())
])
gridcv_LR = GridSearchCV(df4_XLNET_LR, param_grid = grid_LR, scoring='f1_micro', cv= 5, n_jobs = -1, verbose = 2, refit = True)
gridcv_LR.fit(df4_X_train, df4_y_train)
stop_train = timeit.default_timer()

# Prediction
start_predict = timeit.default_timer()
df4_XLNET_LR_pred = gridcv_LR.predict(df4_X_test)
stop_predict = timeit.default_timer()

# Training Results
cv_results = gridcv_LR.cv_results_['split0_test_score'],
gridcv_LR.cv_results_['split1_test_score'],
gridcv_LR.cv_results_['split2_test_score'],
gridcv_LR.cv_results_['split3_test_score'],
gridcv_LR.cv_results_['split4_test_score']
mean_score = np.mean(cv_results)
std_score = np.std(cv_results)

# Print the mean and standard deviation of each fold's performance
print("Mean cross-validation score: {:.2f}".format(mean_score))
print("Standard deviation of cross-validation score: {:.2f}".format(std_score))

# Print the best hyperparameters found by GridSearchCV
print("Best hyperparameters: {}".format(gridcv_LR.best_params_))
print("Best score: {}".format(gridcv_LR.best_score_))

# Prediction Results
print(classification_report(df4_y_test, df4_XLNET_LR_pred))
print("Accuracy: ", accuracy_score(df4_y_test, df4_XLNET_LR_pred))
print("f1_score: ", f1_score(df4_y_test, df4_XLNET_LR_pred, average='micro'))
print("precision_score: ", precision_score(df4_y_test, df4_XLNET_LR_pred, average='micro'))
print("recall_score: ", recall_score(df4_y_test, df4_XLNET_LR_pred, average='micro'))

# Computational Cost
print('Train Time: ', stop_train - start_train, 'Seconds')
print('Prediction Time : ', stop_predict - start_predict, 'Seconds')
print('Inference Time: ', (stop_predict - start_predict) * 1000 / len(df4_X_test), 'Milliseconds')

# Confusion matrix
cm = confusion_matrix(df4_y_test, df4_XLNET_LR_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Saved model and preds
np.savetxt("df4_XLNET_LR_pred.csv", df4_XLNET_LR_pred, delimiter=",")
joblib.dump(gridcv_LR, 'df4_XLNET_LR.sav')

