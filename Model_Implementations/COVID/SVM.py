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
df3_X_train = df3_train["Text"].tolist()
df3_y_train = df3_train["Label"].tolist()
df3_X_test = df3_test["Text"].tolist()
df3_y_test = df3_test["Label"].tolist()

#######################################TF
#Training and gridsearchcv
start_train = timeit.default_timer()
gridcv_SVM.fit(TF_df3_train, df3_train_class)
stop_train = timeit.default_timer()

#Prediction
start_predict = timeit.default_timer()
TF_df3_pred = gridcv_SVM.predict(TF_df3_test)
stop_predict = timeit.default_timer()

#Training Results
cv_results = gridcv_SVM.cv_results_['split0_test_score'],
gridcv_SVM.cv_results_['split1_test_score'],
gridcv_SVM.cv_results_['split2_test_score'],
gridcv_SVM.cv_results_['split3_test_score'],
gridcv_SVM.cv_results_['split4_test_score']
mean_score = np.mean(cv_results)
std_score = np.std(cv_results)

# Print the mean and standard deviation of each fold's performance
print("Mean cross-validation score: {:.2f}".format(mean_score))
print("Standard deviation of cross-validation score: {:.2f}".format(std_score))

# Print the best hyperparameters found by GridSearchCV
print("Best hyperparameters: {}".format(gridcv_SVM.best_params_))
print("Best score: {}".format(gridcv_SVM.best_score_))

#Prediction Results
print(classification_report(df3_test_class, TF_df3_pred))
print("Accuracy: ", accuracy_score(df3_test_class, TF_df3_pred))
print("f1_score: ", f1_score(df3_test_class, TF_df3_pred, average='micro'))
print("precision_score: ", precision_score(df3_test_class, TF_df3_pred, average='micro'))
print("recall_score: ", recall_score(df3_test_class, TF_df3_pred, average='micro'))

#Computational Cost
print('Train Time: ', stop_train - start_train)
print('Prediction Time : ', stop_predict - start_predict)
print('Inference Time: ', (stop_predict - start_predict)/TF_df3_test.shape[0])

#Confusion matrix
cm = confusion_matrix(df3_test_class, TF_df3_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

#Saved model and preds
np.savetxt("df3_SVM_TF_pred.csv", TF_df3_pred, delimiter=",")
joblib.dump(gridcv_SVM, 'df3_TF_SVM.sav')

#######################################TFIDF
#Training and gridsearchcv
start_train = timeit.default_timer()
gridcv_SVM.fit(TFIDF_df3_train, df3_train_class)
stop_train = timeit.default_timer()

#Prediction
start_predict = timeit.default_timer()
TFIDF_df3_pred = gridcv_SVM.predict(TFIDF_df3_test)
stop_predict = timeit.default_timer()

#Training Results
cv_results = gridcv_SVM.cv_results_['split0_test_score'],
gridcv_SVM.cv_results_['split1_test_score'],
gridcv_SVM.cv_results_['split2_test_score'],
gridcv_SVM.cv_results_['split3_test_score'],
gridcv_SVM.cv_results_['split4_test_score']
mean_score = np.mean(cv_results)
std_score = np.std(cv_results)

# Print the mean and standard deviation of each fold's performance
print("Mean cross-validation score: {:.2f}".format(mean_score))
print("Standard deviation of cross-validation score: {:.2f}".format(std_score))

# Print the best hyperparameters found by GridSearchCV
print("Best hyperparameters: {}".format(gridcv_SVM.best_params_))
print("Best score: {}".format(gridcv_SVM.best_score_))

#Prediction Results
print(classification_report(df3_test_class, TFIDF_df3_pred))
print("Accuracy: ", accuracy_score(df3_test_class, TFIDF_df3_pred))
print("f1_score: ", f1_score(df3_test_class, TFIDF_df3_pred, average='micro'))
print("precision_score: ", precision_score(df3_test_class, TFIDF_df3_pred, average='micro'))
print("recall_score: ", recall_score(df3_test_class, TFIDF_df3_pred, average='micro'))

#Computational Cost
print('Train Time: ', stop_train - start_train)
print('Prediction Time : ', stop_predict - start_predict)
print('Inference Time: ', (stop_predict - start_predict)/TFIDF_df3_test.shape[0])

#Confusion matrix
cm = confusion_matrix(df3_test_class, TFIDF_df3_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

#Saved model and preds
np.savetxt("df3_SVM_TFIDF_pred.csv", TFIDF_df3_pred, delimiter=",")
joblib.dump(gridcv_SVM, 'df3_TFIDF_SVM.sav')

#######################################W2V
#Training and gridsearchcv
start_train = timeit.default_timer()
gridcv_SVM.fit(W2V_df3_train, df3_train_class)
stop_train = timeit.default_timer()

#Prediction
start_predict = timeit.default_timer()
W2V_df3_pred = gridcv_SVM.predict(W2V_df3_test)
stop_predict = timeit.default_timer()

#Training Results
cv_results = gridcv_SVM.cv_results_['split0_test_score'],
gridcv_SVM.cv_results_['split1_test_score'],
gridcv_SVM.cv_results_['split2_test_score'],
gridcv_SVM.cv_results_['split3_test_score'],
gridcv_SVM.cv_results_['split4_test_score']
mean_score = np.mean(cv_results)
std_score = np.std(cv_results)

# Print the mean and standard deviation of each fold's performance
print("Mean cross-validation score: {:.2f}".format(mean_score))
print("Standard deviation of cross-validation score: {:.2f}".format(std_score))

# Print the best hyperparameters found by GridSearchCV
print("Best hyperparameters: {}".format(gridcv_SVM.best_params_))
print("Best score: {}".format(gridcv_SVM.best_score_))

#Prediction Results
print(classification_report(df3_test_class, W2V_df3_pred))
print("Accuracy: ", accuracy_score(df3_test_class, W2V_df3_pred))
print("f1_score: ", f1_score(df3_test_class, W2V_df3_pred, average='micro'))
print("precision_score: ", precision_score(df3_test_class, W2V_df3_pred, average='micro'))
print("recall_score: ", recall_score(df3_test_class, W2V_df3_pred, average='micro'))

#Computational Cost
print('Train Time: ', stop_train - start_train)
print('Prediction Time : ', stop_predict - start_predict)
print('Inference Time: ', (stop_predict - start_predict)/W2V_df3_test.shape[0])

#Confusion matrix
cm = confusion_matrix(df3_test_class, W2V_df3_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

#Saved model and preds
np.savetxt("df3_SVM_W2V_pred.csv", W2V_df3_pred, delimiter=",")
joblib.dump(gridcv_SVM, 'df3_W2V_SVM.sav')

#######################################Glove
#Training and gridsearchcv
start_train = timeit.default_timer()
gridcv_SVM.fit(Glove_df3_train, df3_train_class)
stop_train = timeit.default_timer()

#Prediction
start_predict = timeit.default_timer()
Glove_df3_pred = gridcv_SVM.predict(Glove_df3_test)
stop_predict = timeit.default_timer()

#Training Results
cv_results = gridcv_SVM.cv_results_['split0_test_score'],
gridcv_SVM.cv_results_['split1_test_score'],
gridcv_SVM.cv_results_['split2_test_score'],
gridcv_SVM.cv_results_['split3_test_score'],
gridcv_SVM.cv_results_['split4_test_score']
mean_score = np.mean(cv_results)
std_score = np.std(cv_results)

# Print the mean and standard deviation of each fold's performance
print("Mean cross-validation score: {:.2f}".format(mean_score))
print("Standard deviation of cross-validation score: {:.2f}".format(std_score))

# Print the best hyperparameters found by GridSearchCV
print("Best hyperparameters: {}".format(gridcv_SVM.best_params_))
print("Best score: {}".format(gridcv_SVM.best_score_))

#Prediction Results
print(classification_report(df3_test_class, Glove_df3_pred))
print("Accuracy: ", accuracy_score(df3_test_class, Glove_df3_pred))
print("f1_score: ", f1_score(df3_test_class, Glove_df3_pred, average='micro'))
print("precision_score: ", precision_score(df3_test_class, Glove_df3_pred, average='micro'))
print("recall_score: ", recall_score(df3_test_class, Glove_df3_pred, average='micro'))

#Computational Cost
print('Train Time: ', stop_train - start_train)
print('Prediction Time : ', stop_predict - start_predict)
print('Inference Time: ', (stop_predict - start_predict)/Glove_df3_test.shape[0])

#Confusion matrix
cm = confusion_matrix(df3_test_class, Glove_df3_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

#Saved model and preds
np.savetxt("df3_SVM_Glove_pred.csv", Glove_df3_pred, delimiter=",")
joblib.dump(gridcv_SVM, 'df3_Glove_SVM.sav')

#######################################Fasttext
#Training and gridsearchcv
start_train = timeit.default_timer()
gridcv_SVM.fit(Fasttext_df3_train, df3_train_class)
stop_train = timeit.default_timer()

#Prediction
start_predict = timeit.default_timer()
Fasttext_df3_pred = gridcv_SVM.predict(Fasttext_df3_test)
stop_predict = timeit.default_timer()

#Training Results
cv_results = gridcv_SVM.cv_results_['split0_test_score'],
gridcv_SVM.cv_results_['split1_test_score'],
gridcv_SVM.cv_results_['split2_test_score'],
gridcv_SVM.cv_results_['split3_test_score'],
gridcv_SVM.cv_results_['split4_test_score']
mean_score = np.mean(cv_results)
std_score = np.std(cv_results)

# Print the mean and standard deviation of each fold's performance
print("Mean cross-validation score: {:.2f}".format(mean_score))
print("Standard deviation of cross-validation score: {:.2f}".format(std_score))

# Print the best hyperparameters found by GridSearchCV
print("Best hyperparameters: {}".format(gridcv_SVM.best_params_))
print("Best score: {}".format(gridcv_SVM.best_score_))

#Prediction Results
print(classification_report(df3_test_class, Fasttext_df3_pred))
print("Accuracy: ", accuracy_score(df3_test_class, Fasttext_df3_pred))
print("f1_score: ", f1_score(df3_test_class, Fasttext_df3_pred, average='micro'))
print("precision_score: ", precision_score(df3_test_class, Fasttext_df3_pred, average='micro'))
print("recall_score: ", recall_score(df3_test_class, Fasttext_df3_pred, average='micro'))

#Computational Cost
print('Train Time: ', stop_train - start_train)
print('Prediction Time : ', stop_predict - start_predict)
print('Inference Time: ', (stop_predict - start_predict)/Fasttext_df3_test.shape[0])

#Confusion matrix
cm = confusion_matrix(df3_test_class, Fasttext_df3_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

#Saved model and preds
np.savetxt("df3_SVM_Fasttext_pred.csv", Fasttext_df3_pred, delimiter=",")
joblib.dump(gridcv_SVM, 'df3_Fasttext_SVM.sav')

#######################################ELMO

#Training and gridsearchcv
start_train = timeit.default_timer()
gridcv_SVM.fit(ELMO_df3_train, df3_train_class)
stop_train = timeit.default_timer()

#Prediction
start_predict = timeit.default_timer()
ELMO_df3_pred = gridcv_SVM.predict(ELMO_df3_test)
stop_predict = timeit.default_timer()

#Training Results
cv_results = gridcv_SVM.cv_results_['split0_test_score'],
gridcv_SVM.cv_results_['split1_test_score'],
gridcv_SVM.cv_results_['split2_test_score'],
gridcv_SVM.cv_results_['split3_test_score'],
gridcv_SVM.cv_results_['split4_test_score']
mean_score = np.mean(cv_results)
std_score = np.std(cv_results)

# Print the mean and standard deviation of each fold's performance
print("Mean cross-validation score: {:.2f}".format(mean_score))
print("Standard deviation of cross-validation score: {:.2f}".format(std_score))

# Print the best hyperparameters found by GridSearchCV
print("Best hyperparameters: {}".format(gridcv_SVM.best_params_))
print("Best score: {}".format(gridcv_SVM.best_score_))

#Prediction Results
print(classification_report(df3_test_class, ELMO_df3_pred))
print("Accuracy: ", accuracy_score(df3_test_class, ELMO_df3_pred))
print("f1_score: ", f1_score(df3_test_class, ELMO_df3_pred, average='micro'))
print("precision_score: ", precision_score(df3_test_class, ELMO_df3_pred, average='micro'))
print("recall_score: ", recall_score(df3_test_class, ELMO_df3_pred, average='micro'))

#Computational Cost
print('Train Time: ', stop_train - start_train)
print('Prediction Time : ', stop_predict - start_predict)
print('Inference Time: ', (stop_predict - start_predict)/ELMO_df3_test.shape[0])

#Confusion matrix
cm = confusion_matrix(df3_test_class, ELMO_df3_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

#Saved model and preds
np.savetxt("df3_ELMO_SVM_pred.csv", ELMO_df3_pred, delimiter=",")
joblib.dump(gridcv_SVM, 'df3_ELMO_SVM.sav')

#######################################BERT

#Training and gridsearchcv
start_train = timeit.default_timer()
df3_BERT_SVM = Pipeline([
    ("embedding", HFTransformersLanguage("bert-base-uncased")),
    ("model", SVC())
])
gridcv_SVM = GridSearchCV(df3_BERT_SVM, param_grid = grid_SVM, scoring='f1_micro', cv= 5, n_jobs = -1, verbose = 2, refit = True)
gridcv_SVM.fit(df3_X_train, df3_y_train)
stop_train = timeit.default_timer()

# Prediction
start_predict = timeit.default_timer()
df3_BERT_SVM_pred = gridcv_SVM.predict(df3_X_test)
stop_predict = timeit.default_timer()

# Training Results
cv_results = gridcv_SVM.cv_results_['split0_test_score'],
gridcv_SVM.cv_results_['split1_test_score'],
gridcv_SVM.cv_results_['split2_test_score'],
gridcv_SVM.cv_results_['split3_test_score'],
gridcv_SVM.cv_results_['split4_test_score']
mean_score = np.mean(cv_results)
std_score = np.std(cv_results)

# Print the mean and standard deviation of each fold's performance
print("Mean cross-validation score: {:.2f}".format(mean_score))
print("Standard deviation of cross-validation score: {:.2f}".format(std_score))

# Print the best hyperparameters found by GridSearchCV
print("Best hyperparameters: {}".format(gridcv_SVM.best_params_))
print("Best score: {}".format(gridcv_SVM.best_score_))

# Prediction Results
print(classification_report(df3_y_test, df3_BERT_SVM_pred))
print("Accuracy: ", accuracy_score(df3_y_test, df3_BERT_SVM_pred))
print("f1_score: ", f1_score(df3_y_test, df3_BERT_SVM_pred, average='micro'))
print("precision_score: ", precision_score(df3_y_test, df3_BERT_SVM_pred, average='micro'))
print("recall_score: ", recall_score(df3_y_test, df3_BERT_SVM_pred, average='micro'))

# Computational Cost
print('Train Time: ', stop_train - start_train, 'Seconds')
print('Prediction Time : ', stop_predict - start_predict, 'Seconds')
print('Inference Time: ', (stop_predict - start_predict) * 1000 / len(df3_X_test), 'Milliseconds')

# Confusion matrix
cm = confusion_matrix(df3_y_test, df3_BERT_SVM_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Saved model and preds
np.savetxt("df3_BERT_SVM_pred.csv", df3_BERT_SVM_pred, delimiter=",")
joblib.dump(gridcv_SVM, 'df3_BERT_SVM.sav')

#######################################DistilBERT

#Training and gridsearchcv
start_train = timeit.default_timer()
df3_DistilBERT_SVM = Pipeline([
    ("embedding", HFTransformersLanguage("distilbert-base-uncased")),
    ("model", SVC())
])
gridcv_SVM = GridSearchCV(df3_DistilBERT_SVM, param_grid = grid_SVM, scoring='f1_micro', cv= 5, n_jobs = -1, verbose = 2, refit = True)
gridcv_SVM.fit(df3_X_train, df3_y_train)
stop_train = timeit.default_timer()

# Prediction
start_predict = timeit.default_timer()
df3_DistilBERT_SVM_pred = gridcv_SVM.predict(df3_X_test)
stop_predict = timeit.default_timer()

# Training Results
cv_results = gridcv_SVM.cv_results_['split0_test_score'],
gridcv_SVM.cv_results_['split1_test_score'],
gridcv_SVM.cv_results_['split2_test_score'],
gridcv_SVM.cv_results_['split3_test_score'],
gridcv_SVM.cv_results_['split4_test_score']
mean_score = np.mean(cv_results)
std_score = np.std(cv_results)

# Print the mean and standard deviation of each fold's performance
print("Mean cross-validation score: {:.2f}".format(mean_score))
print("Standard deviation of cross-validation score: {:.2f}".format(std_score))

# Print the best hyperparameters found by GridSearchCV
print("Best hyperparameters: {}".format(gridcv_SVM.best_params_))
print("Best score: {}".format(gridcv_SVM.best_score_))

# Prediction Results
print(classification_report(df3_y_test, df3_DistilBERT_SVM_pred))
print("Accuracy: ", accuracy_score(df3_y_test, df3_DistilBERT_SVM_pred))
print("f1_score: ", f1_score(df3_y_test, df3_DistilBERT_SVM_pred, average='micro'))
print("precision_score: ", precision_score(df3_y_test, df3_DistilBERT_SVM_pred, average='micro'))
print("recall_score: ", recall_score(df3_y_test, df3_DistilBERT_SVM_pred, average='micro'))

# Computational Cost
print('Train Time: ', stop_train - start_train, 'Seconds')
print('Prediction Time : ', stop_predict - start_predict, 'Seconds')
print('Inference Time: ', (stop_predict - start_predict) * 1000 / len(df3_X_test), 'Milliseconds')

# Confusion matrix
cm = confusion_matrix(df3_y_test, df3_DistilBERT_SVM_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Saved model and preds
np.savetxt("df3_DistilBERT_SVM_pred.csv", df3_DistilBERT_SVM_pred, delimiter=",")
joblib.dump(gridcv_SVM, 'df3_DistilBERT_SVM.sav')

#######################################ALBERT

#Training and gridsearchcv
start_train = timeit.default_timer()
df3_ALBERT_SVM = Pipeline([
    ("embedding", HFTransformersLanguage("albert-base-v2")),
    ("model", SVC())
])
gridcv_SVM = GridSearchCV(df3_ALBERT_SVM, param_grid = grid_SVM, scoring='f1_micro', cv= 5, n_jobs = -1, verbose = 2, refit = True)
gridcv_SVM.fit(df3_X_train, df3_y_train)
stop_train = timeit.default_timer()

# Prediction
start_predict = timeit.default_timer()
df3_ALBERT_SVM_pred = gridcv_SVM.predict(df3_X_test)
stop_predict = timeit.default_timer()

# Training Results
cv_results = gridcv_SVM.cv_results_['split0_test_score'],
gridcv_SVM.cv_results_['split1_test_score'],
gridcv_SVM.cv_results_['split2_test_score'],
gridcv_SVM.cv_results_['split3_test_score'],
gridcv_SVM.cv_results_['split4_test_score']
mean_score = np.mean(cv_results)
std_score = np.std(cv_results)

# Print the mean and standard deviation of each fold's performance
print("Mean cross-validation score: {:.2f}".format(mean_score))
print("Standard deviation of cross-validation score: {:.2f}".format(std_score))

# Print the best hyperparameters found by GridSearchCV
print("Best hyperparameters: {}".format(gridcv_SVM.best_params_))
print("Best score: {}".format(gridcv_SVM.best_score_))

# Prediction Results
print(classification_report(df3_y_test, df3_ALBERT_SVM_pred))
print("Accuracy: ", accuracy_score(df3_y_test, df3_ALBERT_SVM_pred))
print("f1_score: ", f1_score(df3_y_test, df3_ALBERT_SVM_pred, average='micro'))
print("precision_score: ", precision_score(df3_y_test, df3_ALBERT_SVM_pred, average='micro'))
print("recall_score: ", recall_score(df3_y_test, df3_ALBERT_SVM_pred, average='micro'))

# Computational Cost
print('Train Time: ', stop_train - start_train, 'Seconds')
print('Prediction Time : ', stop_predict - start_predict, 'Seconds')
print('Inference Time: ', (stop_predict - start_predict) * 1000 / len(df3_X_test), 'Milliseconds')

# Confusion matrix
cm = confusion_matrix(df3_y_test, df3_ALBERT_SVM_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Saved model and preds
np.savetxt("df3_ALBERT_SVM_pred.csv", df3_ALBERT_SVM_pred, delimiter=",")
joblib.dump(gridcv_SVM, 'df3_ALBERT_SVM.sav')

#######################################BART

#Training and gridsearchcv
start_train = timeit.default_timer()
df3_BART_SVM = Pipeline([
    ("embedding", HFTransformersLanguage("facebook/bart-base")),
    ("model", SVC())
])
gridcv_SVM = GridSearchCV(df3_BART_SVM, param_grid = grid_SVM, scoring='f1_micro', cv= 5, n_jobs = -1, verbose = 2, refit = True)
gridcv_SVM.fit(df3_X_train, df3_y_train)
stop_train = timeit.default_timer()

# Prediction
start_predict = timeit.default_timer()
df3_BART_SVM_pred = gridcv_SVM.predict(df3_X_test)
stop_predict = timeit.default_timer()

# Training Results
cv_results = gridcv_SVM.cv_results_['split0_test_score'],
gridcv_SVM.cv_results_['split1_test_score'],
gridcv_SVM.cv_results_['split2_test_score'],
gridcv_SVM.cv_results_['split3_test_score'],
gridcv_SVM.cv_results_['split4_test_score']
mean_score = np.mean(cv_results)
std_score = np.std(cv_results)

# Print the mean and standard deviation of each fold's performance
print("Mean cross-validation score: {:.2f}".format(mean_score))
print("Standard deviation of cross-validation score: {:.2f}".format(std_score))

# Print the best hyperparameters found by GridSearchCV
print("Best hyperparameters: {}".format(gridcv_SVM.best_params_))
print("Best score: {}".format(gridcv_SVM.best_score_))

# Prediction Results
print(classification_report(df3_y_test, df3_BART_SVM_pred))
print("Accuracy: ", accuracy_score(df3_y_test, df3_BART_SVM_pred))
print("f1_score: ", f1_score(df3_y_test, df3_BART_SVM_pred, average='micro'))
print("precision_score: ", precision_score(df3_y_test, df3_BART_SVM_pred, average='micro'))
print("recall_score: ", recall_score(df3_y_test, df3_BART_SVM_pred, average='micro'))

# Computational Cost
print('Train Time: ', stop_train - start_train, 'Seconds')
print('Prediction Time : ', stop_predict - start_predict, 'Seconds')
print('Inference Time: ', (stop_predict - start_predict) * 1000 / len(df3_X_test), 'Milliseconds')

# Confusion matrix
cm = confusion_matrix(df3_y_test, df3_BART_SVM_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Saved model and preds
np.savetxt("df3_BART_SVM_pred.csv", df3_BART_SVM_pred, delimiter=",")
joblib.dump(gridcv_SVM, 'df3_BART_SVM.sav')

#######################################RoBERTa

#Training and gridsearchcv
start_train = timeit.default_timer()
df3_RoBERTa_SVM = Pipeline([
    ("embedding", HFTransformersLanguage("roberta-base")),
    ("model", SVC())
])
gridcv_SVM = GridSearchCV(df3_RoBERTa_SVM, param_grid = grid_SVM, scoring='f1_micro', cv= 5, n_jobs = -1, verbose = 2, refit = True)
gridcv_SVM.fit(df3_X_train, df3_y_train)
stop_train = timeit.default_timer()

# Prediction
start_predict = timeit.default_timer()
df3_RoBERTa_SVM_pred = gridcv_SVM.predict(df3_X_test)
stop_predict = timeit.default_timer()

# Training Results
cv_results = gridcv_SVM.cv_results_['split0_test_score'],
gridcv_SVM.cv_results_['split1_test_score'],
gridcv_SVM.cv_results_['split2_test_score'],
gridcv_SVM.cv_results_['split3_test_score'],
gridcv_SVM.cv_results_['split4_test_score']
mean_score = np.mean(cv_results)
std_score = np.std(cv_results)

# Print the mean and standard deviation of each fold's performance
print("Mean cross-validation score: {:.2f}".format(mean_score))
print("Standard deviation of cross-validation score: {:.2f}".format(std_score))

# Print the best hyperparameters found by GridSearchCV
print("Best hyperparameters: {}".format(gridcv_SVM.best_params_))
print("Best score: {}".format(gridcv_SVM.best_score_))

# Prediction Results
print(classification_report(df3_y_test, df3_RoBERTa_SVM_pred))
print("Accuracy: ", accuracy_score(df3_y_test, df3_RoBERTa_SVM_pred))
print("f1_score: ", f1_score(df3_y_test, df3_RoBERTa_SVM_pred, average='micro'))
print("precision_score: ", precision_score(df3_y_test, df3_RoBERTa_SVM_pred, average='micro'))
print("recall_score: ", recall_score(df3_y_test, df3_RoBERTa_SVM_pred, average='micro'))

# Computational Cost
print('Train Time: ', stop_train - start_train, 'Seconds')
print('Prediction Time : ', stop_predict - start_predict, 'Seconds')
print('Inference Time: ', (stop_predict - start_predict) * 1000 / len(df3_X_test), 'Milliseconds')

# Confusion matrix
cm = confusion_matrix(df3_y_test, df3_RoBERTa_SVM_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Saved model and preds
np.savetxt("df3_RoBERTa_SVM_pred.csv", df3_RoBERTa_SVM_pred, delimiter=",")
joblib.dump(gridcv_SVM, 'df3_RoBERTa_SVM.sav')

#######################################ELECTRA

#Training and gridsearchcv
start_train = timeit.default_timer()
df3_ELECTRA_SVM = Pipeline([
    ("embedding", HFTransformersLanguage("google/electra-small-discriminator")),
    ("model", SVC())
])
gridcv_SVM = GridSearchCV(df3_ELECTRA_SVM, param_grid = grid_SVM, scoring='f1_micro', cv= 5, n_jobs = -1, verbose = 2, refit = True)
gridcv_SVM.fit(df3_X_train, df3_y_train)
stop_train = timeit.default_timer()

# Prediction
start_predict = timeit.default_timer()
df3_ELECTRA_SVM_pred = gridcv_SVM.predict(df3_X_test)
stop_predict = timeit.default_timer()

# Training Results
cv_results = gridcv_SVM.cv_results_['split0_test_score'],
gridcv_SVM.cv_results_['split1_test_score'],
gridcv_SVM.cv_results_['split2_test_score'],
gridcv_SVM.cv_results_['split3_test_score'],
gridcv_SVM.cv_results_['split4_test_score']
mean_score = np.mean(cv_results)
std_score = np.std(cv_results)

# Print the mean and standard deviation of each fold's performance
print("Mean cross-validation score: {:.2f}".format(mean_score))
print("Standard deviation of cross-validation score: {:.2f}".format(std_score))

# Print the best hyperparameters found by GridSearchCV
print("Best hyperparameters: {}".format(gridcv_SVM.best_params_))
print("Best score: {}".format(gridcv_SVM.best_score_))

# Prediction Results
print(classification_report(df3_y_test, df3_ELECTRA_SVM_pred))
print("Accuracy: ", accuracy_score(df3_y_test, df3_ELECTRA_SVM_pred))
print("f1_score: ", f1_score(df3_y_test, df3_ELECTRA_SVM_pred, average='micro'))
print("precision_score: ", precision_score(df3_y_test, df3_ELECTRA_SVM_pred, average='micro'))
print("recall_score: ", recall_score(df3_y_test, df3_ELECTRA_SVM_pred, average='micro'))

# Computational Cost
print('Train Time: ', stop_train - start_train, 'Seconds')
print('Prediction Time : ', stop_predict - start_predict, 'Seconds')
print('Inference Time: ', (stop_predict - start_predict) * 1000 / len(df3_X_test), 'Milliseconds')

# Confusion matrix
cm = confusion_matrix(df3_y_test, df3_ELECTRA_SVM_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Saved model and preds
np.savetxt("df3_ELECTRA_SVM_pred.csv", df3_ELECTRA_SVM_pred, delimiter=",")
joblib.dump(gridcv_SVM, 'df3_ELECTRA_SVM.sav')

#######################################XLNet

#Training and gridsearchcv
start_train = timeit.default_timer()
df3_XLNET_SVM = Pipeline([
    ("embedding", HFTransformersLanguage("xlnet-base-cased")),
    ("model", SVC())
])
gridcv_SVM = GridSearchCV(df3_XLNET_SVM, param_grid = grid_SVM, scoring='f1_micro', cv= 5, n_jobs = -1, verbose = 2, refit = True)
gridcv_SVM.fit(df3_X_train, df3_y_train)
stop_train = timeit.default_timer()

# Prediction
start_predict = timeit.default_timer()
df3_XLNET_SVM_pred = gridcv_SVM.predict(df3_X_test)
stop_predict = timeit.default_timer()

# Training Results
cv_results = gridcv_SVM.cv_results_['split0_test_score'],
gridcv_SVM.cv_results_['split1_test_score'],
gridcv_SVM.cv_results_['split2_test_score'],
gridcv_SVM.cv_results_['split3_test_score'],
gridcv_SVM.cv_results_['split4_test_score']
mean_score = np.mean(cv_results)
std_score = np.std(cv_results)

# Print the mean and standard deviation of each fold's performance
print("Mean cross-validation score: {:.2f}".format(mean_score))
print("Standard deviation of cross-validation score: {:.2f}".format(std_score))

# Print the best hyperparameters found by GridSearchCV
print("Best hyperparameters: {}".format(gridcv_SVM.best_params_))
print("Best score: {}".format(gridcv_SVM.best_score_))

# Prediction Results
print(classification_report(df3_y_test, df3_XLNET_SVM_pred))
print("Accuracy: ", accuracy_score(df3_y_test, df3_XLNET_SVM_pred))
print("f1_score: ", f1_score(df3_y_test, df3_XLNET_SVM_pred, average='micro'))
print("precision_score: ", precision_score(df3_y_test, df3_XLNET_SVM_pred, average='micro'))
print("recall_score: ", recall_score(df3_y_test, df3_XLNET_SVM_pred, average='micro'))

# Computational Cost
print('Train Time: ', stop_train - start_train, 'Seconds')
print('Prediction Time : ', stop_predict - start_predict, 'Seconds')
print('Inference Time: ', (stop_predict - start_predict) * 1000 / len(df3_X_test), 'Milliseconds')

# Confusion matrix
cm = confusion_matrix(df3_y_test, df3_XLNET_SVM_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Saved model and preds
np.savetxt("df3_XLNET_SVM_pred.csv", df3_XLNET_SVM_pred, delimiter=",")
joblib.dump(gridcv_SVM, 'df3_XLNET_SVM.sav')

