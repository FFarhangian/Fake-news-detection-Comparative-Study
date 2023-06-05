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
df2_X_train = df2_train["Text"].tolist()
df2_y_train = df2_train["Label"].tolist()
df2_X_test = df2_test["Text"].tolist()
df2_y_test = df2_test["Label"].tolist()

#######################################TF
#Training and gridsearchcv
start_train = timeit.default_timer()
gridcv_LR.fit(TF_df2_train, df2_train_class)
stop_train = timeit.default_timer()

#Prediction
start_predict = timeit.default_timer()
TF_df2_pred = gridcv_LR.predict(TF_df2_test)
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
print(classification_report(df2_test_class, TF_df2_pred))
print("Accuracy: ", accuracy_score(df2_test_class, TF_df2_pred))
print("f1_score: ", f1_score(df2_test_class, TF_df2_pred, average='micro'))
print("precision_score: ", precision_score(df2_test_class, TF_df2_pred, average='micro'))
print("recall_score: ", recall_score(df2_test_class, TF_df2_pred, average='micro'))

#Computational Cost
print('Train Time: ', stop_train - start_train)
print('Prediction Time : ', stop_predict - start_predict)
print('Inference Time: ', (stop_predict - start_predict)/TF_df2_test.shape[0])

#Confusion matrix
cm = confusion_matrix(df2_test_class, TF_df2_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

#Saved model and preds
np.savetxt("df2_LR_TF_pred.csv", TF_df2_pred, delimiter=",")
joblib.dump(gridcv_LR, 'df2_TF_LR.sav')

#######################################TFIDF
#Training and gridsearchcv
start_train = timeit.default_timer()
gridcv_LR.fit(TFIDF_df2_train, df2_train_class)
stop_train = timeit.default_timer()

#Prediction
start_predict = timeit.default_timer()
TFIDF_df2_pred = gridcv_LR.predict(TFIDF_df2_test)
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
print(classification_report(df2_test_class, TFIDF_df2_pred))
print("Accuracy: ", accuracy_score(df2_test_class, TFIDF_df2_pred))
print("f1_score: ", f1_score(df2_test_class, TFIDF_df2_pred, average='micro'))
print("precision_score: ", precision_score(df2_test_class, TFIDF_df2_pred, average='micro'))
print("recall_score: ", recall_score(df2_test_class, TFIDF_df2_pred, average='micro'))

#Computational Cost
print('Train Time: ', stop_train - start_train)
print('Prediction Time : ', stop_predict - start_predict)
print('Inference Time: ', (stop_predict - start_predict)/TFIDF_df2_test.shape[0])

#Confusion matrix
cm = confusion_matrix(df2_test_class, TFIDF_df2_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

#Saved model and preds
np.savetxt("df2_LR_TFIDF_pred.csv", TFIDF_df2_pred, delimiter=",")
joblib.dump(gridcv_LR, 'df2_TFIDF_LR.sav')

#######################################W2V
#Training and gridsearchcv
start_train = timeit.default_timer()
gridcv_LR.fit(W2V_df2_train, df2_train_class)
stop_train = timeit.default_timer()

#Prediction
start_predict = timeit.default_timer()
W2V_df2_pred = gridcv_LR.predict(W2V_df2_test)
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
print(classification_report(df2_test_class, W2V_df2_pred))
print("Accuracy: ", accuracy_score(df2_test_class, W2V_df2_pred))
print("f1_score: ", f1_score(df2_test_class, W2V_df2_pred, average='micro'))
print("precision_score: ", precision_score(df2_test_class, W2V_df2_pred, average='micro'))
print("recall_score: ", recall_score(df2_test_class, W2V_df2_pred, average='micro'))

#Computational Cost
print('Train Time: ', stop_train - start_train)
print('Prediction Time : ', stop_predict - start_predict)
print('Inference Time: ', (stop_predict - start_predict)/W2V_df2_test.shape[0])

#Confusion matrix
cm = confusion_matrix(df2_test_class, W2V_df2_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

#Saved model and preds
np.savetxt("df2_LR_W2V_pred.csv", W2V_df2_pred, delimiter=",")
joblib.dump(gridcv_LR, 'df2_W2V_LR.sav')

#######################################Glove
#Training and gridsearchcv
start_train = timeit.default_timer()
gridcv_LR.fit(Glove_df2_train, df2_train_class)
stop_train = timeit.default_timer()

#Prediction
start_predict = timeit.default_timer()
Glove_df2_pred = gridcv_LR.predict(Glove_df2_test)
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
print(classification_report(df2_test_class, Glove_df2_pred))
print("Accuracy: ", accuracy_score(df2_test_class, Glove_df2_pred))
print("f1_score: ", f1_score(df2_test_class, Glove_df2_pred, average='micro'))
print("precision_score: ", precision_score(df2_test_class, Glove_df2_pred, average='micro'))
print("recall_score: ", recall_score(df2_test_class, Glove_df2_pred, average='micro'))

#Computational Cost
print('Train Time: ', stop_train - start_train)
print('Prediction Time : ', stop_predict - start_predict)
print('Inference Time: ', (stop_predict - start_predict)/Glove_df2_test.shape[0])

#Confusion matrix
cm = confusion_matrix(df2_test_class, Glove_df2_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

#Saved model and preds
np.savetxt("df2_LR_Glove_pred.csv", Glove_df2_pred, delimiter=",")
joblib.dump(gridcv_LR, 'df2_Glove_LR.sav')

#######################################Fasttext
#Training and gridsearchcv
start_train = timeit.default_timer()
gridcv_LR.fit(Fasttext_df2_train, df2_train_class)
stop_train = timeit.default_timer()

#Prediction
start_predict = timeit.default_timer()
Fasttext_df2_pred = gridcv_LR.predict(Fasttext_df2_test)
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
print(classification_report(df2_test_class, Fasttext_df2_pred))
print("Accuracy: ", accuracy_score(df2_test_class, Fasttext_df2_pred))
print("f1_score: ", f1_score(df2_test_class, Fasttext_df2_pred, average='micro'))
print("precision_score: ", precision_score(df2_test_class, Fasttext_df2_pred, average='micro'))
print("recall_score: ", recall_score(df2_test_class, Fasttext_df2_pred, average='micro'))

#Computational Cost
print('Train Time: ', stop_train - start_train)
print('Prediction Time : ', stop_predict - start_predict)
print('Inference Time: ', (stop_predict - start_predict)/Fasttext_df2_test.shape[0])

#Confusion matrix
cm = confusion_matrix(df2_test_class, Fasttext_df2_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

#Saved model and preds
np.savetxt("df2_LR_Fasttext_pred.csv", Fasttext_df2_pred, delimiter=",")
joblib.dump(gridcv_LR, 'df2_Fasttext_LR.sav')

#######################################ELMO

#Training and gridsearchcv
start_train = timeit.default_timer()
gridcv_LR.fit(ELMO_df2_train, df2_train_class)
stop_train = timeit.default_timer()

#Prediction
start_predict = timeit.default_timer()
ELMO_df2_pred = gridcv_LR.predict(ELMO_df2_test)
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
print(classification_report(df2_test_class, ELMO_df2_pred))
print("Accuracy: ", accuracy_score(df2_test_class, ELMO_df2_pred))
print("f1_score: ", f1_score(df2_test_class, ELMO_df2_pred, average='micro'))
print("precision_score: ", precision_score(df2_test_class, ELMO_df2_pred, average='micro'))
print("recall_score: ", recall_score(df2_test_class, ELMO_df2_pred, average='micro'))

#Computational Cost
print('Train Time: ', stop_train - start_train)
print('Prediction Time : ', stop_predict - start_predict)
print('Inference Time: ', (stop_predict - start_predict)/ELMO_df2_test.shape[0])

#Confusion matrix
cm = confusion_matrix(df2_test_class, ELMO_df2_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

#Saved model and preds
np.savetxt("df2_ELMO_LR_pred.csv", ELMO_df2_pred, delimiter=",")
joblib.dump(gridcv_LR, 'df2_ELMO_LR.sav')

#######################################BERT

#Training and gridsearchcv
start_train = timeit.default_timer()
df2_BERT_LR = Pipeline([
    ("embedding", HFTransformersLanguage("bert-base-uncased")),
    ("model", LogisticRegression())
])
gridcv_LR = GridSearchCV(df2_BERT_LR, param_grid = grid_LR, scoring='f1_micro', cv= 5, n_jobs = -1, verbose = 2, refit = True)
gridcv_LR.fit(df2_X_train, df2_y_train)
stop_train = timeit.default_timer()

# Prediction
start_predict = timeit.default_timer()
df2_BERT_LR_pred = gridcv_LR.predict(df2_X_test)
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
print(classification_report(df2_y_test, df2_BERT_LR_pred))
print("Accuracy: ", accuracy_score(df2_y_test, df2_BERT_LR_pred))
print("f1_score: ", f1_score(df2_y_test, df2_BERT_LR_pred, average='micro'))
print("precision_score: ", precision_score(df2_y_test, df2_BERT_LR_pred, average='micro'))
print("recall_score: ", recall_score(df2_y_test, df2_BERT_LR_pred, average='micro'))

# Computational Cost
print('Train Time: ', stop_train - start_train, 'Seconds')
print('Prediction Time : ', stop_predict - start_predict, 'Seconds')
print('Inference Time: ', (stop_predict - start_predict) * 1000 / len(df2_X_test), 'Milliseconds')

# Confusion matrix
cm = confusion_matrix(df2_y_test, df2_BERT_LR_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Saved model and preds
np.savetxt("df2_BERT_LR_pred.csv", df2_BERT_LR_pred, delimiter=",")
joblib.dump(gridcv_LR, 'df2_BERT_LR.sav')

#######################################DistilBERT

#Training and gridsearchcv
start_train = timeit.default_timer()
df2_DistilBERT_LR = Pipeline([
    ("embedding", HFTransformersLanguage("distilbert-base-uncased")),
    ("model", LogisticRegression())
])
gridcv_LR = GridSearchCV(df2_DistilBERT_LR, param_grid = grid_LR, scoring='f1_micro', cv= 5, n_jobs = -1, verbose = 2, refit = True)
gridcv_LR.fit(df2_X_train, df2_y_train)
stop_train = timeit.default_timer()

# Prediction
start_predict = timeit.default_timer()
df2_DistilBERT_LR_pred = gridcv_LR.predict(df2_X_test)
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
print(classification_report(df2_y_test, df2_DistilBERT_LR_pred))
print("Accuracy: ", accuracy_score(df2_y_test, df2_DistilBERT_LR_pred))
print("f1_score: ", f1_score(df2_y_test, df2_DistilBERT_LR_pred, average='micro'))
print("precision_score: ", precision_score(df2_y_test, df2_DistilBERT_LR_pred, average='micro'))
print("recall_score: ", recall_score(df2_y_test, df2_DistilBERT_LR_pred, average='micro'))

# Computational Cost
print('Train Time: ', stop_train - start_train, 'Seconds')
print('Prediction Time : ', stop_predict - start_predict, 'Seconds')
print('Inference Time: ', (stop_predict - start_predict) * 1000 / len(df2_X_test), 'Milliseconds')

# Confusion matrix
cm = confusion_matrix(df2_y_test, df2_DistilBERT_LR_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Saved model and preds
np.savetxt("df2_DistilBERT_LR_pred.csv", df2_DistilBERT_LR_pred, delimiter=",")
joblib.dump(gridcv_LR, 'df2_DistilBERT_LR.sav')

#######################################ALBERT

#Training and gridsearchcv
start_train = timeit.default_timer()
df2_ALBERT_LR = Pipeline([
    ("embedding", HFTransformersLanguage("albert-base-v2")),
    ("model", LogisticRegression())
])
gridcv_LR = GridSearchCV(df2_ALBERT_LR, param_grid = grid_LR, scoring='f1_micro', cv= 5, n_jobs = -1, verbose = 2, refit = True)
gridcv_LR.fit(df2_X_train, df2_y_train)
stop_train = timeit.default_timer()

# Prediction
start_predict = timeit.default_timer()
df2_ALBERT_LR_pred = gridcv_LR.predict(df2_X_test)
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
print(classification_report(df2_y_test, df2_ALBERT_LR_pred))
print("Accuracy: ", accuracy_score(df2_y_test, df2_ALBERT_LR_pred))
print("f1_score: ", f1_score(df2_y_test, df2_ALBERT_LR_pred, average='micro'))
print("precision_score: ", precision_score(df2_y_test, df2_ALBERT_LR_pred, average='micro'))
print("recall_score: ", recall_score(df2_y_test, df2_ALBERT_LR_pred, average='micro'))

# Computational Cost
print('Train Time: ', stop_train - start_train, 'Seconds')
print('Prediction Time : ', stop_predict - start_predict, 'Seconds')
print('Inference Time: ', (stop_predict - start_predict) * 1000 / len(df2_X_test), 'Milliseconds')

# Confusion matrix
cm = confusion_matrix(df2_y_test, df2_ALBERT_LR_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Saved model and preds
np.savetxt("df2_ALBERT_LR_pred.csv", df2_ALBERT_LR_pred, delimiter=",")
joblib.dump(gridcv_LR, 'df2_ALBERT_LR.sav')

#######################################BART

#Training and gridsearchcv
start_train = timeit.default_timer()
df2_BART_LR = Pipeline([
    ("embedding", HFTransformersLanguage("facebook/bart-base")),
    ("model", LogisticRegression())
])
gridcv_LR = GridSearchCV(df2_BART_LR, param_grid = grid_LR, scoring='f1_micro', cv= 5, n_jobs = -1, verbose = 2, refit = True)
gridcv_LR.fit(df2_X_train, df2_y_train)
stop_train = timeit.default_timer()

# Prediction
start_predict = timeit.default_timer()
df2_BART_LR_pred = gridcv_LR.predict(df2_X_test)
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
print(classification_report(df2_y_test, df2_BART_LR_pred))
print("Accuracy: ", accuracy_score(df2_y_test, df2_BART_LR_pred))
print("f1_score: ", f1_score(df2_y_test, df2_BART_LR_pred, average='micro'))
print("precision_score: ", precision_score(df2_y_test, df2_BART_LR_pred, average='micro'))
print("recall_score: ", recall_score(df2_y_test, df2_BART_LR_pred, average='micro'))

# Computational Cost
print('Train Time: ', stop_train - start_train, 'Seconds')
print('Prediction Time : ', stop_predict - start_predict, 'Seconds')
print('Inference Time: ', (stop_predict - start_predict) * 1000 / len(df2_X_test), 'Milliseconds')

# Confusion matrix
cm = confusion_matrix(df2_y_test, df2_BART_LR_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Saved model and preds
np.savetxt("df2_BART_LR_pred.csv", df2_BART_LR_pred, delimiter=",")
joblib.dump(gridcv_LR, 'df2_BART_LR.sav')

#######################################RoBERTa

#Training and gridsearchcv
start_train = timeit.default_timer()
df2_RoBERTa_LR = Pipeline([
    ("embedding", HFTransformersLanguage("roberta-base")),
    ("model", LogisticRegression())
])
gridcv_LR = GridSearchCV(df2_RoBERTa_LR, param_grid = grid_LR, scoring='f1_micro', cv= 5, n_jobs = -1, verbose = 2, refit = True)
gridcv_LR.fit(df2_X_train, df2_y_train)
stop_train = timeit.default_timer()

# Prediction
start_predict = timeit.default_timer()
df2_RoBERTa_LR_pred = gridcv_LR.predict(df2_X_test)
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
print(classification_report(df2_y_test, df2_RoBERTa_LR_pred))
print("Accuracy: ", accuracy_score(df2_y_test, df2_RoBERTa_LR_pred))
print("f1_score: ", f1_score(df2_y_test, df2_RoBERTa_LR_pred, average='micro'))
print("precision_score: ", precision_score(df2_y_test, df2_RoBERTa_LR_pred, average='micro'))
print("recall_score: ", recall_score(df2_y_test, df2_RoBERTa_LR_pred, average='micro'))

# Computational Cost
print('Train Time: ', stop_train - start_train, 'Seconds')
print('Prediction Time : ', stop_predict - start_predict, 'Seconds')
print('Inference Time: ', (stop_predict - start_predict) * 1000 / len(df2_X_test), 'Milliseconds')

# Confusion matrix
cm = confusion_matrix(df2_y_test, df2_RoBERTa_LR_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Saved model and preds
np.savetxt("df2_RoBERTa_LR_pred.csv", df2_RoBERTa_LR_pred, delimiter=",")
joblib.dump(gridcv_LR, 'df2_RoBERTa_LR.sav')

#######################################ELECTRA

#Training and gridsearchcv
start_train = timeit.default_timer()
df2_ELECTRA_LR = Pipeline([
    ("embedding", HFTransformersLanguage("google/electra-small-discriminator")),
    ("model", LogisticRegression())
])
gridcv_LR = GridSearchCV(df2_ELECTRA_LR, param_grid = grid_LR, scoring='f1_micro', cv= 5, n_jobs = -1, verbose = 2, refit = True)
gridcv_LR.fit(df2_X_train, df2_y_train)
stop_train = timeit.default_timer()

# Prediction
start_predict = timeit.default_timer()
df2_ELECTRA_LR_pred = gridcv_LR.predict(df2_X_test)
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
print(classification_report(df2_y_test, df2_ELECTRA_LR_pred))
print("Accuracy: ", accuracy_score(df2_y_test, df2_ELECTRA_LR_pred))
print("f1_score: ", f1_score(df2_y_test, df2_ELECTRA_LR_pred, average='micro'))
print("precision_score: ", precision_score(df2_y_test, df2_ELECTRA_LR_pred, average='micro'))
print("recall_score: ", recall_score(df2_y_test, df2_ELECTRA_LR_pred, average='micro'))

# Computational Cost
print('Train Time: ', stop_train - start_train, 'Seconds')
print('Prediction Time : ', stop_predict - start_predict, 'Seconds')
print('Inference Time: ', (stop_predict - start_predict) * 1000 / len(df2_X_test), 'Milliseconds')

# Confusion matrix
cm = confusion_matrix(df2_y_test, df2_ELECTRA_LR_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Saved model and preds
np.savetxt("df2_ELECTRA_LR_pred.csv", df2_ELECTRA_LR_pred, delimiter=",")
joblib.dump(gridcv_LR, 'df2_ELECTRA_LR.sav')

#######################################XLNet

#Training and gridsearchcv
start_train = timeit.default_timer()
df2_XLNET_LR = Pipeline([
    ("embedding", HFTransformersLanguage("xlnet-base-cased")),
    ("model", LogisticRegression())
])
gridcv_LR = GridSearchCV(df2_XLNET_LR, param_grid = grid_LR, scoring='f1_micro', cv= 5, n_jobs = -1, verbose = 2, refit = True)
gridcv_LR.fit(df2_X_train, df2_y_train)
stop_train = timeit.default_timer()

# Prediction
start_predict = timeit.default_timer()
df2_XLNET_LR_pred = gridcv_LR.predict(df2_X_test)
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
print(classification_report(df2_y_test, df2_XLNET_LR_pred))
print("Accuracy: ", accuracy_score(df2_y_test, df2_XLNET_LR_pred))
print("f1_score: ", f1_score(df2_y_test, df2_XLNET_LR_pred, average='micro'))
print("precision_score: ", precision_score(df2_y_test, df2_XLNET_LR_pred, average='micro'))
print("recall_score: ", recall_score(df2_y_test, df2_XLNET_LR_pred, average='micro'))

# Computational Cost
print('Train Time: ', stop_train - start_train, 'Seconds')
print('Prediction Time : ', stop_predict - start_predict, 'Seconds')
print('Inference Time: ', (stop_predict - start_predict) * 1000 / len(df2_X_test), 'Milliseconds')

# Confusion matrix
cm = confusion_matrix(df2_y_test, df2_XLNET_LR_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Saved model and preds
np.savetxt("df2_XLNET_LR_pred.csv", df2_XLNET_LR_pred, delimiter=",")
joblib.dump(gridcv_LR, 'df2_XLNET_LR.sav')

