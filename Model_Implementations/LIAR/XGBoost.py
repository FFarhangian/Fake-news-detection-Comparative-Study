import joblib
import timeit
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, KFold, PredefinedSplit, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.XGBoost import SVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, XGBoostClassifier
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
from sklearn.XGBoost import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, XGBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb


# Before using transformers make sure that the data convert to list
df1_X_train = df1_train["Text"].tolist()
df1_y_train = df1_train["Label"].tolist()
df1_X_test = df1_test["Text"].tolist()
df1_y_test = df1_test["Label"].tolist()

#######################################TF
#Training and gridsearchcv
start_train = timeit.default_timer()
gridcv_XGBoost.fit(TF_df1_train, df1_train_class)
stop_train = timeit.default_timer()

#Prediction
start_predict = timeit.default_timer()
TF_df1_pred = gridcv_XGBoost.predict(TF_df1_test)
stop_predict = timeit.default_timer()

#Training Results
cv_results = gridcv_XGBoost.cv_results_['split0_test_score'],
gridcv_XGBoost.cv_results_['split1_test_score'],
gridcv_XGBoost.cv_results_['split2_test_score'],
gridcv_XGBoost.cv_results_['split3_test_score'],
gridcv_XGBoost.cv_results_['split4_test_score']
mean_score = np.mean(cv_results)
std_score = np.std(cv_results)

# Print the mean and standard deviation of each fold's performance
print("Mean cross-validation score: {:.2f}".format(mean_score))
print("Standard deviation of cross-validation score: {:.2f}".format(std_score))

# Print the best hyperparameters found by GridSearchCV
print("Best hyperparameters: {}".format(gridcv_XGBoost.best_params_))
print("Best score: {}".format(gridcv_XGBoost.best_score_))

#Prediction Results
print(classification_report(df1_test_class, TF_df1_pred))
print("Accuracy: ", accuracy_score(df1_test_class, TF_df1_pred))
print("f1_score: ", f1_score(df1_test_class, TF_df1_pred, average='micro'))
print("precision_score: ", precision_score(df1_test_class, TF_df1_pred, average='micro'))
print("recall_score: ", recall_score(df1_test_class, TF_df1_pred, average='micro'))

#Computational Cost
print('Train Time: ', stop_train - start_train)
print('Prediction Time : ', stop_predict - start_predict)
print('Inference Time: ', (stop_predict - start_predict)/TF_df1_test.shape[0])

#Confusion matrix
cm = confusion_matrix(df1_test_class, TF_df1_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

#Saved model and preds
np.savetxt("df1_XGBoost_TF_pred.csv", TF_df1_pred, delimiter=",")
joblib.dump(gridcv_XGBoost, 'df1_TF_XGBoost.sav')

#######################################TFIDF
#Training and gridsearchcv
start_train = timeit.default_timer()
gridcv_XGBoost.fit(TFIDF_df1_train, df1_train_class)
stop_train = timeit.default_timer()

#Prediction
start_predict = timeit.default_timer()
TFIDF_df1_pred = gridcv_XGBoost.predict(TFIDF_df1_test)
stop_predict = timeit.default_timer()

#Training Results
cv_results = gridcv_XGBoost.cv_results_['split0_test_score'],
gridcv_XGBoost.cv_results_['split1_test_score'],
gridcv_XGBoost.cv_results_['split2_test_score'],
gridcv_XGBoost.cv_results_['split3_test_score'],
gridcv_XGBoost.cv_results_['split4_test_score']
mean_score = np.mean(cv_results)
std_score = np.std(cv_results)

# Print the mean and standard deviation of each fold's performance
print("Mean cross-validation score: {:.2f}".format(mean_score))
print("Standard deviation of cross-validation score: {:.2f}".format(std_score))

# Print the best hyperparameters found by GridSearchCV
print("Best hyperparameters: {}".format(gridcv_XGBoost.best_params_))
print("Best score: {}".format(gridcv_XGBoost.best_score_))

#Prediction Results
print(classification_report(df1_test_class, TFIDF_df1_pred))
print("Accuracy: ", accuracy_score(df1_test_class, TFIDF_df1_pred))
print("f1_score: ", f1_score(df1_test_class, TFIDF_df1_pred, average='micro'))
print("precision_score: ", precision_score(df1_test_class, TFIDF_df1_pred, average='micro'))
print("recall_score: ", recall_score(df1_test_class, TFIDF_df1_pred, average='micro'))

#Computational Cost
print('Train Time: ', stop_train - start_train)
print('Prediction Time : ', stop_predict - start_predict)
print('Inference Time: ', (stop_predict - start_predict)/TFIDF_df1_test.shape[0])

#Confusion matrix
cm = confusion_matrix(df1_test_class, TFIDF_df1_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

#Saved model and preds
np.savetxt("df1_XGBoost_TFIDF_pred.csv", TFIDF_df1_pred, delimiter=",")
joblib.dump(gridcv_XGBoost, 'df1_TFIDF_XGBoost.sav')

#######################################W2V
#Training and gridsearchcv
start_train = timeit.default_timer()
gridcv_XGBoost.fit(W2V_df1_train, df1_train_class)
stop_train = timeit.default_timer()

#Prediction
start_predict = timeit.default_timer()
W2V_df1_pred = gridcv_XGBoost.predict(W2V_df1_test)
stop_predict = timeit.default_timer()

#Training Results
cv_results = gridcv_XGBoost.cv_results_['split0_test_score'],
gridcv_XGBoost.cv_results_['split1_test_score'],
gridcv_XGBoost.cv_results_['split2_test_score'],
gridcv_XGBoost.cv_results_['split3_test_score'],
gridcv_XGBoost.cv_results_['split4_test_score']
mean_score = np.mean(cv_results)
std_score = np.std(cv_results)

# Print the mean and standard deviation of each fold's performance
print("Mean cross-validation score: {:.2f}".format(mean_score))
print("Standard deviation of cross-validation score: {:.2f}".format(std_score))

# Print the best hyperparameters found by GridSearchCV
print("Best hyperparameters: {}".format(gridcv_XGBoost.best_params_))
print("Best score: {}".format(gridcv_XGBoost.best_score_))

#Prediction Results
print(classification_report(df1_test_class, W2V_df1_pred))
print("Accuracy: ", accuracy_score(df1_test_class, W2V_df1_pred))
print("f1_score: ", f1_score(df1_test_class, W2V_df1_pred, average='micro'))
print("precision_score: ", precision_score(df1_test_class, W2V_df1_pred, average='micro'))
print("recall_score: ", recall_score(df1_test_class, W2V_df1_pred, average='micro'))

#Computational Cost
print('Train Time: ', stop_train - start_train)
print('Prediction Time : ', stop_predict - start_predict)
print('Inference Time: ', (stop_predict - start_predict)/W2V_df1_test.shape[0])

#Confusion matrix
cm = confusion_matrix(df1_test_class, W2V_df1_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

#Saved model and preds
np.savetxt("df1_XGBoost_W2V_pred.csv", W2V_df1_pred, delimiter=",")
joblib.dump(gridcv_XGBoost, 'df1_W2V_XGBoost.sav')

#######################################Glove
#Training and gridsearchcv
start_train = timeit.default_timer()
gridcv_XGBoost.fit(Glove_df1_train, df1_train_class)
stop_train = timeit.default_timer()

#Prediction
start_predict = timeit.default_timer()
Glove_df1_pred = gridcv_XGBoost.predict(Glove_df1_test)
stop_predict = timeit.default_timer()

#Training Results
cv_results = gridcv_XGBoost.cv_results_['split0_test_score'],
gridcv_XGBoost.cv_results_['split1_test_score'],
gridcv_XGBoost.cv_results_['split2_test_score'],
gridcv_XGBoost.cv_results_['split3_test_score'],
gridcv_XGBoost.cv_results_['split4_test_score']
mean_score = np.mean(cv_results)
std_score = np.std(cv_results)

# Print the mean and standard deviation of each fold's performance
print("Mean cross-validation score: {:.2f}".format(mean_score))
print("Standard deviation of cross-validation score: {:.2f}".format(std_score))

# Print the best hyperparameters found by GridSearchCV
print("Best hyperparameters: {}".format(gridcv_XGBoost.best_params_))
print("Best score: {}".format(gridcv_XGBoost.best_score_))

#Prediction Results
print(classification_report(df1_test_class, Glove_df1_pred))
print("Accuracy: ", accuracy_score(df1_test_class, Glove_df1_pred))
print("f1_score: ", f1_score(df1_test_class, Glove_df1_pred, average='micro'))
print("precision_score: ", precision_score(df1_test_class, Glove_df1_pred, average='micro'))
print("recall_score: ", recall_score(df1_test_class, Glove_df1_pred, average='micro'))

#Computational Cost
print('Train Time: ', stop_train - start_train)
print('Prediction Time : ', stop_predict - start_predict)
print('Inference Time: ', (stop_predict - start_predict)/Glove_df1_test.shape[0])

#Confusion matrix
cm = confusion_matrix(df1_test_class, Glove_df1_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

#Saved model and preds
np.savetxt("df1_XGBoost_Glove_pred.csv", Glove_df1_pred, delimiter=",")
joblib.dump(gridcv_XGBoost, 'df1_Glove_XGBoost.sav')

#######################################Fasttext
#Training and gridsearchcv
start_train = timeit.default_timer()
gridcv_XGBoost.fit(Fasttext_df1_train, df1_train_class)
stop_train = timeit.default_timer()

#Prediction
start_predict = timeit.default_timer()
Fasttext_df1_pred = gridcv_XGBoost.predict(Fasttext_df1_test)
stop_predict = timeit.default_timer()

#Training Results
cv_results = gridcv_XGBoost.cv_results_['split0_test_score'],
gridcv_XGBoost.cv_results_['split1_test_score'],
gridcv_XGBoost.cv_results_['split2_test_score'],
gridcv_XGBoost.cv_results_['split3_test_score'],
gridcv_XGBoost.cv_results_['split4_test_score']
mean_score = np.mean(cv_results)
std_score = np.std(cv_results)

# Print the mean and standard deviation of each fold's performance
print("Mean cross-validation score: {:.2f}".format(mean_score))
print("Standard deviation of cross-validation score: {:.2f}".format(std_score))

# Print the best hyperparameters found by GridSearchCV
print("Best hyperparameters: {}".format(gridcv_XGBoost.best_params_))
print("Best score: {}".format(gridcv_XGBoost.best_score_))

#Prediction Results
print(classification_report(df1_test_class, Fasttext_df1_pred))
print("Accuracy: ", accuracy_score(df1_test_class, Fasttext_df1_pred))
print("f1_score: ", f1_score(df1_test_class, Fasttext_df1_pred, average='micro'))
print("precision_score: ", precision_score(df1_test_class, Fasttext_df1_pred, average='micro'))
print("recall_score: ", recall_score(df1_test_class, Fasttext_df1_pred, average='micro'))

#Computational Cost
print('Train Time: ', stop_train - start_train)
print('Prediction Time : ', stop_predict - start_predict)
print('Inference Time: ', (stop_predict - start_predict)/Fasttext_df1_test.shape[0])

#Confusion matrix
cm = confusion_matrix(df1_test_class, Fasttext_df1_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

#Saved model and preds
np.savetxt("df1_XGBoost_Fasttext_pred.csv", Fasttext_df1_pred, delimiter=",")
joblib.dump(gridcv_XGBoost, 'df1_Fasttext_XGBoost.sav')

#######################################ELMO

#Training and gridsearchcv
start_train = timeit.default_timer()
gridcv_XGBoost.fit(ELMO_df1_train, df1_train_class)
stop_train = timeit.default_timer()

#Prediction
start_predict = timeit.default_timer()
ELMO_df1_pred = gridcv_XGBoost.predict(ELMO_df1_test)
stop_predict = timeit.default_timer()

#Training Results
cv_results = gridcv_XGBoost.cv_results_['split0_test_score'],
gridcv_XGBoost.cv_results_['split1_test_score'],
gridcv_XGBoost.cv_results_['split2_test_score'],
gridcv_XGBoost.cv_results_['split3_test_score'],
gridcv_XGBoost.cv_results_['split4_test_score']
mean_score = np.mean(cv_results)
std_score = np.std(cv_results)

# Print the mean and standard deviation of each fold's performance
print("Mean cross-validation score: {:.2f}".format(mean_score))
print("Standard deviation of cross-validation score: {:.2f}".format(std_score))

# Print the best hyperparameters found by GridSearchCV
print("Best hyperparameters: {}".format(gridcv_XGBoost.best_params_))
print("Best score: {}".format(gridcv_XGBoost.best_score_))

#Prediction Results
print(classification_report(df1_test_class, ELMO_df1_pred))
print("Accuracy: ", accuracy_score(df1_test_class, ELMO_df1_pred))
print("f1_score: ", f1_score(df1_test_class, ELMO_df1_pred, average='micro'))
print("precision_score: ", precision_score(df1_test_class, ELMO_df1_pred, average='micro'))
print("recall_score: ", recall_score(df1_test_class, ELMO_df1_pred, average='micro'))

#Computational Cost
print('Train Time: ', stop_train - start_train)
print('Prediction Time : ', stop_predict - start_predict)
print('Inference Time: ', (stop_predict - start_predict)/ELMO_df1_test.shape[0])

#Confusion matrix
cm = confusion_matrix(df1_test_class, ELMO_df1_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

#Saved model and preds
np.savetxt("df1_ELMO_XGBoost_pred.csv", ELMO_df1_pred, delimiter=",")
joblib.dump(gridcv_XGBoost, 'df1_ELMO_XGBoost.sav')

#######################################BERT

#Training and gridsearchcv
start_train = timeit.default_timer()
df1_BERT_XGBoost = Pipeline([
    ("embedding", HFTransformersLanguage("bert-base-uncased")),
    ("model", xgb.XGBClassifier())
])
gridcv_XGBoost = GridSearchCV(df1_BERT_XGBoost, param_grid = grid_XGBoost, scoring='f1_micro', cv= 5, n_jobs = -1, verbose = 2, refit = True)
gridcv_XGBoost.fit(df1_X_train, df1_y_train)
stop_train = timeit.default_timer()

# Prediction
start_predict = timeit.default_timer()
df1_BERT_XGBoost_pred = gridcv_XGBoost.predict(df1_X_test)
stop_predict = timeit.default_timer()

# Training Results
cv_results = gridcv_XGBoost.cv_results_['split0_test_score'],
gridcv_XGBoost.cv_results_['split1_test_score'],
gridcv_XGBoost.cv_results_['split2_test_score'],
gridcv_XGBoost.cv_results_['split3_test_score'],
gridcv_XGBoost.cv_results_['split4_test_score']
mean_score = np.mean(cv_results)
std_score = np.std(cv_results)

# Print the mean and standard deviation of each fold's performance
print("Mean cross-validation score: {:.2f}".format(mean_score))
print("Standard deviation of cross-validation score: {:.2f}".format(std_score))

# Print the best hyperparameters found by GridSearchCV
print("Best hyperparameters: {}".format(gridcv_XGBoost.best_params_))
print("Best score: {}".format(gridcv_XGBoost.best_score_))

# Prediction Results
print(classification_report(df1_y_test, df1_BERT_XGBoost_pred))
print("Accuracy: ", accuracy_score(df1_y_test, df1_BERT_XGBoost_pred))
print("f1_score: ", f1_score(df1_y_test, df1_BERT_XGBoost_pred, average='micro'))
print("precision_score: ", precision_score(df1_y_test, df1_BERT_XGBoost_pred, average='micro'))
print("recall_score: ", recall_score(df1_y_test, df1_BERT_XGBoost_pred, average='micro'))

# Computational Cost
print('Train Time: ', stop_train - start_train, 'Seconds')
print('Prediction Time : ', stop_predict - start_predict, 'Seconds')
print('Inference Time: ', (stop_predict - start_predict) * 1000 / len(df1_X_test), 'Milliseconds')

# Confusion matrix
cm = confusion_matrix(df1_y_test, df1_BERT_XGBoost_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Saved model and preds
np.savetxt("df1_BERT_XGBoost_pred.csv", df1_BERT_XGBoost_pred, delimiter=",")
joblib.dump(gridcv_XGBoost, 'df1_BERT_XGBoost.sav')

#######################################DistilBERT

#Training and gridsearchcv
start_train = timeit.default_timer()
df1_DistilBERT_XGBoost = Pipeline([
    ("embedding", HFTransformersLanguage("distilbert-base-uncased")),
    ("model", xgb.XGBClassifier())
])
gridcv_XGBoost = GridSearchCV(df1_DistilBERT_XGBoost, param_grid = grid_XGBoost, scoring='f1_micro', cv= 5, n_jobs = -1, verbose = 2, refit = True)
gridcv_XGBoost.fit(df1_X_train, df1_y_train)
stop_train = timeit.default_timer()

# Prediction
start_predict = timeit.default_timer()
df1_DistilBERT_XGBoost_pred = gridcv_XGBoost.predict(df1_X_test)
stop_predict = timeit.default_timer()

# Training Results
cv_results = gridcv_XGBoost.cv_results_['split0_test_score'],
gridcv_XGBoost.cv_results_['split1_test_score'],
gridcv_XGBoost.cv_results_['split2_test_score'],
gridcv_XGBoost.cv_results_['split3_test_score'],
gridcv_XGBoost.cv_results_['split4_test_score']
mean_score = np.mean(cv_results)
std_score = np.std(cv_results)

# Print the mean and standard deviation of each fold's performance
print("Mean cross-validation score: {:.2f}".format(mean_score))
print("Standard deviation of cross-validation score: {:.2f}".format(std_score))

# Print the best hyperparameters found by GridSearchCV
print("Best hyperparameters: {}".format(gridcv_XGBoost.best_params_))
print("Best score: {}".format(gridcv_XGBoost.best_score_))

# Prediction Results
print(classification_report(df1_y_test, df1_DistilBERT_XGBoost_pred))
print("Accuracy: ", accuracy_score(df1_y_test, df1_DistilBERT_XGBoost_pred))
print("f1_score: ", f1_score(df1_y_test, df1_DistilBERT_XGBoost_pred, average='micro'))
print("precision_score: ", precision_score(df1_y_test, df1_DistilBERT_XGBoost_pred, average='micro'))
print("recall_score: ", recall_score(df1_y_test, df1_DistilBERT_XGBoost_pred, average='micro'))

# Computational Cost
print('Train Time: ', stop_train - start_train, 'Seconds')
print('Prediction Time : ', stop_predict - start_predict, 'Seconds')
print('Inference Time: ', (stop_predict - start_predict) * 1000 / len(df1_X_test), 'Milliseconds')

# Confusion matrix
cm = confusion_matrix(df1_y_test, df1_DistilBERT_XGBoost_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Saved model and preds
np.savetxt("df1_DistilBERT_XGBoost_pred.csv", df1_DistilBERT_XGBoost_pred, delimiter=",")
joblib.dump(gridcv_XGBoost, 'df1_DistilBERT_XGBoost.sav')

#######################################ALBERT

#Training and gridsearchcv
start_train = timeit.default_timer()
df1_ALBERT_XGBoost = Pipeline([
    ("embedding", HFTransformersLanguage("albert-base-v2")),
    ("model", xgb.XGBClassifier())
])
gridcv_XGBoost = GridSearchCV(df1_ALBERT_XGBoost, param_grid = grid_XGBoost, scoring='f1_micro', cv= 5, n_jobs = -1, verbose = 2, refit = True)
gridcv_XGBoost.fit(df1_X_train, df1_y_train)
stop_train = timeit.default_timer()

# Prediction
start_predict = timeit.default_timer()
df1_ALBERT_XGBoost_pred = gridcv_XGBoost.predict(df1_X_test)
stop_predict = timeit.default_timer()

# Training Results
cv_results = gridcv_XGBoost.cv_results_['split0_test_score'],
gridcv_XGBoost.cv_results_['split1_test_score'],
gridcv_XGBoost.cv_results_['split2_test_score'],
gridcv_XGBoost.cv_results_['split3_test_score'],
gridcv_XGBoost.cv_results_['split4_test_score']
mean_score = np.mean(cv_results)
std_score = np.std(cv_results)

# Print the mean and standard deviation of each fold's performance
print("Mean cross-validation score: {:.2f}".format(mean_score))
print("Standard deviation of cross-validation score: {:.2f}".format(std_score))

# Print the best hyperparameters found by GridSearchCV
print("Best hyperparameters: {}".format(gridcv_XGBoost.best_params_))
print("Best score: {}".format(gridcv_XGBoost.best_score_))

# Prediction Results
print(classification_report(df1_y_test, df1_ALBERT_XGBoost_pred))
print("Accuracy: ", accuracy_score(df1_y_test, df1_ALBERT_XGBoost_pred))
print("f1_score: ", f1_score(df1_y_test, df1_ALBERT_XGBoost_pred, average='micro'))
print("precision_score: ", precision_score(df1_y_test, df1_ALBERT_XGBoost_pred, average='micro'))
print("recall_score: ", recall_score(df1_y_test, df1_ALBERT_XGBoost_pred, average='micro'))

# Computational Cost
print('Train Time: ', stop_train - start_train, 'Seconds')
print('Prediction Time : ', stop_predict - start_predict, 'Seconds')
print('Inference Time: ', (stop_predict - start_predict) * 1000 / len(df1_X_test), 'Milliseconds')

# Confusion matrix
cm = confusion_matrix(df1_y_test, df1_ALBERT_XGBoost_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Saved model and preds
np.savetxt("df1_ALBERT_XGBoost_pred.csv", df1_ALBERT_XGBoost_pred, delimiter=",")
joblib.dump(gridcv_XGBoost, 'df1_ALBERT_XGBoost.sav')

#######################################BART

#Training and gridsearchcv
start_train = timeit.default_timer()
df1_BART_XGBoost = Pipeline([
    ("embedding", HFTransformersLanguage("facebook/bart-base")),
    ("model", xgb.XGBClassifier())
])
gridcv_XGBoost = GridSearchCV(df1_BART_XGBoost, param_grid = grid_XGBoost, scoring='f1_micro', cv= 5, n_jobs = -1, verbose = 2, refit = True)
gridcv_XGBoost.fit(df1_X_train, df1_y_train)
stop_train = timeit.default_timer()

# Prediction
start_predict = timeit.default_timer()
df1_BART_XGBoost_pred = gridcv_XGBoost.predict(df1_X_test)
stop_predict = timeit.default_timer()

# Training Results
cv_results = gridcv_XGBoost.cv_results_['split0_test_score'],
gridcv_XGBoost.cv_results_['split1_test_score'],
gridcv_XGBoost.cv_results_['split2_test_score'],
gridcv_XGBoost.cv_results_['split3_test_score'],
gridcv_XGBoost.cv_results_['split4_test_score']
mean_score = np.mean(cv_results)
std_score = np.std(cv_results)

# Print the mean and standard deviation of each fold's performance
print("Mean cross-validation score: {:.2f}".format(mean_score))
print("Standard deviation of cross-validation score: {:.2f}".format(std_score))

# Print the best hyperparameters found by GridSearchCV
print("Best hyperparameters: {}".format(gridcv_XGBoost.best_params_))
print("Best score: {}".format(gridcv_XGBoost.best_score_))

# Prediction Results
print(classification_report(df1_y_test, df1_BART_XGBoost_pred))
print("Accuracy: ", accuracy_score(df1_y_test, df1_BART_XGBoost_pred))
print("f1_score: ", f1_score(df1_y_test, df1_BART_XGBoost_pred, average='micro'))
print("precision_score: ", precision_score(df1_y_test, df1_BART_XGBoost_pred, average='micro'))
print("recall_score: ", recall_score(df1_y_test, df1_BART_XGBoost_pred, average='micro'))

# Computational Cost
print('Train Time: ', stop_train - start_train, 'Seconds')
print('Prediction Time : ', stop_predict - start_predict, 'Seconds')
print('Inference Time: ', (stop_predict - start_predict) * 1000 / len(df1_X_test), 'Milliseconds')

# Confusion matrix
cm = confusion_matrix(df1_y_test, df1_BART_XGBoost_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Saved model and preds
np.savetxt("df1_BART_XGBoost_pred.csv", df1_BART_XGBoost_pred, delimiter=",")
joblib.dump(gridcv_XGBoost, 'df1_BART_XGBoost.sav')

#######################################RoBERTa

#Training and gridsearchcv
start_train = timeit.default_timer()
df1_RoBERTa_XGBoost = Pipeline([
    ("embedding", HFTransformersLanguage("roberta-base")),
    ("model", xgb.XGBClassifier())
])
gridcv_XGBoost = GridSearchCV(df1_RoBERTa_XGBoost, param_grid = grid_XGBoost, scoring='f1_micro', cv= 5, n_jobs = -1, verbose = 2, refit = True)
gridcv_XGBoost.fit(df1_X_train, df1_y_train)
stop_train = timeit.default_timer()

# Prediction
start_predict = timeit.default_timer()
df1_RoBERTa_XGBoost_pred = gridcv_XGBoost.predict(df1_X_test)
stop_predict = timeit.default_timer()

# Training Results
cv_results = gridcv_XGBoost.cv_results_['split0_test_score'],
gridcv_XGBoost.cv_results_['split1_test_score'],
gridcv_XGBoost.cv_results_['split2_test_score'],
gridcv_XGBoost.cv_results_['split3_test_score'],
gridcv_XGBoost.cv_results_['split4_test_score']
mean_score = np.mean(cv_results)
std_score = np.std(cv_results)

# Print the mean and standard deviation of each fold's performance
print("Mean cross-validation score: {:.2f}".format(mean_score))
print("Standard deviation of cross-validation score: {:.2f}".format(std_score))

# Print the best hyperparameters found by GridSearchCV
print("Best hyperparameters: {}".format(gridcv_XGBoost.best_params_))
print("Best score: {}".format(gridcv_XGBoost.best_score_))

# Prediction Results
print(classification_report(df1_y_test, df1_RoBERTa_XGBoost_pred))
print("Accuracy: ", accuracy_score(df1_y_test, df1_RoBERTa_XGBoost_pred))
print("f1_score: ", f1_score(df1_y_test, df1_RoBERTa_XGBoost_pred, average='micro'))
print("precision_score: ", precision_score(df1_y_test, df1_RoBERTa_XGBoost_pred, average='micro'))
print("recall_score: ", recall_score(df1_y_test, df1_RoBERTa_XGBoost_pred, average='micro'))

# Computational Cost
print('Train Time: ', stop_train - start_train, 'Seconds')
print('Prediction Time : ', stop_predict - start_predict, 'Seconds')
print('Inference Time: ', (stop_predict - start_predict) * 1000 / len(df1_X_test), 'Milliseconds')

# Confusion matrix
cm = confusion_matrix(df1_y_test, df1_RoBERTa_XGBoost_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Saved model and preds
np.savetxt("df1_RoBERTa_XGBoost_pred.csv", df1_RoBERTa_XGBoost_pred, delimiter=",")
joblib.dump(gridcv_XGBoost, 'df1_RoBERTa_XGBoost.sav')

#######################################ELECTRA

#Training and gridsearchcv
start_train = timeit.default_timer()
df1_ELECTRA_XGBoost = Pipeline([
    ("embedding", HFTransformersLanguage("google/electra-small-discriminator")),
    ("model", xgb.XGBClassifier())
])
gridcv_XGBoost = GridSearchCV(df1_ELECTRA_XGBoost, param_grid = grid_XGBoost, scoring='f1_micro', cv= 5, n_jobs = -1, verbose = 2, refit = True)
gridcv_XGBoost.fit(df1_X_train, df1_y_train)
stop_train = timeit.default_timer()

# Prediction
start_predict = timeit.default_timer()
df1_ELECTRA_XGBoost_pred = gridcv_XGBoost.predict(df1_X_test)
stop_predict = timeit.default_timer()

# Training Results
cv_results = gridcv_XGBoost.cv_results_['split0_test_score'],
gridcv_XGBoost.cv_results_['split1_test_score'],
gridcv_XGBoost.cv_results_['split2_test_score'],
gridcv_XGBoost.cv_results_['split3_test_score'],
gridcv_XGBoost.cv_results_['split4_test_score']
mean_score = np.mean(cv_results)
std_score = np.std(cv_results)

# Print the mean and standard deviation of each fold's performance
print("Mean cross-validation score: {:.2f}".format(mean_score))
print("Standard deviation of cross-validation score: {:.2f}".format(std_score))

# Print the best hyperparameters found by GridSearchCV
print("Best hyperparameters: {}".format(gridcv_XGBoost.best_params_))
print("Best score: {}".format(gridcv_XGBoost.best_score_))

# Prediction Results
print(classification_report(df1_y_test, df1_ELECTRA_XGBoost_pred))
print("Accuracy: ", accuracy_score(df1_y_test, df1_ELECTRA_XGBoost_pred))
print("f1_score: ", f1_score(df1_y_test, df1_ELECTRA_XGBoost_pred, average='micro'))
print("precision_score: ", precision_score(df1_y_test, df1_ELECTRA_XGBoost_pred, average='micro'))
print("recall_score: ", recall_score(df1_y_test, df1_ELECTRA_XGBoost_pred, average='micro'))

# Computational Cost
print('Train Time: ', stop_train - start_train, 'Seconds')
print('Prediction Time : ', stop_predict - start_predict, 'Seconds')
print('Inference Time: ', (stop_predict - start_predict) * 1000 / len(df1_X_test), 'Milliseconds')

# Confusion matrix
cm = confusion_matrix(df1_y_test, df1_ELECTRA_XGBoost_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Saved model and preds
np.savetxt("df1_ELECTRA_XGBoost_pred.csv", df1_ELECTRA_XGBoost_pred, delimiter=",")
joblib.dump(gridcv_XGBoost, 'df1_ELECTRA_XGBoost.sav')

#######################################XLNet

#Training and gridsearchcv
start_train = timeit.default_timer()
df1_XLNET_XGBoost = Pipeline([
    ("embedding", HFTransformersLanguage("xlnet-base-cased")),
    ("model", xgb.XGBClassifier())
])
gridcv_XGBoost = GridSearchCV(df1_XLNET_XGBoost, param_grid = grid_XGBoost, scoring='f1_micro', cv= 5, n_jobs = -1, verbose = 2, refit = True)
gridcv_XGBoost.fit(df1_X_train, df1_y_train)
stop_train = timeit.default_timer()

# Prediction
start_predict = timeit.default_timer()
df1_XLNET_XGBoost_pred = gridcv_XGBoost.predict(df1_X_test)
stop_predict = timeit.default_timer()

# Training Results
cv_results = gridcv_XGBoost.cv_results_['split0_test_score'],
gridcv_XGBoost.cv_results_['split1_test_score'],
gridcv_XGBoost.cv_results_['split2_test_score'],
gridcv_XGBoost.cv_results_['split3_test_score'],
gridcv_XGBoost.cv_results_['split4_test_score']
mean_score = np.mean(cv_results)
std_score = np.std(cv_results)

# Print the mean and standard deviation of each fold's performance
print("Mean cross-validation score: {:.2f}".format(mean_score))
print("Standard deviation of cross-validation score: {:.2f}".format(std_score))

# Print the best hyperparameters found by GridSearchCV
print("Best hyperparameters: {}".format(gridcv_XGBoost.best_params_))
print("Best score: {}".format(gridcv_XGBoost.best_score_))

# Prediction Results
print(classification_report(df1_y_test, df1_XLNET_XGBoost_pred))
print("Accuracy: ", accuracy_score(df1_y_test, df1_XLNET_XGBoost_pred))
print("f1_score: ", f1_score(df1_y_test, df1_XLNET_XGBoost_pred, average='micro'))
print("precision_score: ", precision_score(df1_y_test, df1_XLNET_XGBoost_pred, average='micro'))
print("recall_score: ", recall_score(df1_y_test, df1_XLNET_XGBoost_pred, average='micro'))

# Computational Cost
print('Train Time: ', stop_train - start_train, 'Seconds')
print('Prediction Time : ', stop_predict - start_predict, 'Seconds')
print('Inference Time: ', (stop_predict - start_predict) * 1000 / len(df1_X_test), 'Milliseconds')

# Confusion matrix
cm = confusion_matrix(df1_y_test, df1_XLNET_XGBoost_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Saved model and preds
np.savetxt("df1_XLNET_XGBoost_pred.csv", df1_XLNET_XGBoost_pred, delimiter=",")
joblib.dump(gridcv_XGBoost, 'df1_XLNET_XGBoost.sav')


