from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import matplotlib
import timeit

df1_baseline_model = DummyClassifier(strategy='most_frequent')

start_train = timeit.default_timer()
df1_baseline_model.fit(df1_train, df1_train_class)
stop_train = timeit.default_timer()

start_predict = timeit.default_timer()
df1_pred = df1_baseline_model.predict(df1_test)
stop_predict = timeit.default_timer()

#Prediction Results
print(classification_report(df1_test_class, df1_pred))
print("Accuracy: ", accuracy_score(df1_test_class, df1_pred))
print("f1_score: ", f1_score(df1_test_class, df1_pred, average='micro'))
print("precision_score: ", precision_score(df1_test_class, df1_pred, average='micro'))
print("recall_score: ", recall_score(df1_test_class, df1_pred, average='micro'))

#Computational Cost
print('Train Time: ', stop_train - start_train)
print('Prediction Time : ', stop_predict - start_predict)
print('Inference Time: ', (stop_predict - start_predict)/df1_test.shape[0])

#Confusion matrix
cm = confusion_matrix(df1_test_class, df1_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

#Saved model and preds
np.savetxt("df1_pred.csv", df1_pred, delimiter=",")
joblib.dump(df1_baseline_model, 'df1_baseline_model.sav')

