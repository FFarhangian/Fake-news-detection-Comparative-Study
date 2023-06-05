from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import matplotlib
import timeit

df4_baseline_model = DummyClassifier(strategy='most_frequent')

start_train = timeit.default_timer()
df4_baseline_model.fit(df4_train, df4_train_class)
stop_train = timeit.default_timer()

start_predict = timeit.default_timer()
df4_pred = df4_baseline_model.predict(df4_test)
stop_predict = timeit.default_timer()

#Prediction Results
print(classification_report(df4_test_class, df4_pred))
print("Accuracy: ", accuracy_score(df4_test_class, df4_pred))
print("f1_score: ", f1_score(df4_test_class, df4_pred, average='micro'))
print("precision_score: ", precision_score(df4_test_class, df4_pred, average='micro'))
print("recall_score: ", recall_score(df4_test_class, df4_pred, average='micro'))

#Computational Cost
print('Train Time: ', stop_train - start_train)
print('Prediction Time : ', stop_predict - start_predict)
print('Inference Time: ', (stop_predict - start_predict)/df4_test.shape[0])

#Confusion matrix
cm = confusion_matrix(df4_test_class, df4_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

#Saved model and preds
np.savetxt("df4_pred.csv", df4_pred, delimiter=",")
joblib.dump(df4_baseline_model, 'df4_baseline_model.sav')

