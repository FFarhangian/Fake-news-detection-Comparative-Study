from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import matplotlib
import timeit

df3_baseline_model = DummyClassifier(strategy='most_frequent')

start_train = timeit.default_timer()
df3_baseline_model.fit(df3_train, df3_train_class)
stop_train = timeit.default_timer()

start_predict = timeit.default_timer()
df3_pred = df3_baseline_model.predict(df3_test)
stop_predict = timeit.default_timer()

#Prediction Results
print(classification_report(df3_test_class, df3_pred))
print("Accuracy: ", accuracy_score(df3_test_class, df3_pred))
print("f1_score: ", f1_score(df3_test_class, df3_pred, average='micro'))
print("precision_score: ", precision_score(df3_test_class, df3_pred, average='micro'))
print("recall_score: ", recall_score(df3_test_class, df3_pred, average='micro'))

#Computational Cost
print('Train Time: ', stop_train - start_train)
print('Prediction Time : ', stop_predict - start_predict)
print('Inference Time: ', (stop_predict - start_predict)/df3_test.shape[0])

#Confusion matrix
cm = confusion_matrix(df3_test_class, df3_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

#Saved model and preds
np.savetxt("df3_pred.csv", df3_pred, delimiter=",")
joblib.dump(df3_baseline_model, 'df3_baseline_model.sav')
