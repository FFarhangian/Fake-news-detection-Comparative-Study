from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import matplotlib
import timeit

df2_baseline_model = DummyClassifier(strategy='most_frequent')

start_train = timeit.default_timer()
df2_baseline_model.fit(df2_train, df2_train_class)
stop_train = timeit.default_timer()

start_predict = timeit.default_timer()
df2_pred = df2_baseline_model.predict(df2_test)
stop_predict = timeit.default_timer()

#Prediction Results
print(classification_report(df2_test_class, df2_pred))
print("Accuracy: ", accuracy_score(df2_test_class, df2_pred))
print("f1_score: ", f1_score(df2_test_class, df2_pred, average='micro'))
print("precision_score: ", precision_score(df2_test_class, df2_pred, average='micro'))
print("recall_score: ", recall_score(df2_test_class, df2_pred, average='micro'))

#Computational Cost
print('Train Time: ', stop_train - start_train)
print('Prediction Time : ', stop_predict - start_predict)
print('Inference Time: ', (stop_predict - start_predict)/df2_test.shape[0])

#Confusion matrix
cm = confusion_matrix(df2_test_class, df2_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

#Saved model and preds
np.savetxt("df2_pred.csv", df2_pred, delimiter=",")
joblib.dump(df2_baseline_model, 'df2_baseline_model.sav')

