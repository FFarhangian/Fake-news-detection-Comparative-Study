from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb

#Parameters
grid_SVM = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    # 'kernel': ['rbf', 'linear', 'poly' , 'sigmoid']
    'kernel': ['rbf']
    }

grid_KNN = { 
    'n_neighbors': list(range(1,10))
}

grid_LR = { 
    'solver' : ['liblinear'],
    'penalty' : ['l1', 'l2'],
    'C' : [100, 10, 1.0, 0.1, 0.01]
}

grid_BNB = { 
    'alpha' : [0.1]
    # 'alpha' : [0.1, 0.5, 1],
    # 'fit_prior' : [True, False]
}

grid_MNB = { 
    # 'alpha' : [0.1]
    # 'alpha' : [0.1, 0.5, 1],
    # 'fit_prior' : [True, False],
}

# grid_DA = { 
#     'solver': ['svd', 'lsqr', 'eigen']
# }

# grid_DT = { 
#     'n_estimators': [10, 20, 50, 100]
# }

grid_RF = { 
    'bootstrap': [True, False],
    'max_depth': [5 ,10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
    'max_features': ['auto', 'sqrt', 'log2'],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10],
    'n_estimators': [10, 20, 50, 100, 200, 400, 600, 800, 1000],
    'criterion' :['gini', 'entropy']
}

grid_AdaBoost = { 
    'n_estimators': [10, 50, 100, 200, 300, 400, 500, 1000, 5000],
    'learning_rate' : [0.001, 0.01, 0.1, 0.2, 0.5]
}

grid_XGBoost = { 
    'n_estimators': [200,300,400,500],
    'max_features': ['sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy'],
    'random_state' : [18]
}



#Models
SVM = SVC()
LR = LogisticRegression()
KNN = KNeighborsClassifier()
MNB = GaussianNB()
BNB = BernoulliNB()
RF = RandomForestClassifier()
XGBoost = xgb.XGBClassifier()
AdaBoost = AdaBoostClassifier()

#Model Selection
gridcv_SVM = GridSearchCV(estimator = SVM, param_grid = grid_SVM, scoring='accuracy', cv= 5, n_jobs = -1, verbose = 2, refit = True )
gridcv_LR = GridSearchCV(estimator = LR, param_grid = grid_LR, scoring='accuracy', cv= 5, n_jobs = -1, verbose = 2, refit = True)
gridcv_KNN = GridSearchCV(estimator = KNN, param_grid = grid_KNN, scoring='accuracy', cv= 5, n_jobs = -1, verbose = 2, refit = True)
gridcv_MNB = GridSearchCV(estimator = MNB, param_grid = grid_MNB, scoring='accuracy', cv= 5, n_jobs = -1, verbose = 2, refit = True)
gridcv_BNB = GridSearchCV(estimator = BNB, param_grid = grid_BNB, scoring='accuracy', cv= 5, n_jobs = -1, verbose = 2, refit = True)
gridcv_XGBoost = GridSearchCV(estimator = XGBoost, param_grid = grid_XGBoost, scoring='accuracy', cv= 5, n_jobs = -1, verbose = 2, refit = True)
gridcv_AdaBoost = GridSearchCV(estimator = AdaBoost, param_grid = grid_AdaBoost, scoring='accuracy', cv= 5, n_jobs = -1, verbose = 2, refit = True)
gridcv_RF = GridSearchCV(estimator = RF, param_grid = grid_RF, scoring='accuracy', cv= 5, n_jobs = -1, verbose = 2, refit = True)
# gridcv_DA = GridSearchCV(estimator = DA, param_grid = grid_DA, scoring='accuracy', cv= 5, verbose = 2, refit = True)
# gridcv_DT = GridSearchCV(estimator = DT, param_grid = grid_DT, scoring='accuracy', cv= 5, verbose = 2, refit = True)
