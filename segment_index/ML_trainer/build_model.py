import os
import pandas as pd

from sklearn.linear_model import LassoCV, Lasso
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def build_model(random_state:int):

    # model parameter setting
    model_params = {

        "rf": {"model": RandomForestClassifier(random_state = random_state),
               "params": {"n_estimators": [10, 100],
                          "max_depth": [10, 20],
                          "max_features": ['auto', 'sqrt'],
                          "min_samples_leaf": [2, 4]}},

        "dt": {"model": DecisionTreeClassifier(random_state = random_state),
               "params":{"criterion": ["entropy", "gini"],
                         "max_depth": [10, 20],
                         "min_samples_leaf": [2, 4]}},

        "svc": {"model": SVC(random_state = random_state, probability=True),
                "params":{"kernel": ["linear", "rbf"],
                          "C": [1, 10],
                          "gamma": [0.5, 0.01]}},

        "mlp": {"model": MLPClassifier(random_state = random_state),
                "params":{"hidden_layer_sizes": [(4,), (8,), (10,)],
                          "max_iter": [100, 200],
                          "activation": ["identity", "tanh", "relu"],
                          "solver": ["sgd", "adam"]}},

        "gbm": {"model": GradientBoostingClassifier(random_state = random_state),
                "params":{"n_estimators": [50, 100, 150],
                          "learning_rate": [0.1, 0.05],
                          "max_depth": [2, 4],
                          "min_samples_leaf": [2, 4],
                          "min_samples_split": [2, 4]}},

        "xgb": {"model": XGBClassifier(random_state = random_state),
                "params":{"max_depth":[5, 6, 8],
                          "min_child_weight":[1, 3, 5],
                          "gamma":[0, 1, 2, 3],
                          "n_estimators":[50, 100]}},

        "lgbm": {"model": LGBMClassifier(random_state = random_state),
                 "params":{"boosting_type":['gbdt'],
                           "max_depth":[2,3,4,5],
                           "num_leaves":[3,4,5,6,7],
                           "learning_rate":[0.01, 0.05, 1.0],
                           "objective":['binary'],
                           "metric":['auc'],
                           "colsample_bytree":[0.5, 0.7, 0.9, 1.0],
                           "subsample":[0.5, 0.7, 0.9, 1.0]}}
               }


    return model_params

