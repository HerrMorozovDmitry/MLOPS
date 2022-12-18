import json
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")


class Model:
    def __init__(self, type, params, dataset_path, target, drop_cols):
        self.type = type
        self.trained = False
        self.prediction = None
        self.accuracy = None

        model_selection = {
            "LogReg": LogisticRegression,
            "SVM": SVC,
            "DecisionTree": DecisionTreeClassifier,
        }
        model = model_selection[self.type]

        try:
            self.__data = pd.read_csv(dataset_path).drop(columns=drop_cols).dropna()
            self.__X_train = self.__data.drop(columns=target)
            self.__y_train = self.__data[target]

            column_transformer = ColumnTransformer(
                [
                    (
                        "enc",
                        OneHotEncoder(handle_unknown="ignore"),
                        self.__X_train.select_dtypes("object").columns,
                    ),
                    (
                        "scaling",
                        StandardScaler(),
                        self.__X_train.select_dtypes("number").columns,
                    ),
                ]
            )
            self.__pipeline = Pipeline(
                steps=[
                    ("enc_and_scaling", column_transformer),
                    ("classifier", model(**params)),
                ]
            )
        except Exception:
            raise

    def fit(self):
        try:
            self.__pipeline.fit(self.__X_train, self.__y_train)
            self.trained = True
            self.prediction = self.__pipeline.predict(self.__X_train)
            self.accuracy = round(accuracy_score(self.prediction, self.__y_train), 5)
            return "Model was Trained and Tested for Accuracy"
        except Exception:
            raise

    def predict(self):
        try:
            return json.dumps(self.prediction.tolist())
        except Exception:
            raise