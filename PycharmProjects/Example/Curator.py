# Author: Cory Rudorf, me@coryrudorf.com

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler, CategoricalEncoder
from sklearn.pipeline import Pipeline, FeatureUnion


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


def build_num_pipeline(selector=None, imputer=None,std_scaler=None):
    steps = []
    if selector is not None:
        steps.append(('selector', selector))
    if imputer is not None:
        steps.append(('imputer', imputer))
    if std_scaler is not None:
        steps.append(('std_scaler', std_scaler))

    num_pipeline = Pipeline(steps)

    return num_pipeline


def predict_regression_model(model, dataset, labels):
    model_reg = model.fit(dataset, labels)
    model_predictions = model_reg.predict(dataset)
    return model_predictions
