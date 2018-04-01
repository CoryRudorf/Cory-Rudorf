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
    pipeline_dict = {}
    if selector is not None:
        steps.append(('selector', selector))
    if imputer is not None:
        steps.append(('imputer', imputer))
    if std_scaler is not None:
        steps.append(('std_scaler', std_scaler))

    return steps
    # num_pipeline = Pipeline([
    #     ('selector', DataFrameSelector(selector)),
    #     ('imputer', Imputer(strategy="median")),
    #     ('attribs_adder', CombinedAttributesAdder()),
    #     ('std_scaler', StandardScaler())])


test_list = (build_num_pipeline(selector='Test1', imputer='Test2', std_scaler='Test3'))

print(test_list[0][1])

