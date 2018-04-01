# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import tarfile
from six.moves import urllib
import pandas as pd
from pandas import plotting
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import Imputer

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


# fetch_housing_data()

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    print(csv_path)
    return pd.read_csv(csv_path)


housing = load_housing_data()


#
# print(housing.describe())
#
# housing.hist(bins=50, figsize=(20,15))
# plt.show()

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


#
# test_set, train_set = split_train_test(housing, 0.2)
#
# print(len(train_set), "train +", len(test_set), "test")

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

housing = strat_train_set.copy()

# Basic longitude/latitude scatter plot, alpha shows density of points.
# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

###############################################################################
# ---color scaled housing median prices using jet
###############################################################################
# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
#             s=housing["population"]/100, label="population", figsize=(10,7),
#             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True
#             )
# plt.legend()
# plt.show()

###############################################################################
# .corr() method in pandas shows all columns correlation with one another.
###############################################################################
# corr_matrix = housing.corr()
# print(corr_matrix["median_house_value"].sort_values(ascending=False))

###############################################################################
# ---from pandas library, plotting.scatter_matrix() plots every numerical atrribute
# against every other numerical attribute. Below we specify which numerical
# attributes we want plotted.
###############################################################################
# attributes = ["median_house_value", "median_income", "total_rooms",
#              "housing_median_age"]
# plotting.scatter_matrix(housing[attributes], figsize=(12,8))

###############################################################################
# Focus on most promising attribute, which in this cause is median income.
###############################################################################
# housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
# plt.show()

###############################################################################
# ---Houses by SF Coordinates
# Detour: Narrowing down houses by longitude/latitude coordinates. The below is
# for all houses in dataset that are considered within SF city limits by coords.
###############################################################################
# SF_N = 37.929824
# SF_S = 37.63983
# SF_W = -123.173825
# SF_E = -122.28178

# sf_housing = housing[(housing['latitude'] >= SF_S) & (housing['latitude'] <= SF_N)
# & (housing['longitude'] >= SF_W) & (housing['longitude'] <= SF_N)]
#
# print(sf_housing.describe())
#
##color scaled housing median prices using jet
# sf_housing.plot(kind="scatter", x="total_bedrooms", y="median_house_value", alpha=0.1)
# plt.show()

###############################################################################
# ---Creating new features based on attribute combinations.
###############################################################################

housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]

corr_matrix = housing.corr()
# print(corr_matrix["median_house_value"].sort_values(ascending=False))

###############################################################################
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

###############################################################################
# ---Imputer, preprocessing sklearn class
# to replace missing numerical values
###############################################################################
imputer = Imputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)

X = imputer.transform(housing_num)
# print(imputer.statistics_)

housing_tr = pd.DataFrame(X, columns=housing_num.columns)

###############################################################################

housing_cat = housing["ocean_proximity"]
# print(housing_cat.head(10))

###############################################################################
# ---Convert categories from text to number using pandas factorize() method, which maps
# each category to a different integer. factorize() returns two objects: 1) Integer for each
# unique category, 


# print(housing_cat_encoded[:10])

# ---and 2) returns the list of categories that were factorized.
# print(housing_categories)

###############################################################################
# --- One Hot Encoding
# A major issue with converting categorical features to numeric factors is that
# ML algorithms will assume two nearby values are more similar than 2 distant values.
# A common solution to this problem is to create one binary attribute per category,
# where 1 is hot and the category is true for the observation, and 0 is cold and false
# for the observation.
from sklearn.preprocessing import OneHotEncoder, CategoricalEncoder

# Instantiate OneHotEncoder object from sklearn preprocessing class
# encoder = OneHotEncoder()
# # transform factorized categorical data from housing set  and reshape it since
# # fit.transform expects a 2D array.
# housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1, 1))
# Output is a numpy sparse matrix. This is so python doesnt use extra memory storing
# 0 elements. A sparse matrix only saves nonzero vales.
# print(type(housing_cat_1hot))
# ---convert 1hot to dense numpy array
# print(housing_cat_1hot.toarray())
###############################################################################
# ---Custom Transformers
# sklearn provides many transformer classes out of the box, but it will be
# necessary to build out custom tranformers for specific clean up operations or
# combining specific attributes.
# Easy way to do this is to create a custom transformer class for each dataset and
# implement 3 methods: fit(), transform(), and fit_transform(), which is covered by
# integrating the TransformerMixin as a base class.
# Also suggested is to use BaseEstimator to get two extra methods (get_params() and
# set_params()) that will be useful for automatic hyperparameter tuning.
# ---Example Custom Transformer Class
from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)
# print(housing_extra_attribs)
###############################################################################
# ---Feature Scaling
# Machine Learning algorithms dont perform well when the input numerical attributes
# have very different scales. Two ways to get all attributes to have the same scale:
# min-max scaling and standardization.
# ---Min-Max Scaling
# Also referred to as normalization, where values are shifted and rescaled so that
# they end up ranging from 0 to 1, and we subtract the min value and dividing by the max
# minus the min.
# The sklearn transformer MinMaxScaler achieves this functionality.
# ---Standardization
# To standardize numerical values, it subtracts the mean value, then divides by
# the variance so that the resulting distribution has unit variance. Standardization
# does not bound values to a fixed range, which is a problem for some algorithms.
# Standardization is however less affected by outliers than min-max.
# The sklearn transformer StandardScaler acheives this functionality.
# ---Scaling Warning
# It is important to fit scalers to the training data only, not the full dataset
# (including the test set). Only then can you use them to transform the training set/test set.
###############################################################################
# ---Transformation Pipeline
# sklearn provides a Pipeline class that can help organize and order transformation steps
# for a given dataset to prepare the data for training.
# The Pipeline constructor takes a list of name/estimator pairs defining a sequence
# of steps. All but the last estimator must be transformers (they must have a fit_transform()
# method.)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn_pandas import DataFrameMapper
num_pipeline = Pipeline([
    ('imputer', Imputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler())])

housing_num_tr = num_pipeline.fit_transform(housing_num)

mapper1 = DataFrameMapper([
    ("ocean_proximity", [LabelBinarizer()])], sparse=False)

# cat_pipeline = Pipeline([
#     ('labeler', StringIndexer()),
#     ('encoder', OneHotEncoder(handle_unknown='ignore'))])
#
# housing_cat_tr = cat_pipeline.fit_transform(housing)
# print(housing_cat_tr)
#housing_cat = housing["ocean_proximity"]
# housing_cat_encoded, housing_categories = housing_cat.factorize()
#
# encoder = OneHotEncoder()
# # transform factorized categorical data from housing set  and reshape it since
# # fit.transform expects a 2D array.
# housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1, 1))
# print(housing_cat_1hot)
# cat_1hot_df = pd.DataFrame(housing_cat_encoded, columns=housing_categories)
# print(cat_1hot_df)


# print(housing_cat_tr)

# print(housing_num_tr)
###############################################################################
# ---Feeding a Pandas DF containing non-numerical columns directly into a Pipeline
# Example DataFrameSelector class transforms data by selecting desired attributes,
# dropping the rest, and converting the resulting DataFrame into a NumPy array.
# You could use this class to select numerical only values, then categorical values only,
# then running them through 2 pipelines for each feature types.
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


# ---Pipelines with DataFrameSelector
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

new_df = housing[cat_attribs]
#print(new_df)
num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', Imputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler())])
cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
     ('cat_encoder', CategoricalEncoder(encoding="onehot-dense"))])
# ---Combining Pipelines with sklearn class FeatureUnion
# Give a list of transformers or pipelines.
from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline)
])
# Run entire pipeline:
housing_prepared = full_pipeline.fit_transform(housing)
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:", lin_reg.predict(some_data_prepared))
print("Predictions:", list(some_labels))

from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)