import os
import pickle
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from joblib import dump, load

# Features to drop
INLAND = 19
ISLAND = 21
NEAR_BAY = 22

class FeatureEngineering(BaseEstimator, TransformerMixin):
    """
    FeatureEngineering is a class that performs feature engineering on the input data.
    Methods:
    - fit(X, y=None): Fits the transformer on the input data.
    - transform(X): Transforms the input data by adding new engineered features.  
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X['bedrooms_per_room'] = X['total_bedrooms'] / X['total_rooms']
        X['is_capped'] = X['median_house_value'] == 500000
        X['population_per_household'] = X['population'] / X['households']
        X['rooms_per_household'] = X['total_rooms'] / X['households']
        X['income_per_household'] = X['median_income'] / X['households']
        X['income_per_bedroom'] = X['median_income'] / X['total_bedrooms']
        X['income_per_room'] = X['median_income'] / X['total_rooms']
        X['age_of_population'] = X['housing_median_age'] / X['population']
        X['age_of_households'] = X['housing_median_age'] / X['households']
        X['population_density'] = X['population'] / X['total_rooms']
        X['near_ocean'] = X['ocean_proximity'] == 'NEAR OCEAN'
        X['less_1h_ocean'] = X['ocean_proximity'] == '<1H OCEAN'
        return X

class DropColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if type(X) == pd.DataFrame:
            if self.column in X.columns:
                X = X.drop(columns=[self.column])
        elif type(X) == np.ndarray:
            X = np.delete(X, self.column, axis=1)
        return X
    
def get_pipeline():
    """
    Returns a pipeline that preprocesses the data.
    """
    pipeline = Pipeline([
        ('feature_eng', FeatureEngineering()),
        ('drop_median_house_value', DropColumnTransformer('median_house_value')),
        ('drop_ocean_proximity', DropColumnTransformer('ocean_proximity')),
        ('imputer', SimpleImputer(strategy='median')),
        ('std_scaler', StandardScaler()),
    ])
    return pipeline

def load_pipeline(config):
    """
    Loads a serialized pipeline from the given file path.

    Parameters:
    config (dict): Configuration dictionary containing the paths and parameters.

    Returns:
    Pipeline: A pipeline object that preprocesses the data.
    """
    pipeline_path = config['data']['pipeline']['model_save_path']
    with open(pipeline_path, 'rb') as f:
        pipeline = pickle.load(f)
    return pipeline

def get_datasets(config):
    """
    Reads the housing data from the given file path and performs data preprocessing using a pipeline.
    Checks if a serialized pipeline exists. If not, creates and fits a new one, then saves it.

    Parameters:
    config (dict): Configuration dictionary containing the paths and parameters.

    Returns:
    tuple: A tuple containing the preprocessed training and testing datasets along with their corresponding labels.
           The tuple is in the following order: (X_train, X_test, y_train, y_test)
    """

    data_path = config['data']['data_path']
    pipeline_path = config['data']['pipeline']['model_save_path']
    test_size = config['data']['test_size']
    random_state = config['data']['random_state']

    housing_data = pd.read_csv(data_path)
    train_set, test_set = train_test_split(housing_data, test_size=test_size, random_state=random_state)

    y_train = train_set['median_house_value'].to_numpy()
    y_test = test_set['median_house_value'].to_numpy()

    if os.path.exists(pipeline_path):
        with open(pipeline_path, 'rb') as f:
            pipeline = pickle.load(f)
        X_train = pipeline.transform(train_set)
    else:
        pipeline = get_pipeline()
        X_train = pipeline.fit_transform(train_set)
        with open(pipeline_path, 'wb') as f:
            pickle.dump(pipeline, f)
    X_test = pipeline.transform(test_set)

    return X_train, X_test, y_train, y_test
