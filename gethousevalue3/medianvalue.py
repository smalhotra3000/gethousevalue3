import os
import tarfile
import pickle

import numpy as np
import pandas as pd
from six.moves import urllib
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Defining the variables
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("inputs", "raw_data")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"



# Function to get inputs from the user in the form of arguments
def get_user_inputs():
    parser = argparse.ArgumentParser()

    # Argument to get location of data folder
    parser.add_argument(
        "--data_folder",
        help="give path to folder where training & validation data will be stored",
        default="inputs/processed",
    )

    # Argument to get log level
    parser.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Select Log level",
        default="DEBUG",
    )

    # Argument to get path for log file
    parser.add_argument("--log_path", help="Specify the path to the log file", type=str)

    # Argument to know if user wants logs to be displayed in console output
    parser.add_argument(
        "--no_console_log",
        choices=["y", "n"],
        default="n",
        help="Disable logging to the console (y/n)",
    )

    user_inputs = parser.parse_args()

    Data_folder = user_inputs.data_folder
    Log_level = user_inputs.log_level
    Log_file_path = user_inputs.log_path
    Log_to_file = bool(Log_file_path)
    No_console_log = user_inputs.no_console_log

    return Data_folder, Log_level, Log_file_path, Log_to_file, No_console_log


# Function to download housing data
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


# Funtion to read housing data
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def ingest_data():
    try:
        # Getting the data
        fetch_housing_data()
        housing = load_housing_data()

        # Dividing the data into train & test datasets
        train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

        # Dividing the data into stratified train & test datasets
        # based on buckets created on the median_income column
        housing["income_cat"] = pd.cut(
            housing["median_income"],
            bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
            labels=[1, 2, 3, 4, 5],
        )

        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in split.split(housing, housing["income_cat"]):
            strat_train_set = housing.loc[train_index]
            strat_test_set = housing.loc[test_index]

        # Removing the income_cat column
        for set_ in (strat_train_set, strat_test_set):
            set_.drop("income_cat", axis=1, inplace=True)
            
        return strat_train_set, strat_test_set

    except Exception as e:
        print("Data ingest failed with error: ", e)
        return None, None



def train(strat_train_set, model_type, Random_State=42, N_Estimators=30, Max_Features=5):
    try:
        # Removing training labels from data
        housing = strat_train_set.drop("median_house_value", axis=1)

        # Removing categorical columns
        housing_num = housing.drop("ocean_proximity", axis=1)

        # Storing the training labels in a seperate array
        housing_labels = strat_train_set["median_house_value"].copy()

        # Imputing the training data using median strategy
        imputer = SimpleImputer(strategy="median")

        imputer.fit(housing_num)
        X = imputer.transform(housing_num)
        
        # Creating the calculated fields
        housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)
        housing_tr["rooms_per_household"] = housing_tr["total_rooms"] / housing_tr["households"]
        housing_tr["bedrooms_per_room"] = housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
        housing_tr["population_per_household"] = housing_tr["population"] / housing_tr["households"]

        housing_cat = housing[["ocean_proximity"]]
        housing_prepared = housing_tr.join(pd.get_dummies(housing_cat, drop_first=True))

        # Creating the model
        if model_type=='Linear':
                Model_ = LinearRegression()

        elif model_type=='DecisionTree':
                Model_ = DecisionTreeRegressor(random_state=Random_State)

        elif model_type=='RandomForest':
            Model_ = RandomForestRegressor(n_estimators= N_Estimators, max_features= Max_Features, random_state=Random_State)
      
        Model_.fit(housing_prepared, housing_labels)
        
        return Model_

    except Exception as e:
        print("Model creation failed with error: ", e)
        return None

    
    
def score(strat_train_set, strat_test_set, Model_):
    try:
        # Removing the label column from the training data
        housing = strat_train_set.drop("median_house_value", axis=1)

        # Removing categorical variables
        housing_num = housing.drop("ocean_proximity", axis=1)

        # Imputing the training data using median strategy
        imputer = SimpleImputer(strategy="median")

        imputer.fit(housing_num)
        X = imputer.transform(housing_num)

        # Creating the calculated fields
        housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)
        housing_tr["rooms_per_household"] = housing_tr["total_rooms"] / housing_tr["households"]
        housing_tr["bedrooms_per_room"] = housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
        housing_tr["population_per_household"] = housing_tr["population"] / housing_tr["households"]

        housing_cat = housing[["ocean_proximity"]]
        housing_prepared = housing_tr.join(pd.get_dummies(housing_cat, drop_first=True))

        # Storing the training lables in a seperate array
        housing_labels = strat_train_set["median_house_value"].copy()

        # Removing lable column from test data
        X_test = strat_test_set.drop("median_house_value", axis=1)

        # Removing categorical column from test data
        X_test_num = X_test.drop("ocean_proximity", axis=1)

        # Imputing the test data
        X_test_prepared = imputer.transform(X_test_num)

        X_test_tr = pd.DataFrame(X_test_prepared, columns=X_test_num.columns, index=X_test.index)
        X_test_tr["rooms_per_household"] = X_test_tr["total_rooms"] / X_test_tr["households"]
        X_test_tr["bedrooms_per_room"] = X_test_tr["total_bedrooms"] / X_test_tr["total_rooms"]
        X_test_tr["population_per_household"] = X_test_tr["population"] / X_test_tr["households"]

        X_test_cat = X_test[["ocean_proximity"]]

        X_test_prepared = X_test_tr.join(pd.get_dummies(X_test_cat, drop_first=True))

        # Storing the test labels in a seperate array
        y_test = strat_test_set["median_house_value"].copy()

        # Making prediction using model
        housing_predictions = Model_.predict(housing_prepared)

        # Calculating the accuracy for linear regression model
        mae = mean_absolute_error(housing_labels, housing_predictions)
        mse = mean_squared_error(housing_labels, housing_predictions)
        rmse = np.sqrt(mse)
        return rmse

    except Exception as e:
        print("Model scoring failed with error: ", e)
        return None
    

if __name__ == "__main__":
    Train_data, Test_data = ingest_data()
    prediction_model = train(Train_data, 'Linear')
    rmse_score = score(Train_data, Test_data, prediction_model)
    print (rmse_score)
    