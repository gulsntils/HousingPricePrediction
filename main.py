import pandas as pd
import sys

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer

# training set
TRAIN_FILE_PATH = "train.csv"
# test set
TEST_FILE_PATH = "test.csv"
# numeric columns
numeric_cols = []

# Utility function to get imputet data
def get_imputed_data(X_train, X_test):
	# get columns with missing values
	cols_with_missing = [col for col in X_train.columns 
								if X_train[col].isnull().any()]

	# add a label for every column that is missing
	for col in cols_with_missing:
		X_train[col + '_was_missing'] = X_train[col].isnull()
		X_test[col + '_was_missing'] = X_test[col].isnull()

	# impute data
	imputer = Imputer()
	imputed_X_train = imputer.fit_transform(X_train)
	imputed_X_test = imputer.transform(X_test)

	return imputed_X_train, imputed_X_test

# get training data from csv file
def get_training_data():
	training_data = pd.read_csv(TRAIN_FILE_PATH)
	# select target
	y = training_data.SalePrice

	global numeric_cols

	# select numeric columns
	numeric_cols = [col for col in training_data.columns
		if training_data.dtypes[col] != 'object']
	numeric_cols.remove('SalePrice')

	# select numeric data
	X = training_data[numeric_cols]

	return y, X

# get test data from csv file
def get_test_data():
	test_data = pd.read_csv(TEST_FILE_PATH)

	test_cols = [col for col in test_data.columns
		if test_data.dtypes[col] != 'object']

	X = test_data[numeric_cols]

	return X, test_data

# prediction model
def predict(X_train, y_train, X_test):
	# define model
	model = RandomForestRegressor()

	# fit the parameters
	model.fit(X_train, y_train)

	# predict
	y = model.predict(X_test)
	return y

# test model on training set
def test_on_train_set(y, X):
	# split data into training and validation data, for both predictors and target
	# The split is based on a random number generator.
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

	# make predictions
	predictions = predict(X_train, y_train, X_test)

	# calculate mean absolute error
	mae = mean_absolute_error(y_test, predictions)

	return mae

# transfer predictions into a csv file
def build_prediction_file(predictions, test_data):
	# create data drame object
	data_frame = pd.DataFrame({'Id': test_data.Id, 'SalePrice': predictions})
	# build csv file
	data_frame.to_csv('predictions.csv', index = False)


# extract training set
y_train, X_train = get_training_data()
# extract test set
X_test, test_data = get_test_data()

# get imputed data
X_train, X_test = get_imputed_data(X_train, X_test)

# test model on training set
error = test_on_train_set(y_train, X_train)
print 'Mean absolute error = ', error

# make predictions on actual test set
predictions = predict(X_train, y_train, X_test)
build_prediction_file(predictions, test_data)