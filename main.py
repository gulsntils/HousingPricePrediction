import pandas as pd
import sys

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# training set
TRAIN_FILE_PATH = "train.csv"
# test set
TEST_FILE_PATH = "test.csv"
# sample columns
COLUMNS_OF_INTEREST = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath',
			'BedroomAbvGr', 'TotRmsAbvGrd']

# get training data from csv file
def get_training_data():
	training_data = pd.read_csv(TRAIN_FILE_PATH)
	# select target
	y = training_data.SalePrice
	# select sample columns
	X = training_data[COLUMNS_OF_INTEREST]

	return y, X

# get test data from csv file
def get_test_data():
	test_data = pd.read_csv(TEST_FILE_PATH)

	X = test_data[COLUMNS_OF_INTEREST]

	return X, test_data

# prediction model
def predict(X_train, y_train, X_test, max_leaf_nodes):
	# define model
	model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)

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

	# initialize mae and max_depth
	best_mae = sys.maxsize
	max_depth = 5


	# find the depth of decision tree regressor to get
	# least mean absolute error
	for max_leaf_nodes in range(5,500):
		# make predictions
		predictions = predict(X_train, y_train, X_test, max_leaf_nodes)

		# calculate mean absolute error
		mae = mean_absolute_error(y_test, predictions)

		if (mae < best_mae):
			best_mae = mae
			max_depth = max_leaf_nodes

	return best_mae, max_depth

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

# test model on training set
error, max_leaf_nodes = test_on_train_set(y_train, X_train)
print 'Mean absolute error = ', error
print 'Best model has depth = ', max_leaf_nodes

# make predictions on actual test set
predictions = predict(X_train, y_train, X_test, max_leaf_nodes)
build_prediction_file(predictions, test_data)