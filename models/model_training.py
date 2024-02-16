from utils.data_preprocessing import prepare_data, prepare_test_data
from models.gaussian_process import get_gp_model

def prepare_train_test_data(xy_train, X_test, conf):
    """
    Prepares the training and test datasets for model training and evaluation.

    Parameters:
    - xy_train: DataFrame containing the training data with features and target.
    - X_test: DataFrame containing the test data features.
    - conf: Dictionary containing configuration parameters including target column name and paths to pickle files for scalers.

    Returns:
    - x_train: DataFrame of scaled training features.
    - y_train: DataFrame of scaled training target.
    - x_test: DataFrame of scaled test features.
    """
    
    # Prepare training data
    x_train, y_train = prepare_data(xy_train, conf['target']['y_target'], conf['pickle_paths']['pick_coef'], conf['pickle_paths']['pick_coef_y'])
    
    # Prepare test data
    x_test = prepare_test_data(X_test, conf['pickle_paths']['pick_coef'])

    return x_train, y_train, x_test

def train_and_predict(x_train, y_train, x_test):
    """
    Trains a Gaussian Process Regressor model using the provided training data and makes predictions on the test data.

    Parameters:
    - x_train: DataFrame or numpy array containing the training features.
    - y_train: DataFrame or numpy array containing the training target values.
    - x_test: DataFrame or numpy array containing the test features for which predictions need to be made.

    Returns:
    - model: The trained Gaussian Process Regressor model.
    - y_pred: Predictions made by the model on the test data.
    """

    # Instantiate and train the Gaussian Process Regressor model using the training data
    model = get_gp_model(x_train, y_train)

    # Use the trained model to make predictions on the test data
    y_pred = model.predict(x_test)

    return model, y_pred
