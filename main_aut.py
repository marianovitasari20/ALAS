import numpy as np
from evaluation.metrics import evaluate_model
from utils.data_preprocessing import prepare_y_test, unscale_predictions
from visualization.plot_aut import plot_predictions
from models.model_training import prepare_train_test_data, train_and_predict
from config_aut import configurations as conf

def print_statistical_properties(X_test, y_test, y_pred):
    """
    Creates a combined DataFrame from test features, true labels, and predictions, 
    and prints the descriptive statistics of the combined dataset.

    Parameters:
    - x_test: DataFrame containing the test features.
    - y_test: Array or Series containing the true labels for the test data.
    - y_pred: Array or Series containing the predicted labels for the test data.
    """
    
    # Combine x_test, y_test, and y_pred into a single DataFrame
    combined_df = X_test.copy()
    combined_df['y_test'] = y_test
    combined_df['y_pred'] = y_pred
    
    print(combined_df)

    # Print the descriptive statistics for the combined dataset
    print(combined_df.describe())

def main(Xy_train, X_test, y_test, Xy_test_geo, conf):
    # Prepare training and testing data
    X_train, y_train, x_test = prepare_train_test_data(Xy_train, X_test, conf)

    # Train the model and make predictions
    _, y_pred = train_and_predict(X_train, y_train, x_test)

    # Prepare y_test for evaluation
    y_test_01, z_min, z_max = prepare_y_test(y_test)

    # Evaluate the model performance
    r2, ssim, mape, rmspe, psnr = evaluate_model(y_pred, y_test_01, conf, z_min, z_max)
    print(r2, ssim, mape, rmspe, psnr)

    # Scale back the predictions to the original scale
    y_pred = unscale_predictions(y_pred, conf["pickle_paths"]["pick_coef_y"])

    # Apply inverse log transformation
    y_pred = np.power(10, y_pred)

    # Print the descriptive statistics for the combined dataset (X_test, y_test, y_pred)
    print_statistical_properties(X_test, y_test, y_pred)
    
    # Plot the predictions
    plot_predictions(y_test, y_pred, Xy_test_geo, conf)

if __name__ == "__main__":
    # Assume all datasets (Xy_train, X_test, y_test, Xy_test_geo) are ready
    
    # Call the main function with the prepared datasets and configuration
    main(Xy_train, X_test, y_test, Xy_test_geo, conf)

