import numpy as np
import pandas as pd
from sklearn import metrics
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import sys
sys.path.append("..") 
from utils.data_preprocessing import unscale_predictions, normalize_scores

def compute_rmspe(actual, predicted):
    """
    Calculate the Root Mean Square Percentage Error (RMSPE).

    Parameters:
    - actual: Array-like, actual observed values.
    - predicted: Array-like, predicted values.

    Returns:
    - rmspe: The RMSPE value as a float.

    Notes:
    - Values in 'actual' equal to zero can cause division by zero errors. Such values are excluded from the calculation.
    """
    
    # Ensure 'actual' and 'predicted' are numpy arrays for element-wise operations
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Exclude zero values to avoid division by zero
    mask = actual != 0
    if not np.any(mask):
        raise ValueError("No non-zero actual values provided; RMSPE cannot be calculated.")
    
    rmspe_value = np.sqrt(np.mean(np.square((actual[mask] - predicted[mask]) / actual[mask])))
    return rmspe_value

def calculate_metrics(y_test, y_pred):
    """
    Evaluate the performance of model predictions using multiple metrics.

    Parameters:
    - y_test: Array-like, true labels.
    - y_pred: Array-like, predicted labels by the model.

    Returns:
    - r2: R2 score indicating the coefficient of determination.
    - ssim_score: Structural Similarity Index (SSIM) for comparing similarity between two images.
    - mape: Mean Absolute Percentage Error.
    - rmspe: Root Mean Square Percentage Error.
    - psnr: Peak Signal-to-Noise Ratio, a measure of the peak error between the original and reconstructed image.
    """
    
    # Validate input arrays are not empty
    if len(y_test) == 0 or len(y_pred) == 0:
        raise ValueError("Input arrays cannot be empty.")

    # Calculate R2 score
    r2 = metrics.r2_score(y_test, y_pred)

    # Calculate SSIM and PSNR scores
    ssim_score = ssim(y_test, y_pred, data_range=1)
    psnr = compare_psnr(y_test, y_pred, data_range=1)

    # Prepare a DataFrame for further calculations
    df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})

    # Replace zeros with NaN to avoid division errors in MAPE and RMSPE calculations
    df.replace(0, np.nan, inplace=True)
    df.dropna(inplace=True)

    # Calculate MAPE and RMSPE
    mape = metrics.mean_absolute_percentage_error(df['y_test'], df['y_pred']) if not df.empty else np.nan
    rmspe = compute_rmspe(df['y_test'], df['y_pred']) if not df.empty else np.nan
    
    return r2, ssim_score, mape, rmspe, psnr

def evaluate_model(y_pred, y_test_01, conf, z_min, z_max):
    """
    Evaluates the model predictions against the actual test labels using multiple metrics.

    Parameters:
    - y_pred: Array of model predictions before unscaled.
    - y_test_01: Array of actual test labels, normalized between 0 and 1.
    - conf: Configuration dictionary containing pickle path.
    - z_min: The minimum value used for normalization.
    - z_max: The maximum value used for normalization.

    Returns:
    - r2: The R^2 (coefficient of determination) regression score, indicating the proportion of variance in the dependent variable predictable from the independent variables.
    - ssim: The Structural Similarity Index (SSIM) for comparing the similarity of two images or sets of data.
    - mape: The Mean Absolute Percentage Error between the predictions and actual values, representing the average of the absolute percentage errors.
    - rmspe: The Root Mean Square Percentage Error between the predictions and actual values, providing a measure of the differences between predicted and observed values.
    """

    # Post-processing: unscale and normalize predictions
    y_pred = unscale_predictions(y_pred, conf["pickle_paths"]["pick_coef_y"])
    y_pred_01 = normalize_scores(y_pred.flatten(), z_min, z_max)
    
    # Evaluate predictions
    r2, ssim, mape, rmspe, psnr = calculate_metrics(y_test_01, y_pred_01)

    return r2, ssim, mape, rmspe, psnr

def update_metrics_history(metrics_history, r2, ssim, mape, rmspe, alpha, beta):
    """
    Updates the history of various metrics and parameters over the iterations of an active learning loop.
    This function is used to keep track of the performance of an active learning loop over multiple iterations. 
    By recording the history of different metrics and the alpha and beta weights, 
    it allows for analysis and visualization of the learning process over time, 
    helping to understand how the model's performance and the query strategy's parameters evolve.
    
    Parameters:
    - metrics_history: A dictionary where each key corresponds to a metric or parameter (e.g., 'r2', 'ssim', 'mape', 'rmspe', 'alpha', 'beta'), and each value is a list that stores the history of that metric/parameter over iterations.
    - r2: The R-squared value obtained in the current iteration.
    - ssim: The Structural Similarity Index (SSIM) obtained in the current iteration.
    - mape: The Mean Absolute Percentage Error obtained in the current iteration.
    - rmspe: The Root Mean Square Percentage Error obtained in the current iteration.
    - alpha: The alpha weight used in the fusion score calculation for the current iteration.
    - beta: The beta weight used in the fusion score calculation for the current iteration.

    Returns:
    None. The function updates the `metrics_history` dictionary in-place.
    """
    
    metrics_history['r2'].append(r2)
    metrics_history['ssim'].append(ssim)
    metrics_history['mape'].append(mape)
    metrics_history['rmspe'].append(rmspe)
    metrics_history['alpha'].append(alpha)
    metrics_history['beta'].append(beta)


