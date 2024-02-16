import pandas as pd 
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

def prepare_data(df, target_col, scaler_pickle, target_scaler_pickle):
    """
    Prepares the dataset by cleaning, extracting features and target, transforming, and scaling.
    
    Parameters:
    - df: DataFrame containing the dataset.
    - target_col: String, the name of the target column.
    - scaler_pickle: String, file path to save the scaler for features.
    - target_scaler_pickle: String, file path to save the scaler for the target.

    Returns:
    - X: Scaled features as a DataFrame.
    - y: Scaled target as a DataFrame.
    """
    
    Xy = clean_data(df)
    X = extract_features(Xy, target_col)
    y = extract_target(Xy, target_col)
    
    X = log_transform(X)
    y = log_transform(y)
    
    X, y = scale_data(X, y, scaler_pickle, target_scaler_pickle)
    
    return X, y

def prepare_test_data(X_test, pickle_path):
    """
    Prepares the test dataset by applying feature transformation and scaling based on a pre-fitted scaler.

    Parameters:
    - X_test: DataFrame containing the test dataset features.
    - pickle_path: String, the file path to the pre-fitted scaler saved as a pickle file.

    Returns:
    - X_test: The transformed and scaled test dataset features as a DataFrame.
    """

    # Load the pre-fitted scaler from the specified pickle file
    scaler = load_scaler(pickle_path)

    # Apply any necessary feature transformations to the test dataset
    X_test = log_transform(X_test)

    # Scale the test dataset features using the loaded scaler
    X_test = scale_test_data(X_test, scaler)
        
    return X_test

def clean_data(df):
    """
    Cleans the data by replacing invalid values with NaN and dropping rows with NaN values.
    
    Parameters:
    - df: DataFrame to be cleaned.

    Returns:
    - cleaned_df: DataFrame after cleaning.
    """
    
    # Replace invalid values with NaN
    cleaned_df = df.map(lambda x: np.nan if x in [np.inf, -np.inf] or x < 0 else x)
    
    # Drop rows with NaN values
    cleaned_df = cleaned_df.dropna()
    
    return cleaned_df

def extract_features(df, target_col):
    """
    Extracts features from the DataFrame by dropping the target column.
    
    Parameters:
    - df: DataFrame containing features and target.
    - target_col: Name of the target column.

    Returns:
    - Features DataFrame.
    """
    
    return df.drop(columns=[target_col])

def extract_target(df, target_col):
    """
    Extracts the target column from the DataFrame.
    
    Parameters:
    - df: DataFrame containing features and target.
    - target_col: Name of the target column.

    Returns:
    - Target Series.
    """
    
    return df[target_col]

def log_transform(data):
    """
    Applies log transformation to the data.
    
    Parameters:
    - data: DataFrame.

    Returns:
    - Transformed features DataFrame.
    """
    
    data = np.log10(data) #.replace(0, np.nan)).fillna(0)  # Avoid log(0) and replace NaNs back with 0
    return data

def scale_data(X, y, scaler_pickle, target_scaler_pickle):
    """
    Scales the features and target using StandardScaler and saves the scalers.
    
    Parameters:
    - X: Features DataFrame to be scaled.
    - y: Target Series to be scaled.
    - scaler_pickle: File path to save the features scaler.
    - target_scaler_pickle: File path to save the target scaler.

    Returns:
    - Scaled features and target as DataFrames.
    """
    
    scaler = StandardScaler() 
    
    # Scale features and save the scaler
    scaler.fit(X)
    pickle.dump(scaler, open(scaler_pickle, 'wb'))
    X_scaled = pd.DataFrame(scaler.transform(X), 
                            columns=X.columns, 
                            index=X.index)
    
    # Scale target and save the scaler
    y_reshaped = y.values.reshape(-1, 1)
    scaler.fit(y_reshaped)
    pickle.dump(scaler, open(target_scaler_pickle, 'wb'))
    y_scaled = pd.DataFrame(scaler.transform(y_reshaped), 
                            index=X.index)
    
    return X_scaled, y_scaled

def scale_test_data(X, scaler):
    """
    Scales the provided dataset using the specified pre-fitted scaler.

    Parameters:
    - X: DataFrame containing features to be scaled.
    - scaler: Pre-fitted scaler object to be used for scaling.

    Returns:
    - X_scaled: The scaled features as a DataFrame, preserving original column names and indices.
    """
    
    # Apply the scaler transformation to the dataset and return the result as a DataFrame
    X_scaled = pd.DataFrame(scaler.transform(X), 
                            columns=X.columns, 
                            index=X.index)
    
    return X_scaled

def load_scaler(pickle_path):
    """
    Loads a scaler object from a specified pickle file.

    Parameters:
    - pickle_path: String, the file path to the pickle file where the scaler object is saved.

    Returns:
    - The scaler object loaded from the pickle file.
    """
    
    # Open the pickle file in read-binary mode and load the scaler object
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)

def unscale_predictions(y_pred, pickle_path):
    """
    Reverses the scaling transformation on model predictions to convert them back to their original scale.
    
    This function first loads a scaler object from a specified pickle file and then applies the scaler's
    inverse_transform method to the predictions. 

    Parameters:
    - y_pred: Array-like, the model's predictions which were scaled during the model training process.
    - pickle_path: String, the file path to the pickle file where the scaler object is saved.

    Returns:
    - y_pred: Array-like, the predictions reverted to their original scale.
    """
    # Load the pre-fitted scaler from the specified pickle file
    scaler = load_scaler(pickle_path)

    # Ensure y_pred is in the correct shape for the inverse transformation, and apply the inverse transformation
    y_pred = scaler.inverse_transform(y_pred.reshape(y_pred.shape[0],-1))

    return y_pred

def normalize_scores(scores, min_score=None, max_score=None):
    """
    Normalize the scores between 0 and 1 using min-max scaling.

    Parameters:
    - scores: A list or numpy array of scores to be normalized.
    - min_score: Minimum score for normalization. If None, the minimum score will be computed from scores.
    - max_score: Maximum score for normalization. If None, the maximum score will be computed from scores.

    Returns:
    - norm_scores: The normalized scores, scaled between 0 and 1.
    """
    
    if min_score is None:
        min_score = min(scores)
        
    if max_score is None:
        max_score = max(scores)
    
    # If all scores are the same, return the original scores
    if max_score == min_score:
        return scores
    
    # Normalize the scores between 0 and 1 using min-max scaling
    norm_scores = (scores - min_score) / (max_score - min_score)
    
    return norm_scores

def prepare_y_test(y_test):
    """
    Prepares the test labels (y_test) by applying a logarithmic transformation and normalizing the scores to a 0-1 range.

    Parameters:
    - y_test: Array-like, original test labels.

    Returns:
    - y_test_01: Array-like, normalized test labels after log transformation and scaling to a 0-1 range.
    - z_min: The minimum value in the transformed test labels, used for normalization.
    - z_max: The maximum value in the transformed test labels, used for normalization.
    """
    
    # Apply log transformation to y_test
    y_test_log10 = log_transform(y_test)  
    
    # Determine the minimum and maximum values of the log-transformed y_test for normalization
    z_min = y_test_log10.min()
    z_max = y_test_log10.max()

    # Normalize the log-transformed y_test to a 0-1 range
    y_test_01 = normalize_scores(y_test_log10, z_min, z_max)
    
    return y_test_01, z_min, z_max

def convert_units(data, factor):
    """
    Converts the units of the input data by multiplying it with a specified factor.

    Parameters:
    - data: Array-like, the input data to be converted.
    - factor: Float, the factor by which the input data is multiplied to achieve unit conversion.

    Returns:
    - Converted data as a result of the multiplication by the factor.
    """
    
    return data * factor
