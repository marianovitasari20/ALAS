import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import seaborn as sns
from utils.data_preprocessing import convert_units, log_transform, normalize_scores
from visualization.PredictionPlotter import PredictionPlotter

def create_plot_dataframe(lonlat, groundtruth, prediction):
    """
    Create a DataFrame from provided data.

    Parameters:
    - lon (pandas.Series): Series of longitude values.
    - lat (pandas.Series): Series of latitude values.
    - groundtruth (pandas.Series): Series of ground truth values.
    - prediction (pandas.Series): Series of prediction values.

    Returns:
    - pandas.DataFrame: DataFrame containing the provided data.
    """
    
    return pd.DataFrame({'lon': lonlat['lon'], 'lat': lonlat['lat'], 'groundtruth': groundtruth, 'prediction': prediction})

def prepare_data_for_plotting(y_test, y_pred, Xy_test_geo, conf):
    """
    Prepare data for plotting by converting units, applying log transformation, and normalizing.

    Parameters:
    - y_test: The actual test values.
    - y_pred: The predicted values from the model.
    - Xy_test_geo: Geo-spatial data associated with y_test and y_pred.
    - conf: Configuration dictionary containing various settings, including the unit conversion factor. 

    Returns:
    - df_plot: DataFrame containing geo-spatial data and converted y_test and y_pred values.
    - df_diff: DataFrame containing geo-spatial data and normalized differences between y_test and y_pred.
    """

    # Unit conversion
    y_test_new_units = convert_units(y_test, conf['plot']['new_units']).values
    y_pred_new_units = convert_units(y_pred, conf['plot']['new_units']).reshape(-1)  

    # Log transformation
    y_test_log10 = log_transform(y_test)
    y_pred_log10 = log_transform(y_pred)

    # Finding min and max for normalization
    z_min = y_test_log10.min()
    z_max = y_test_log10.max()

    # Normalization
    y_test_01 = normalize_scores(y_test_log10, z_min, z_max).values
    y_pred_01 = normalize_scores(y_pred_log10, z_min, z_max).reshape(-1)  

    # Creating DataFrames for plotting
    df_plot = create_plot_dataframe(Xy_test_geo, y_test_new_units, y_pred_new_units)
    df_diff = create_plot_dataframe(Xy_test_geo, y_test_01, y_pred_01)

    LOG_NORM = mcolors.LogNorm(vmin=5e-8, vmax=5e0)

    return df_plot, df_diff, LOG_NORM

def plot_predictions(y_test, y_pred, Xy_test_geo, conf):
    """
    Plots the groundtruth, prediction, and difference between them using the provided data.

    Parameters:
    - y_test: The actual test values.
    - y_pred: The predicted values from the model.
    - Xy_test_geo: Geo-spatial data associated with y_test and y_pred.
    - conf: Configuration dictionary containing various settings for plotting.
    """
    
    # Set the plotting style
    sns.set_style("darkgrid")

    # Prepare the data for plotting
    df_plot, df_diff, LOG_NORM = prepare_data_for_plotting(y_test, y_pred, Xy_test_geo, conf)

    # Initialize the plotter with the prepared data and configuration
    plotter = PredictionPlotter(df_plot, df_diff, LOG_NORM, conf)

    # Create the figure using the plotter
    fig = plotter.create_figure()

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()

            
