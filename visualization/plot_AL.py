import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick
import seaborn as sns
import shap

sns.set_style('darkgrid')  

def plot_metric(num_queries, metric_data, label, color=None, linestyle='-', marker='o', markersize=1):
    """
    Plots a metric over a specified range of queries, allowing for customization of the plot appearance.

    Parameters:
    - num_queries: Array-like, specifying the x-axis values for the plot, representing the number of queries.
    - metric_data: Array-like, containing the metric values to be plotted.
    - label: String, the label for the plot, used in the legend.
    - color: String or None, optional, the color of the plot line. If None, matplotlib's default color cycle is used.
    - linestyle: String, optional, the style of the plot line (e.g., '-', '--', '-.', ':').
    - marker: String, optional, the marker style for the plot points (e.g., 'o', '*', 's').
    - markersize: Int, optional, the size of the markers.

    """

    plt.plot(num_queries, metric_data, marker=marker, linestyle=linestyle, label=label, markersize=markersize, color=color)

def plot_metrics(results_dict, conf, dpi=200):
    """
    Plots the evolution of evaluation metrics over the number of active learning queries.
    Each metric's values are averaged over multiple runs for each active learning query and plotted against the number of queries.

    Parameters:
    - results_dict: Dictionary containing the metrics collected at each active learning query.
    - conf: Configuration dictionary with settings for active learning and plotting.
    - dpi: Integer, the resolution of the plot in dots per inch.
    """
    
    num_queries = list(range(1, conf["AL"]["num_AL_steps"] + 1))
    metrics_to_plot = ["_r2_list", "_ssim_list", "_mape_list", "_rmspe_list"] 
    metric_labels = ["R2", "SSIM", "MAPE", "RMSPE"]

    # Loop over each metric to create separate plots
    for metric, metric_label in zip(metrics_to_plot, metric_labels):
        plt.figure(figsize=(4, 3.5), dpi=dpi)
        
        # Loop over different active learning query strategies 
        for AL_query, AL_queries_label in zip(conf['main']['AL_queries'], conf['main']['AL_queries_labels']):
            metric_data = results_dict[AL_query][metric]
            avg_metric = np.mean(np.array(metric_data), axis=0)
            
            # Plot the average metric values
            plot_metric(num_queries, avg_metric, f'{AL_queries_label}')

        plt.xlabel('Number of Queries')
        plt.ylabel(f'{metric_label} Value')
        plt.title(f'{metric_label} Values vs. Number of Queries')
        plt.legend()
        plt.grid(True)
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=1))
        plt.show()

def plot_shap_summary(shap_values, x_test, features):
    """
    Generates a SHAP summary bar plot for the given SHAP values and features.

    Parameters:
    - shap_values: The SHAP values for the features.
    - x_test: The test dataset used for SHAP value computation.
    - features: The list of feature names corresponding to the SHAP values.
    """
    
    shap.summary_plot(shap_values, x_test, feature_names=features, plot_type='bar')
    
def find_similar_index(avg_metric, target_value, metric, start_index):
    """
    Finds the index in a metric series where the metric value is similar to a specified target value.

    This function searches for the point in 'avg_metric' starting from 'start_index' where
    the metric value is either just equal to or less than the 'target_value' for error metrics
    like MAPE and RMSPE, or equal to or greater than for other types of metrics.

    Parameters:
    - avg_metric: Array-like, a series of averaged metric values over multiple iterations or runs.
    - target_value: Float, the target metric value to compare against the series.
    - metric: String, the name of the metric being evaluated. Used to determine the comparison direction.
    - start_index: Int, the index in 'avg_metric' from where to start the search.

    Returns:
    - Int or None: The index in 'avg_metric' where the value is similar to 'target_value' based on the criteria,
      or None if no such index is found.
    """

    # Loop through the metric series starting from 'start_index'
    for idx in range(start_index, len(avg_metric)):
        # For error metrics like MAPE and RMSPE, look for values equal to or less than the target value
        if metric in ["_mape_list", "_rmspe_list"] and avg_metric[idx] <= target_value:
            return idx  
        
        # For other types of metrics, look for values equal to or greater than the target value
        elif metric not in ["_mape_list", "_rmspe_list"] and avg_metric[idx] >= target_value:
            return idx 

    # Return None if no similar index is found
    return None

def plot_baseline_vs_the_best(results_dict, conf, dpi=200, brown_color=(132/255, 75/255, 63/255)):
    """
    Plots the comparison of different active learning strategies against a baseline value at a specified index.

    Parameters:
    - results_dict: Dictionary containing the results for each active learning strategy.
    - conf: Configuration dictionary with settings for active learning and plotting.
    - dpi: Resolution of the plots.
    - brown_color: Color to be used for the best AL query plot line.
    """
    
    num_queries = list(range(1, conf["AL"]["num_AL_steps"] + 1))
    metrics_to_plot = ["_r2_list", "_ssim_list", "_mape_list", "_rmspe_list"]
    metric_labels = ["R2", "SSIM", "MAPE", "RMSPE"]
    AL_queries = ["wifi_EUC_epDecay", "random"]
    AL_queries_labels = ["WiFi euc", "random"]
    compare_index = 32

    for metric, metric_label in zip(metrics_to_plot, metric_labels):
        plt.figure(figsize=(4, 3), dpi=dpi)

        for i, AL_query in enumerate(AL_queries):
            avg_metric = np.mean(np.array(results_dict[AL_query][metric]), axis=0)
            value_at_33 = avg_metric[compare_index]
            print(f"avg_metric ({AL_queries_labels[i]}) at index {compare_index + 1}: {value_at_33}")

            if(i == 0):  # Best AL query, e.g., WiFi euc
                wifi_euc_value_at_33 = value_at_33
                plot_metric(num_queries, avg_metric, AL_queries_labels[i], color=brown_color)
            else:  # Baseline AL query, e.g., random
                plot_metric(num_queries, avg_metric, AL_queries_labels[i])

        plt.axhline(y=wifi_euc_value_at_33, color='black', linestyle='--', label=r'$B_{\mathrm{max}}$'+' (Query 33)')
        similar_index = find_similar_index(avg_metric, wifi_euc_value_at_33, metric, compare_index)
        plt.xlim(0, similar_index or len(avg_metric))
        print(f"Index of 'random' that intersects with 'WiFi euc' ({metric_label}): {similar_index}")

        plt.xlabel('Number of Queries')
        plt.ylabel(f'{metric_label} Value')
        plt.title(f'{metric_label} Values vs. Number of Queries')
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=1))
        plt.legend()
        plt.show()