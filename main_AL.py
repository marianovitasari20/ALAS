import numpy as np
from active_learning.ALProcess import ALProcess
from utils.file_utils import save_AL_results
from config_AL import configurations as conf
from visualization.plot_AL import plot_metrics, plot_baseline_vs_the_best

def main(Xy_train, Xy_test, conf):
    """
    Main function to run active learning experiments and collect results.

    Parameters:
    - Xy_train: DataFrame containing the training data.
    - Xy_test: DataFrame containing the test data.
    - conf: Configuration dictionary with settings and parameters for the experiments.

    Returns:
    - results_dict: A dictionary with aggregated results from all active learning experiments.
    """

    # Create dictionaries to store results for each AL_query
    results_dict = {query: {"_r2_list": [], "_ssim_list": [], "_mape_list": [], "_rmspe_list": [], "_selected_indices_list": [], "_alpha_list": [], "_beta_list": []} for query in conf['main']['AL_queries']}

    # Repeat the experiment for the specified number of times in the configuration
    for _ in range(conf["AL"]["num_exp"]):

        # Randomly sample a subset of the training and test data for each experiment
        _xy_train = Xy_train.sample(n=conf['AL']['n_samples'], replace=False)
        _Xy_test_small = Xy_test.sample(n=conf['AL']['n_val_samples'], replace=False)

        # Prepare the test features and labels
        X_test = _Xy_test_small.drop(columns=[conf["target"]["y_target"]])
        y_test = _Xy_test_small[conf["target"]["y_target"]]
        
        # Randomly select initial points for active learning
        selected_indices0 = np.random.choice(_xy_train.index, size=conf['AL']['initial_num_points'], replace=False)
        selected_indices_list = selected_indices0
        
        # Iterate over each active learning query strategy
        for AL_query in conf['main']['AL_queries']:      
            # Initialize the active learning process with the current configuration  
            AL = ALProcess(_xy_train, X_test, y_test, selected_indices_list, AL_query, conf)
            
            # Run the active learning process and collect results
            results = AL.run()
            
            # Aggregate results for each metric across all experiments
            for key, value in results.items():
                results_dict[AL_query][key].append(value)

    return results_dict

if __name__ == "__main__":
    # Assume the dataset (Xy_train and Xy_test) are ready
    
    # Run the main function and collect results
    results_dict = main(Xy_train, Xy_test, conf) 
    
    # Plot the results
    plot_metrics(results_dict, conf) 
    plot_baseline_vs_the_best(results_dict, conf)
    
    # Save the results to a specified location
    save_AL_results(results_dict, conf['main']['folder_path'], conf['main']['file_name'])

