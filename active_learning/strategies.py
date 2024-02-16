from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances 
from sklearn.metrics.pairwise import cosine_distances
import sys
sys.path.append("..") 
from utils.data_preprocessing import normalize_scores
from evaluation.metrics import evaluate_model
from models.model_training import prepare_train_test_data, train_and_predict

def representative_scores(x_unlabeled, num_clusters, random_state=42):
    """
    Calculate representativeness scores of unlabeled data points based on their distance to cluster centroids.

    Parameters:
    - x_unlabeled: DataFrame, the features of unlabeled data points.
    - num_clusters: int, the number of clusters to form.
    - random_state: int, random state for reproducibility.

    Returns:
    - rep_score: DataFrame, representativeness scores for each data point in x_unlabeled.
    """
    
    if x_unlabeled.empty:
        raise ValueError("x_unlabeled is empty")

    if num_clusters <= 0:
        raise ValueError("num_clusters must be positive")
    
    # Cluster the unlabeled data into 'num_clusters' clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state)
    cluster_assignments = kmeans.fit_predict(x_unlabeled)

    # Calculate cluster centroids
    cluster_centers = kmeans.cluster_centers_

    # Calculate the minimum distance of each data point to cluster centroids
    min_distances = []
    for cluster_label in range(num_clusters):
        cluster_indices = np.where(cluster_assignments == cluster_label)[0]
        if len(cluster_indices) > 0:
            cluster_data = x_unlabeled.iloc[cluster_indices]
            distances_to_centroid = np.linalg.norm(cluster_data - cluster_centers[cluster_label], axis=1)
            min_distances.extend(distances_to_centroid)

    # Calculate representativeness scores as the inverse of minimum distances, as we will be selecting based on the highest value
    representativeness_scores = 1 / np.array(min_distances)
    representativeness_scores = normalize_scores(representativeness_scores)

    # Create a DataFrame with the same index as x_unlabeled
    rep_score = pd.DataFrame(representativeness_scores, index=x_unlabeled.index, columns=['rep_score'])

    return rep_score

def diversity_query_strategy_euc(x_unlabeled, num_clusters, random_state=42):
    """
    Calculate diversity scores of unlabeled data points based on average pairwise euclidean distances within clusters.

    Parameters:
    - x_unlabeled: DataFrame, the features of unlabeled data points.
    - num_clusters: int, the number of clusters to form for assessing diversity.
    - random_state: int, random state for reproducibility.

    Returns:
    - div_score: DataFrame, diversity scores for each data point in x_unlabeled.
    """
    
    if x_unlabeled.empty:
        raise ValueError("x_unlabeled is empty")

    if num_clusters <= 0:
        raise ValueError("num_clusters must be positive")
    
    # Cluster the unlabeled data into 'num_clusters' clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state)
    cluster_assignments = kmeans.fit_predict(x_unlabeled)

    diversity_scores = []

    # Calculate diversity scores as the average pairwise dissimilarity within each cluster
    for cluster_label in range(num_clusters):
        cluster_indices = np.where(cluster_assignments == cluster_label)[0]
        if len(cluster_indices) > 0:
            cluster_data = x_unlabeled.iloc[cluster_indices]
            dissimilarity_matrix = pairwise_distances(cluster_data, metric='euclidean')
            average_dissimilarity = np.mean(dissimilarity_matrix, axis=0)
            diversity_scores.extend(average_dissimilarity)

    # Normalize diversity scores
    diversity_scores = normalize_scores(diversity_scores)
    
    # Create a DataFrame with the same index as x_unlabeled
    div_score = pd.DataFrame(diversity_scores, index=x_unlabeled.index, columns=['div_score'])

    return div_score 

def diversity_query_strategy_cosine(x_unlabeled, num_clusters, random_state=42):
    """
    Calculate diversity scores of unlabeled data points based on average pairwise cosine dissimilarity within clusters.

    Parameters:
    - x_unlabeled: DataFrame, the features of unlabeled data points.
    - num_clusters: int, the number of clusters to form for assessing diversity.
    - random_state: int, random state for reproducibility.

    Returns:
    - div_score: DataFrame, diversity scores for each data point in x_unlabeled.
    """
    
    if x_unlabeled.empty:
        raise ValueError("x_unlabeled is empty")

    if num_clusters <= 0:
        raise ValueError("num_clusters must be positive")
    
    # Cluster the unlabeled data into 'num_clusters' clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state)
    cluster_assignments = kmeans.fit_predict(x_unlabeled)

    # Initialize diversity scores array
    diversity_scores = []

    # Calculate diversity scores as the average pairwise dissimilarity within each cluster
    for cluster_label in range(num_clusters):
        cluster_indices = np.where(cluster_assignments == cluster_label)[0]
        if len(cluster_indices) > 0:
            cluster_data = x_unlabeled.iloc[cluster_indices]
            similarity_matrix = cosine_distances(cluster_data)
            avg_cos_dist = np.mean(similarity_matrix, axis=0)
            dissimilarity_matrix = 1 - avg_cos_dist
            diversity_scores.extend(dissimilarity_matrix)

    # Normalize diversity scores
    diversity_scores = normalize_scores(diversity_scores)
    
    # Create a DataFrame with the same index as x_unlabeled
    div_score = pd.DataFrame(diversity_scores, index=x_unlabeled.index, columns=['div_score'])

    return div_score 

def get_unlabeled_predictions_std(model, x_unlabeled):
    """
    Computes the standard deviation of predictions for unlabeled data points using the trained model.
    The function first makes predictions on the unlabeled data points using the trained model. 
    It specifically requests the standard deviation of predictions, which quantifies the model's uncertainty about each prediction. 
    These standard deviations are then normalized to ensure consistency and comparability across different scales of data. 
    The normalized standard deviations are returned as a DataFrame, preserving the index from the `x_unlabeled` input, 
    which can be used to identify the corresponding data points.

    Parameters:
    - model: The trained Gaussian Process Regressor model or any other model that supports prediction with standard deviation.
    - x_unlabeled: DataFrame containing features of unlabeled data points.

    Returns:
    - A DataFrame containing the normalized standard deviations of predictions for each unlabeled data point, with the same index as `x_unlabeled`.
    """
    _, y_pred_std_unlabeled = model.predict(x_unlabeled, return_std=True)
    return pd.DataFrame(normalize_scores(y_pred_std_unlabeled), index=x_unlabeled.index)

def calculate_fusion_scores(AL_conf):
    """
    Calculate fusion scores for unlabeled data by combining representativeness, uncertainty, and diversity scores,
    and select indices based on these scores.

    Parameters:
    - AL_conf: A dictionary containing configuration parameters for active learning.

    Returns:
    - selected_unlabeled_indices: Indices of the selected unlabeled data points.
    """
    
    # Calculate representativeness scores
    rep_scores = representative_scores(AL_conf['x_unlabeled'], AL_conf['num_clusters'])  

    # Combine representativeness (rep) and uncertainty (std) scores
    combined_scores = ((1 - AL_conf['alpha']) * AL_conf['y_pred_std_unlabeled'].iloc[:, 0]) + (AL_conf['alpha'] * rep_scores.iloc[:, 0])

    # Determine the type of diversity score to use based on the query string
    if("wifi_EUC" in AL_conf['query']):
        div_scores = diversity_query_strategy_euc(AL_conf['x_unlabeled'], AL_conf['num_clusters'])  
    elif("wifi_COS" in AL_conf['query']):
        div_scores = diversity_query_strategy_cosine(AL_conf['x_unlabeled'], AL_conf['num_clusters'])  

    # Combine the combined scores with the diversity scores
    fusion_scores = ((1 - AL_conf['beta']) * combined_scores) + (AL_conf['beta'] * div_scores.iloc[:, 0])

    # Select indices based on the highest fusion scores
    selected_unlabeled_indices = fusion_scores.sort_values().index[-AL_conf['num_points_to_add']:]

    return selected_unlabeled_indices

def select_unlabeled_indices(iteration, conf, AL_conf, eval_conf):
    """
    Selects unlabeled indices based on the active learning query strategy.

    Parameters:
    - iteration (int): Current iteration number.
    - conf (dict): Configuration dictionary for the active learning process.
    - AL_conf (dict): Configuration dictionary specific to the active learning query strategy.
    - eval_conf (dict): Configuration dictionary for evaluation.

    Returns:
    - tuple: A tuple containing selected unlabeled indices, alpha, beta, and epsilon.
    """
    
    selected_unlabeled_indices = []
        
    if(AL_conf['query']=="random"):
        # Random selection of unlabeled indices
        selected_unlabeled_indices = np.random.choice(AL_conf['unlabeled_indices'], size=AL_conf['num_points_to_add'], replace=False)
        
    elif(AL_conf['query'] in ["uncertainty", "representativeness", "diversitycosine", "diversityeuclidean"]):
        # Select unlabeled indices based on predefined query strategies
        score_function = {
            "uncertainty": lambda: AL_conf['y_pred_std_unlabeled'].iloc[:, 0],
            "representativeness": lambda: representative_scores(AL_conf['x_unlabeled'], AL_conf['num_clusters']).iloc[:, 0],
            "diversitycosine": lambda: diversity_query_strategy_cosine(AL_conf['x_unlabeled'], AL_conf['num_clusters']).iloc[:, 0],
            "diversityeuclidean": lambda: diversity_query_strategy_euc(AL_conf['x_unlabeled'], AL_conf['num_clusters']).iloc[:, 0],
        }[AL_conf['query']]
        scores = score_function()
        selected_unlabeled_indices = scores.sort_values().index[-AL_conf['num_points_to_add']:]
        
    elif(AL_conf['query'] in ["merge_fusion_cosine", "merge_fusion_euclidean"]):
        # Determine the diversity function based on the query type
        diversity_func = diversity_query_strategy_cosine if AL_conf['query'] == "merge_fusion_cosine" else diversity_query_strategy_euc
        # Calculate representative and diversity scores
        representative_scores_ = representative_scores(AL_conf['x_unlabeled'], AL_conf['num_clusters'])
        diversity_scores_ = diversity_func(AL_conf['x_unlabeled'], AL_conf['num_clusters'])
        # Determine the number of points to select for each query
        n_points = int(AL_conf['num_points_to_add'] / 3)
        # Select unlabeled indices
        selected_unlabeled_indices1 = AL_conf['y_pred_std_unlabeled'].iloc[:, 0].sort_values().index[-n_points:]
        selected_unlabeled_indices2 = representative_scores_.iloc[:, 0].sort_values().index[-n_points:]
        selected_unlabeled_indices3 = diversity_scores_.iloc[:, 0].sort_values().index[-n_points:]
        selected_unlabeled_indices = np.concatenate((selected_unlabeled_indices1, selected_unlabeled_indices2, selected_unlabeled_indices3))
            
    elif(AL_conf['query'] in ["wifi_COS_epDecay", "wifi_EUC_epDecay"]):
        # Initialize alpha and beta at the beginning based on the query type
        if(iteration == 0):
            if(AL_conf['query'] == "wifi_COS_epDecay"):
                AL_conf['alpha'], AL_conf['beta'] = 0.5, 0.5
            elif AL_conf['query'] == "wifi_EUC_epDecay":
                AL_conf['alpha'], AL_conf['beta'] = 0.5, 0.4
        # Adaptive weight adjustment at specified iterations
        if(iteration % 15 == 0):
            AL_conf['alpha'], AL_conf['beta'] = adaptive_weights(AL_conf, eval_conf, conf)
            AL_conf['epsilon'] *= conf['AL']['decay_rate']  # Update epsilon using decay
        selected_unlabeled_indices = calculate_fusion_scores(AL_conf) 
        
    return selected_unlabeled_indices, AL_conf['alpha'], AL_conf['beta'], AL_conf['epsilon']

def eval_adaptive(active_learning_config, evaluation_config, alpha, beta, conf, e_metric="r2"):
    """
    Evaluate scores for adaptive weights algorithm based on a specified evaluation metric.

    Parameters:
    - active_learning_config: A dictionary containing configuration parameters for active learning.
    - evaluation_config: A dictionary containing evaluation configuration parameters.
    - conf: Configuration dictionary for the active learning process.
    - e_metric: Evaluation metric to use ('r2', 'ssim', 'mape', 'rmspe').

    Returns:
    - score: The calculated score based on the chosen evaluation metric.
    """

    active_learning_config['alpha'] = alpha
    active_learning_config['beta'] = beta

    # Select indices based on fusion scores
    selected_indices = calculate_fusion_scores(active_learning_config)

    # Combine selected data with labeled data pool
    xy_train_combined = pd.concat([active_learning_config['D_xy_train'],
                                   active_learning_config['unlabeled_data'].loc[selected_indices]],
                                  axis=0)

    # Prepare training and test data
    X_train, y_train, X_test = prepare_train_test_data(xy_train_combined, evaluation_config['X_test'], conf)

    # Train model and make predictions
    _, y_pred = train_and_predict(X_train, y_train, X_test)

    # # Evaluate predictions
    r2, ssim, mape, rmspe, _ = evaluate_model(y_pred, evaluation_config['y_test_01'], conf, evaluation_config['z_min'], evaluation_config['z_max'])

    # Map evaluation metric to corresponding score
    metric_to_score = {'r2': r2, 'ssim': ssim, 'mape': 1 / mape, 'rmspe': 1 / rmspe}

    return metric_to_score.get(e_metric, "Invalid metric")

def clamp_value(value):
    """
    Ensures a given value is clamped between 0 and 1.

    Parameters:
    - value: The value to be clamped.

    Returns:
    - The clamped value, which will be no less than 0 and no more than 1.
    """
    return max(0, min(value, 1))

def adaptive_weights(AL_config, evaluation_config, conf):
    """
    Adjusts the alpha and beta weights adaptively based on their impact on the chosen metric.

    Parameters:
    - AL_config: A dictionary containing active learning configuration parameters.
    - evaluation_config: A dictionary containing evaluation configuration parameters.
    - conf: Configuration dictionary for the active learning process.

    Returns:
    - Updated alpha, beta after adaptive adjustment.
    """

    alpha = AL_config['alpha']
    beta = AL_config['beta']
    
    # Generate candidate values for alpha and beta weights
    alpha_values = [alpha - 0.15, alpha, alpha + 0.15]
    beta_values = [beta- 0.15, beta, beta + 0.15]

    # Evaluate R2 for each alpha, keeping beta constant
    r2_alpha = [eval_adaptive(AL_config, evaluation_config, clamp_value(a), beta, conf) for a in alpha_values]

    # Evaluate R2 for each beta, keeping alpha constant
    r2_beta = [eval_adaptive(AL_config, evaluation_config,  alpha, clamp_value(b), conf) for b in beta_values]

    # Determine which hyperparameter to update more
    if max(r2_alpha) > max(r2_beta):
        best_index = r2_alpha.index(max(r2_alpha))
        alpha *= (1 + AL_config['epsilon'] * alpha_values[best_index])
    else:
        best_index = r2_beta.index(max(r2_beta))
        beta *= (1 + AL_config['epsilon'] * beta_values[best_index])

    # Ensure alpha and beta are clamped between 0 and 1
    AL_config['alpha'] = clamp_value(alpha)
    AL_config['beta'] = clamp_value(beta)

    return  AL_config['alpha'], AL_config['beta']
