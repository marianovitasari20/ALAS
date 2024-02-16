import numpy as np
import shap
from active_learning.strategies import get_unlabeled_predictions_std, select_unlabeled_indices
from evaluation.metrics import evaluate_model, update_metrics_history
from utils.data_preprocessing import prepare_test_data, prepare_y_test
from models.model_training import prepare_train_test_data, train_and_predict
from visualization.plot_AL import plot_shap_summary

class ALProcess:
    def __init__(self, Xy_pool, X_test, y_test, selected_indices, query, conf):
        """
        Initializes the active learning process with the necessary data and configurations.
        
        Parameters:
        - Xy_pool: The pool of unlabeled data available for querying.
        - X_test: The test dataset used for evaluation.
        - y_test: The true labels for the test dataset.
        - selected_indices: Indices of samples already selected from Xy_pool.
        - query: The querying strategy for active learning.
        - conf: Configuration settings for the active learning process.
        """
        
        self.Xy_pool = Xy_pool
        self.X_test = X_test
        self.y_test = y_test
        self.selected_indices = selected_indices
        self.query = query
        self.conf = conf
        self.metrics_history = {'r2': [], 'ssim': [], 'mape': [], 'rmspe': [], 'alpha': [], 'beta': []}

    def perform_shap_feature_selection(self):
        """
        Performs SHAP-based feature selection every 15 iterations of the active learning process.
        Given the limited amount of labeled data available in an active learning context, this method
        uses the currently labeled data in the pool to train a model. It then computes SHAP values
        for the test set using the trained model, with the labeled training data serving as the 
        background dataset for the SHAP KernelExplainer. This approach helps in understanding 
        feature importance based on the limited labeled data and adjusts the feature set 
        accordingly by retaining only those deemed important based on the SHAP values.
        """
        
        # Training a model using currently labeled data
        xy_train = self.Xy_pool.loc[self.selected_indices]
        x_train, y_train, x_test = prepare_train_test_data(xy_train, self.X_test, self.conf)
        model, _ = train_and_predict(x_train, y_train, x_test)
        
        # Setting up SHAP explainer 
        explainer = shap.KernelExplainer(model.predict, x_train)
        
        # Computing SHAP values for test set features, limiting samples to reduce computation
        shap_values = explainer.shap_values(x_test,n_samples=25)

        # Determine which features have a mean absolute SHAP value above the threshold
        mean_abs_shap = np.abs(np.squeeze(shap_values)).mean(axis=0)
        important_features_mask = mean_abs_shap > self.conf['AL']['SHAP_threshold']

        # Visualizing SHAP values to understand feature contributions
        features = x_train.columns.tolist()
        plot_shap_summary(shap_values, x_test, features)
        
        # Ensure important_features_mask is at least 1D
        important_features_mask = np.atleast_1d(important_features_mask)   
        
        # Check if any feature is marked as important, if not select the feature with the maximum SHAP value
        if not important_features_mask.any():
            max_shap_index = np.argmax(mean_abs_shap) 
            important_features_mask[max_shap_index] = True 
            
        # # Check if any feature is marked as important, if not raise an exception
        # if not important_features_mask.any():
        #     raise ValueError("No features satisfy the criteria. Please adjust the SHAP threshold.")  
        
        # Get the list of important features
        important_features = [feature for feature, is_important in zip(features, important_features_mask) if is_important]
        print("list of important_features:", important_features)
        
        # Add the target variable to the list of features to keep
        features_to_keep = important_features + [self.conf['target']['y_target']]
        
        # Updating Xy_pool and X_test
        self.Xy_pool = self.Xy_pool[features_to_keep]
        self.X_test = self.X_test[important_features]

    def run(self): 
        """
        Executes the active learning loop, performing training, evaluation, and dynamic feature selection based on SHAP values.
        
        Parameters:
        - self: An instance of the ALProcess class.
        
        Returns:
        - A dictionary summarizing the selected indices and evaluation metrics over the active learning iterations.
        """    
        
        y_test_01, z_min, z_max = prepare_y_test(self.y_test)
        alpha, beta, epsilon = -99, -99, self.conf['AL']['epsilon']
        original_Xy_pool = self.Xy_pool
        original_X_test = self.X_test

        # Active learning loop
        for iteration in range(self.conf['AL']['num_AL_steps']):
            # Perform SHAP-based feature selection every 15 iterations
            if(iteration % 15 == 0):
                self.Xy_pool = original_Xy_pool 
                self.X_test = original_X_test
                self.perform_shap_feature_selection()
            
            # Select currently labeled data for training
            xy_train = self.Xy_pool.loc[self.selected_indices]
            
            # Prepare training and test data
            x_train, y_train, x_test = prepare_train_test_data(xy_train, self.X_test, self.conf)
            
            # Train model and predict on test set
            model, y_pred = train_and_predict(x_train, y_train, x_test)
            
            # Evaluate model performance
            r2, ssim, mape, rmspe, _ = evaluate_model(y_pred, y_test_01, self.conf, z_min, z_max)

            # Remove labeled data points from the unlabeled data pool
            unlabeled_indices = list(set(self.Xy_pool.index) - set(self.selected_indices))
            
            unlabeled_data = self.Xy_pool.loc[unlabeled_indices]
            x_unlabeled = unlabeled_data.drop(columns=[self.conf['target']['y_target']])
            x_unlabeled = prepare_test_data(unlabeled_data.drop(columns=[self.conf['target']['y_target']]), self.conf['pickle_paths']['pick_coef'])
            
            # Get standard deviation of predictions for unlabeled data
            y_pred_std_unlabeled = get_unlabeled_predictions_std(model, x_unlabeled)

            # Define the configuration for active learning in this iteration
            AL_conf = {
                'D_xy_train': xy_train,
                'unlabeled_data': unlabeled_data,
                'x_unlabeled': x_unlabeled,
                'num_clusters': self.conf['AL']['num_clusters'],
                'y_pred_std_unlabeled': y_pred_std_unlabeled,
                'alpha': alpha,
                'beta': beta,
                'query': self.query,
                'num_points_to_add': self.conf['AL']['num_points_to_add'],
                'unlabeled_indices': unlabeled_indices,
                'epsilon': epsilon
            }

            # Define the evaluation configuration for this iteration
            eval_conf = {
                'X_test': self.X_test,
                'y_test_01': y_test_01,
                'z_min': z_min,
                'z_max': z_max,
            }
            
            # Select new indices to label based on the active learning strategy
            selected_unlabeled_indices, alpha, beta, epsilon = select_unlabeled_indices(iteration, self.conf, AL_conf, eval_conf)
            
            # Update metrics history
            update_metrics_history(self.metrics_history, r2, ssim, mape, rmspe, alpha, beta)
            
            # Update the list of selected indices with newly selected indices
            self.selected_indices = np.concatenate((self.selected_indices, selected_unlabeled_indices))

        return {
            '_selected_indices_list': self.selected_indices,
            '_r2_list': self.metrics_history['r2'],
            '_ssim_list': self.metrics_history['ssim'],
            '_mape_list': self.metrics_history['mape'],
            '_rmspe_list': self.metrics_history['rmspe'],
            '_alpha_list': self.metrics_history['alpha'],
            '_beta_list': self.metrics_history['beta']
        }
