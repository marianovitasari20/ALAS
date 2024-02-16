from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import pickle

def get_gp_model(X_train, y_train, kernel=None, pickle_path=None):
    """
    Train and return a Gaussian Process Regressor model.
    
    Parameters:
    - X_train: Training features.
    - y_train: Training targets.
    - kernel: Optional kernel for the Gaussian Process.
    - pickle_path: Optional path to save the trained model scaler.
    
    Returns:
    - A trained GaussianProcessRegressor model.
    """
    
    if kernel is None:
        kernel = 1.0 * RBF() + WhiteKernel(noise_level=0.09)

    model = GaussianProcessRegressor(kernel=kernel)
    model.fit(X_train, y_train)

    if pickle_path:
        with open(pickle_path, 'wb') as f:
            pickle.dump(model, f)

    return model
