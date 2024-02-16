import os
import pickle

def save_AL_results(results_dict, folder_path, file_name):
    """
    Saves the provided results dictionary to a specified file within a given folder path. 
    If the folder does not exist, it will be created.

    Parameters:
    - results_dict (dict): A dictionary containing the results of an Active Learning experiment. 
    - folder_path (str): The path to the folder where the results file will be saved. If the folder 
                           does not exist, it will be created by this function.
    - file_name (str): The name of the file (including the extension, usually `.pkl`) where the 
                         results dictionary will be saved. The file will be created in the specified 
                         `folder_path`.
    """
    
    # Check if the folder exists
    if not os.path.exists(folder_path):
        # If it doesn't exist, create it
        os.makedirs(folder_path)
        print(f"The folder '{folder_path}' was created.")
    else:
        print(f"The folder '{folder_path}' already exists.")

    # Save the results_dict to a file within the folder
    with open(os.path.join(folder_path, file_name), 'wb') as file:
        pickle.dump(results_dict, file)


def load_AL_results(folder_path, file_name):
    """
    Loads the results dictionary from a specified file within a given folder path. 

    Parameters:
    - folder_path (str): The path to the folder where the results file is saved.
    - file_name (str): The name of the file (including the extension, usually `.pkl`) from which 
                       the results dictionary will be loaded. The file is assumed to be located 
                       in the specified `folder_path`.

    Returns:
    - results_dict (dict): The loaded dictionary containing the results of an Active Learning 
                           experiment. If the file or folder does not exist, a FileNotFoundError 
                           will be raised.
    """

    # Construct the full file path by joining the folder path and file name
    full_file_path = os.path.join(folder_path, file_name)

    # Check if the file exists at the specified path
    if not os.path.exists(full_file_path):
        # If the file does not exist, raise an error
        raise FileNotFoundError(f"The file '{full_file_path}' does not exist.")

    # Load and return the results_dict from the specified file
    with open(full_file_path, 'rb') as file:
        results_dict = pickle.load(file)

    return results_dict
