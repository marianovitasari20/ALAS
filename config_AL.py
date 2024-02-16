configurations = {
    "pickle_paths": {
        "pick_coef": '/path/to/save/AL_coef.pkl', 
        "pick_coef_y": '/path/to/save/AL_coef_y.pkl' 
    },
    "target": {
        "y_target": "aut_rate"
    },
    "AL": {
        "initial_num_points": 10,
        "num_points_to_add": 3,
        "num_AL_steps": 40,
        "num_clusters": 2,
        "decay_rate": 0.75,
        "epsilon": 0.25,
        "num_exp": 100,
        "SHAP_threshold": 0.5,
        "n_samples": 1000,
        "n_val_samples": 250
    },
    "main": {
        "AL_queries": ["random", "uncertainty", "representativeness", "diversitycosine", "diversityeuclidean", "wifi_EUC_epDecay", "wifi_COS_epDecay", "merge_fusion_euclidean", "merge_fusion_cosine"],
        "AL_queries_labels": ["random", r"$l_{\mathrm{inf}}$", r"$l_{\mathrm{rep}}$", r"$l_{\mathrm{div}}$"+ " cos", r"$l_{\mathrm{div}}$"+" euc", "WiFi euc", "WiFi cos", "MeFi euc", "MeFi cos"],
        "folder_path": '/folder/path/to/save',
        "file_name": "results_dict.pkl"
    }
}

