# ALAS: Active Learning for Autoconversion Rates Prediction from Satellite Data

This repository contains the code developed for the paper "**ALAS: Active Learning for Autoconversion Rates Prediction from Satellite Data**," as published in **AISTATS 2024**, with the following authors: <br>
- **Maria Carolina Novitasari (University College London)**
- **Johannes Quaas (Universität Leipzig)**
- **Miguel R. D. Rodrigues (University College London)**

While our work is specifically designed for autoconversion rates prediction, the code can be repurposed for a broad range of other applications.

## Abstract
High-resolution simulations, such as the ICOsahedral Non-hydrostatic Large-Eddy Model (ICON-LEM), provide valuable insights into the complex interactions among aerosols, clouds, and precipitation, which are the major contributors to climate change uncertainty. However, due to their exorbitant computational costs, they can only be employed for a limited period and geographical area. To address this, we propose a more cost-effective method powered by an emerging machine learning approach to better understand the intricate dynamics of the climate system. Our approach involves active learning techniques by leveraging high-resolution climate simulation as an oracle that is queried based on an abundant amount of unlabeled data drawn from satellite observations. In particular, we aim to predict autoconversion rates, a crucial step in precipitation formation, while significantly reducing the need for a large number of labeled instances. In this study, we present novel methods: custom query strategy fusion for labeling instances -- weight fusion (WiFi) and merge fusion (MeFi) -- along with active feature selection based on SHAP. These methods are designed to tackle real-world challenges -- in this case, climate change, with a specific focus on the prediction of autoconversion rates -- due to their simplicity and practicality in application.

<p align="center">
  <img src="https://github.com/marianovitasari20/ALAS/assets/124182131/cdde0120-2704-4390-9fd9-617af2019924" width="30%" style="margin-right: 10px;"/>
  <img src="https://github.com/marianovitasari20/ALAS/assets/124182131/cc6f5bba-88db-470d-becc-ae9412b39418" width="30%" />
</p>

<p align="center">
  <img src="https://github.com/marianovitasari20/ALAS/assets/124182131/9402096c-0ebc-43d5-97e9-fa831df4cfbd" width="100%" />
</p>



## Setup
To run the code from this repository, you will need to have Python installed along with several additional libraries. 
The required libraries include `numpy`, `shap`, `scikit-learn` (commonly imported as `sklearn`), `pandas`, `matplotlib`, `cartopy`, and `seaborn`. 

### Using pip

You can install these dependencies using `pip`, Python's package installer. Running the following command in your terminal should install all the necessary libraries:

```sh
pip install numpy shap scikit-learn pandas matplotlib seaborn
```

Note: Cartopy has some dependencies that might need to be installed separately and might be better handled with conda. If you encounter difficulties installing Cartopy with pip, consider using conda.

### Using conda (default channel)

If you prefer using `conda`, you can install most of these packages from the default Anaconda channels. To install the required libraries using `conda`, execute the following command:

```sh
conda install numpy shap scikit-learn pandas matplotlib seaborn cartopy
```

### Using conda-forge

For some packages, the `conda-forge` channel might offer more up-to-date versions or better compatibility. To install the required libraries from `conda-forge`, use:

```sh
conda install -c conda-forge numpy shap scikit-learn pandas matplotlib seaborn cartopy
```


## Run

To run experiments related to the **active learning** component, please execute `main_AL.py`. Prior to running, ensure you modify the corresponding configuration file `config_AL.py` as needed.

For experiments on **autoconversion rates prediction**, please execute `main_aut.py` and adjust the corresponding configuration file `config_aut.py` before running.

## Citation
If you use any part of this code, whether in whole or in part, as a component of your project or research, please cite the following paper:

_to be published soon by AISTATS 2024_
