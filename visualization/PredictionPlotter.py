import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def create_custom_colormap(cmap_name, colors, n_bins):
    """
    Create a custom colormap using LinearSegmentedColormap.

    Parameters:
    - cmap_name (str): The name of the colormap.
    - colors (list): List of colors for the colormap.
    - n_bins (int): Number of bins for the colormap.

    Returns:
    - LinearSegmentedColormap: Custom colormap object.
    """
    return LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

class PredictionPlotter:
    def __init__(self, df_plot, df_diff, LOG_NORM, conf):
        """
        Initializes the PredictionPlotter with necessary data and configurations.

        Parameters:
        - df_plot: DataFrame containing 'lon', 'lat', 'groundtruth', and 'prediction' columns for plotting groundtruth and prediction.
        - df_diff: DataFrame containing 'lon', 'lat', 'groundtruth', and 'prediction' columns for plotting differences.
        - LOG_NORM: A matplotlib.colors.Normalize instance to normalize the color scale in log scale.
        - conf: Configuration dictionary containing plot settings and parameters.
        """
        
        self.df_plot = df_plot
        self.df_diff = df_diff
        self.dpi = 200 
        self.conf = conf
        self.LOG_NORM = LOG_NORM
        self.MY_CMAP = create_custom_colormap("my_cmap", conf['plot']['my_colors'], 100)

    @staticmethod
    def percent_formatter(x, pos):
        """Formats tick labels as percentages."""
        return f"{x*100:.0f}%"

    def create_figure(self):
        """
        Creates a matplotlib figure with subplots for groundtruth, predictions, and their differences.

        Returns:
        - fig: The matplotlib figure object containing the subplots.
        """
        
        # fig, axs = plt.subplots(ncols=3, figsize=(10.3, 2.9), dpi=self.dpi, 
        fig, axs = plt.subplots(ncols=3, figsize=(10.5, 2.6), dpi=self.dpi,
        # fig, axs = plt.subplots(ncols=3, figsize=(10.5, 1.9), dpi=self.dpi, 
                                sharey=True, sharex=True,
                                gridspec_kw={'width_ratios': [0.8, 1, 1]},
                                subplot_kw={'projection': ccrs.PlateCarree()})
        
        self.plot_groundtruth(axs[0])  
        self.plot_prediction(axs[1])    
        self.plot_difference(axs[2])
        
        return fig

    def plot_groundtruth(self, ax):
        """
        Plots the groundtruth data on the specified axes.

        Parameters:
        - ax: The matplotlib axes to plot on.
        """
        
        ax.scatter(self.df_plot['lon'], self.df_plot['lat'], 
                   c=self.df_plot['groundtruth'], cmap=self.MY_CMAP, norm=self.LOG_NORM, s=0.3)
        
        ax.set_title('Groundtruth (g m$^{-3}$ h$^{-1}$)')
        ax.set_xlabel('lon')
        ax.set_ylabel('lat')
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=1.5)
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linestyle=':', linewidth=1.5)
        
        gl = ax.gridlines(draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False
              
    def plot_prediction(self, ax):
        """
        Plots the prediction data on the specified axes.

        Parameters:
        - ax: The matplotlib axes to plot on.
        """
        
        fig = ax.scatter(self.df_plot['lon'], self.df_plot['lat'],
                  c=self.df_plot['prediction'], cmap=self.MY_CMAP, norm=self.LOG_NORM, s=0.3)    

        ax.set_title('Prediction (g m$^{-3}$ h$^{-1}$)')
        ax.set_xlabel('lon')
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=1.5)
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linestyle=':', linewidth=1.5)
        
        gl = ax.gridlines(draw_labels=True)
        gl.top_labels = False
        gl.left_labels = False
        gl.right_labels = False
        gl.ylabels_bottom = False
        
        plt.colorbar(fig, ax=ax)

    def plot_difference(self, ax):
        """
        Plots the difference between groundtruth and prediction data on the specified axes.

        Parameters:
        - ax: The matplotlib axes to plot on.
        """
        
        diff = self.df_diff['groundtruth'] - self.df_diff['prediction']
        fig = ax.scatter(self.df_diff['lon'], self.df_diff['lat'], c=diff, cmap=self.conf['plot']['cmap_diff'], vmin=-0.2, vmax=0.2, s=0.3)
        
        ax.set_title('Difference')
        ax.set_xlabel('lon')
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=1.5)
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linestyle=':', linewidth=1.5)
        
        gl = ax.gridlines(draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False
        gl.ylabels_bottom = False
        gl.left_labels = False
        
        plt.colorbar(fig, ax=ax, format=ticker.FuncFormatter(self.percent_formatter))
       
