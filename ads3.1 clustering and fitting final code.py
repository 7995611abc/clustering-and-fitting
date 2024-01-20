import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

# Function to read data using pandas
def read_data(API_E):
    """
    Read data from a CSV file using pandas.

    Parameters:
    - file_path (str): Path to the CSV file.

    Returns:
    - df (pd.DataFrame): Pandas DataFrame containing the data.
    """
    df = pd.read_csv(API_E)
    return df

# Function to transpose and clean the dataframe
def transpose_and_clean(df):
    """
    Transpose and clean the DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - transposed_df (pd.DataFrame): Transposed and cleaned DataFrame.
    """
    transposed_df = df.transpose()
    # Add cleaning steps if needed
    return transposed_df

# Function for clustering using sklearn
def perform_clustering(data, n_clusters):
    """
    Perform clustering on the given data using KMeans.

    Parameters:
    - data (pd.DataFrame): Input data for clustering.
    - n_clusters (int): Number of clusters.

    Returns:
    - cluster_labels (np.array): Cluster labels assigned by KMeans.
    - silhouette_avg (float): Silhouette score for the clustering.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(data)
    silhouette_avg = silhouette_score(data, cluster_labels)
    return cluster_labels, silhouette_avg

# Function for normalization and back scaling of cluster centers
def normalize_and_backscale(data):
    """
    Normalize and back scale the data.

    Parameters:
    - data (pd.DataFrame): Input data.

    Returns:
    - normalized_data (pd.DataFrame): Normalized and back-scaled data.
    """
    # Add normalization and back scaling steps if needed
    normalized_data = data
    return normalized_data

# Function for plotting clustering results
def plot_clustering(data, cluster_labels, centers):
    """
    Plot clustering results.

    Parameters:
    - data (pd.DataFrame): Input data.
    - cluster_labels (np.array): Cluster labels assigned by KMeans.
    - centers (np.array): Cluster centers.
    """
    plt.scatter(data[:, 0], data[:, 1], c=cluster_labels, cmap='viridis', label='Data Points')
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Cluster Centers')
    plt.title('Clustering Results of Nitrous oxide vs CO2 equivalent')
    plt.xlabel('Nitrous oxide emissions')
    plt.ylabel('thousand metric tons of CO2 equivalent')
    plt.legend()
    plt.show()
    
# Function for data fitting
def your_curve_function(x, a, b):
    """
    Sample curve function for curve fitting.

    Parameters:
    - x (np.array): Independent variable.
    - a, b: Parameters of the curve.

    Returns:
    - y (np.array): Dependent variable.
    """
    return a * x + b

def fit_data(x, y):
    """
    Fit data to a curve using curve_fit.

    Parameters:
    - x (np.array): Independent variable.
    - y (np.array): Dependent variable.

    Returns:
    - params (tuple): Parameters of the fitted curve.
    """
    params, _ = curve_fit(your_curve_function, x, y)
    return params

# Function for plotting fit results
def plot_fit(x, y, params, confidence_interval):

    """
    Plot the fit results.

    Parameters:
    - x (np.array): Independent variable.
    - y (np.array): Dependent variable.
    - params (tuple): Parameters of the fitted curve.
    - confidence_interval (tuple): Confidence interval for the fit.
    """
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = your_curve_function(x_fit, *params)
    plt.scatter(x, y, label='Data Points')
    plt.plot(x_fit, y_fit, color='red', label='Fit Curve')
    plt.fill_between(x_fit, y_fit - confidence_interval, y_fit + confidence_interval, color='gray', alpha=0.3, label='Confidence Interval')
    plt.title('Nitrous oxide vs CO2 equivalent Fitting Results')
    plt.xlabel('Nitrous oxide emissions')
    plt.ylabel('thousand metric tons of CO2 equivalent')
    plt.legend()
    plt.show()
      
# cluster usage:
x_line = np.linspace(0, 10, 100)
y_line = np.sin(x_line)
cluster_labels = np.random.randint(0, 3, size=len(x_line))  # cluster labels
cluster_centers = np.array([[2, 0], [5, 1], [8, -1]])  # cluster centers

# Function for making predictions with uncertainty
def make_predictions(model, x_values, num_samples=1000):
    """
    Make predictions with uncertainty.

    Parameters:
    - model: Trained model (e.g., regression model).
    - x_values (np.array): Input values for predictions.

    Returns:
    - predictions (np.array): Predicted values.
    - uncertainty (np.array): Uncertainty in predictions.
    """   
# Main code for reading data, performing analysis, and creating the poster
if __name__ == "__main__":
    
    # Student ID
    student_id = 22086338
    # Generate sample data for line plot
    
    # Abstract
    abstract = """
   This poster presents an analysis of climate change data using clustering and curve fitting techniques.
   The clustering section explores patterns in a two-dimensional dataset using KMeans clustering, highlighting
   the identified clusters and their centers. The curve fitting section demonstrates the fitting of a simple linear
   model to noisy data, showcasing the fitted curve along with the associated confidence interval."""
    
    x_line = np.linspace(0, 10, 100)
    y_line = np.sin(x_line)
    plot_fit(x_line, y_line, [1, 0], np.ones_like(x_line))

    # Generate sample data for clustering
    np.random.seed(42)
    data_cluster = np.random.rand(100, 2)  # Sample 2D data for clustering
    # Normalize data for clustering
    normalized_data_cluster = normalize_and_backscale(data_cluster)
    # Perform clustering
    cluster_labels, silhouette_avg = perform_clustering(data_cluster, n_clusters=3)

    # Plot clustering results
    cluster_centers = KMeans(n_clusters=3, random_state=42).fit(data_cluster).cluster_centers_
    plot_clustering(data_cluster, cluster_labels, cluster_centers)

    # Generate sample data for curve fitting
    x_fit = np.linspace(0, 10, 100)
    true_params = [2, 1]
    y_fit_true = your_curve_function(x_fit, *true_params)
    noise = np.random.normal(0, 1, size=len(x_fit))
    y_fit_noisy = y_fit_true + noise
    # Fit data to a curve
    fitted_params = fit_data(x_fit, y_fit_noisy)
    # Plot fit results
    confidence_interval = np.ones_like(x_fit)  # Replace with actual confidence interval
    plot_fit(x_fit, y_fit_noisy, fitted_params, confidence_interval)
