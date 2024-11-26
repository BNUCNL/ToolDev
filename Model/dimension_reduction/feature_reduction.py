import os
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA, NMF


def dimensionality_reduction(feature_matrix, method='PCA', n_components=60, model_path=None):
    """
    Perform dimensionality reduction on the input feature matrix using either PCA or NMF.

    Parameters:
    - feature_matrix (numpy.ndarray): Input feature matrix to be reduced in dimension.
    - method (str): Dimensionality reduction method to use ('PCA' or 'NMF'). Default is 'PCA'.
    - n_components (int): Number of components to keep. Default is 60.
    - model_path (str): Path to save or load the trained model. If None, the model will not be saved.

    Returns:
    - reduced_features (numpy.ndarray): Transformed feature matrix with reduced dimensions.
    """
    # Ensure non-negative features for NMF
    if method == 'NMF':
        feature_matrix = np.maximum(0, feature_matrix)

    # Scale feature matrix
    scaler = MinMaxScaler()
    feature_matrix = scaler.fit_transform(feature_matrix)

    # Perform dimensionality reduction
    if method == 'PCA':
        transformer = PCA(n_components=n_components, random_state=7866)
    elif method == 'NMF':
        transformer = NMF(n_components=n_components, random_state=7866)
    else:
        raise ValueError("Invalid method. Please choose either 'PCA' or 'NMF'.")

    # Load or fit transformer
    if model_path and os.path.exists(model_path):
        transformer = joblib.load(model_path)
    else:
        transformer.fit(feature_matrix)
        if model_path:
            joblib.dump(transformer, model_path)

    # Transform feature matrix
    reduced_features = transformer.transform(feature_matrix)

    return reduced_features


# Example usage
if __name__ == "__main__":
    # Assuming `feature_matrix` is loaded or generated previously
    feature_matrix = np.random.rand(1000, 2048)  # Example input matrix
    reduced_features_pca = dimensionality_reduction(feature_matrix, method='PCA', n_components=60)
    reduced_features_nmf = dimensionality_reduction(feature_matrix, method='NMF', n_components=60, model_path='nmf_model.pkl')
    
    # Print shapes
    print("PCA reduced features shape:", reduced_features_pca.shape)
    print("NMF reduced features shape:", reduced_features_nmf.shape)
