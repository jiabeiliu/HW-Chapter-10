import numpy as np

class Normalization:
    
    # Min-Max Normalization
    def min_max_normalize(self, X):
        """
        Normalizes the input data using Min-Max scaling to a range [0, 1].
        
        :param X: numpy array, the input data to be normalized
        :return: numpy array, normalized data
        """
        X_min = np.min(X)
        X_max = np.max(X)
        X_norm = (X - X_min) / (X_max - X_min)
        return X_norm
    
    # Z-score Normalization
    def z_score_normalize(self, X):
        """
        Normalizes the input data using Z-score normalization (standardization).
        
        :param X: numpy array, the input data to be normalized
        :return: numpy array, normalized data
        """
        mean = np.mean(X)
        std_dev = np.std(X)
        X_norm = (X - mean) / std_dev
        return X_norm

# Example usage:
data = np.array([10, 20, 30, 40, 50])

normalizer = Normalization()

# Min-Max Normalization
min_max_normalized_data = normalizer.min_max_normalize(data)
print("Min-Max Normalized Data:", min_max_normalized_data)

# Z-Score Normalization
z_score_normalized_data = normalizer.z_score_normalize(data)
print("Z-Score Normalized Data:", z_score_normalized_data)
