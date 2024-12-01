import numpy as np

def calculate_entropy(y):
    probabilities = np.bincount(y) / len(y)
    probabilities = probabilities[probabilities > 0]
    return -np.sum(probabilities * np.log2(probabilities))

def calculate_conditional_entropy(X, y):
    unique_values = np.unique(X)
    conditional_entropy = 0
    for value in unique_values:
        mask = (X == value)
        subset_y = y[mask]
        conditional_entropy += (len(subset_y) / len(y)) * calculate_entropy(subset_y)
    return conditional_entropy

def calculate_information_gain(X, y):
    base_entropy = calculate_entropy(y)
    info_gains = []
    for i in range(X.shape[1]):
        conditional_entropy = calculate_conditional_entropy(X[:, i], y)
        info_gains.append(base_entropy - conditional_entropy)
    return np.array(info_gains)

def select_top_features(X, y, feature_names, top_n=2):
    info_gain_scores = calculate_information_gain(X, y)
    sorted_indices = np.argsort(info_gain_scores)[::-1]
    selected_feature_names = [feature_names[i] for i in sorted_indices[:top_n]]
    return selected_feature_names
