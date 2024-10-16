import numpy as np
from defense_utils import *

def krum(client_state_dicts, num_clients, num_byzantine_clients):
    """
    Apply Krum to a list of client updates in the form of state_dicts with LoRA parameters.
    :param client_state_dicts: List of state_dicts, where each state_dict contains LoRA parameters for a client.
    :param num_clients: Total number of clients.
    :param num_byzantine_clients: Number of suspected Byzantine (malicious) clients.
    :return: Index of the client whose update should be selected as the global update.
    """
    flattened_updates = [flatten_lora_params(state_dict) for state_dict in client_state_dicts]
    
    num_good_clients = num_clients - num_byzantine_clients - 2  # Krum requirement
    distances = np.zeros((num_clients, num_clients))  # Distance matrix
    
    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            distances[i][j] = np.linalg.norm(flattened_updates[i] - flattened_updates[j])
            distances[j][i] = distances[i][j]
    
    krum_scores = []
    for i in range(num_clients):
        # exclude the client itself
        sorted_distances = np.sort(distances[i][distances[i] != 0])
        krum_score = np.sum(sorted_distances[:num_good_clients])
        krum_scores.append(krum_score)# Index of the chosen client update

    return np.argmin(krum_scores)  
def multi_krum(client_state_dicts, num_clients, num_byzantine_clients, n):
    """ 
    Apply Multi-Krum to a list of client updates in the form of state_dicts with LoRA parameters.
    :param client_state_dicts: List of state_dicts, where each state_dict contains LoRA parameters for a client.
    :param num_clients: Total number of clients.
    :param num_byzantine_clients: Number of suspected Byzantine (malicious) clients.
    :param n: Number of clients to select from the Multi-Krum set.
    """
    flattened_updates = [flatten_lora_params(state_dict) for state_dict in client_state_dicts]
    
    num_good_clients = num_clients - num_byzantine_clients - 2  # Krum requirement
    distances = np.zeros((num_clients, num_clients))  # Distance matrix
    
    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            distances[i][j] = np.linalg.norm(flattened_updates[i] - flattened_updates[j])
            distances[j][i] = distances[i][j]
    
    krum_scores = []
    for i in range(num_clients):
        # exclude the client itself
        sorted_distances = np.sort(distances[i][distances[i] != 0])
        krum_score = np.sum(sorted_distances[:num_good_clients])
        krum_scores.append(krum_score)
    
    multi_krum_set = np.argsort(krum_scores)[:n]  # Multi-Krum set
    return multi_krum_set

def detect_anomalies_by_distance(distances, method='sum', threshold=0.002):
    """ 
    Detect outlier clients based on the total distance across all layers.
    :param clean_B_matrices: Clean B matrices from the global model.
    :param client_state_dicts: List of state_dicts, where each state_dict contains LoRA parameters for a client.
    :param method: Method to calculate the total distance ('sum', 'max', or 'mean').
    :param threshold: Threshold for detecting outliers.
    :return: List of indices of outlier clients.
    """
    
    outlier_clients = []
    # For each client, calculate the total distance across all layers
    client_distance = [0.0] * len(distances[next(iter(distances.keys()))])
    for layer_key in distances.keys():
        if method == 'sum':
            for i, distance in enumerate(distances[layer_key]):
                client_distance[i] += distance
        elif method == 'max':
            for i, distance in enumerate(distances[layer_key]):
                client_distance[i] = max(client_distance[i], distance)
        elif method == 'mean':
            for i, distance in enumerate(distances[layer_key]):
                client_distance[i] += distance / len(distances)
    # find the outlier clients
    for i, distance in enumerate(client_distance):
        if distance > threshold:
            outlier_clients.append(i)
    return outlier_clients