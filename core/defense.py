import numpy as np
import torch
from defense_utils import *
from scipy.stats import trim_mean


def krum(client_state_dicts, num_clients, num_byzantine_clients):
    """
    Apply Krum to a list of client updates in the form of state_dicts with LoRA parameters.
    :param client_state_dicts: List of state_dicts, where each state_dict contains LoRA parameters for a client.
    :param num_clients: Total number of clients.
    :param num_byzantine_clients: Number of suspected Byzantine (malicious) clients.
    :return: Index of the client whose update should be selected as the global update.
    """
    flattened_updates = [flatten_lora_params(
        state_dict) for state_dict in client_state_dicts]

    num_good_clients = num_clients - num_byzantine_clients - 2  # Krum requirement
    distances = np.zeros((num_clients, num_clients))  # Distance matrix

    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            distances[i][j] = np.linalg.norm(
                flattened_updates[i] - flattened_updates[j])
            distances[j][i] = distances[i][j]

    krum_scores = []
    for i in range(num_clients):
        # exclude the client itself
        sorted_distances = np.sort(distances[i][distances[i] != 0])
        krum_score = np.sum(sorted_distances[:num_good_clients])
        krum_scores.append(krum_score)  # Index of the chosen client update
    # return the index of the client with the smallest Krum score as a list
    return [np.argmin(krum_scores)]


def multi_krum(client_state_dicts, num_clients, num_byzantine_clients, n):
    """ 
    Apply Multi-Krum to a list of client updates in the form of state_dicts with LoRA parameters.
    :param client_state_dicts: List of state_dicts, where each state_dict contains LoRA parameters for a client.
    :param num_clients: Total number of clients.
    :param num_byzantine_clients: Number of suspected Byzantine (malicious) clients.
    :param n: Number of clients to select from the Multi-Krum set.
    """
    flattened_updates = [flatten_lora_params(
        state_dict) for state_dict in client_state_dicts]

    num_good_clients = num_clients - num_byzantine_clients - 2  # Krum requirement
    distances = np.zeros((num_clients, num_clients))  # Distance matrix

    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            distances[i][j] = np.linalg.norm(
                flattened_updates[i] - flattened_updates[j])
            distances[j][i] = distances[i][j]

    krum_scores = []
    for i in range(num_clients):
        # exclude the client itself
        sorted_distances = np.sort(distances[i][distances[i] != 0])
        krum_score = np.sum(sorted_distances[:num_good_clients])
        krum_scores.append(krum_score)

    multi_krum_set = np.argsort(krum_scores)[:n]  # Multi-Krum set
    return multi_krum_set

def trimmed_mean(client_state_dicts, trim_ratio=0.1):
    """
    Apply Trimmed Mean to a list of client updates with LoRA parameters.
    :param client_state_dicts: List of state_dicts, where each state_dict contains LoRA parameters for a client.
    :param trim_ratio: Proportion of the extreme values to trim from each end.
    :return: Aggregated state_dict based on trimmed mean of client updates.
    """
    num_clients = len(client_state_dicts)
    trim_count = int(trim_ratio * num_clients)  # Number of clients to trim from each end
    
    # Extract LoRA parameters and initialize aggregated weights dictionary
    param_keys = client_state_dicts[0].keys()
    aggregated_weights = {}

    # Iterate over each parameter key in the client state_dicts
    for key in param_keys:
        # Stack weights for the current parameter from all clients
        param_values = np.array([client[key].cpu().numpy() for client in client_state_dicts])

        # Sort and trim the parameter values across clients
        sorted_values = np.sort(param_values, axis=0)
        trimmed_values = sorted_values[trim_count:num_clients - trim_count]  # Trim top and bottom values
        
        # Calculate mean of trimmed values
        trimmed_mean = np.mean(trimmed_values, axis=0)
        
        # Store the trimmed mean in the aggregated weights dictionary
        aggregated_weights[key] = torch.tensor(trimmed_mean).to(client_state_dicts[0][key].device)

    return aggregated_weights

def bulyan(client_state_dicts, num_clients, num_byzantine_clients, trim_ratio=0.1):
    """
    Apply Bulyan aggregation to a list of client updates.
    :param client_state_dicts: List of state_dicts with LoRA parameters for each client.
    :param num_clients: Total number of clients.
    :param num_byzantine_clients: Number of suspected Byzantine clients.
    :return: Aggregated update based on Bulyan's robust aggregation.
    """
    multi_krum_set = multi_krum(client_state_dicts, num_clients, num_byzantine_clients, n=num_clients - 2 * num_byzantine_clients)

    selected_updates = [client_state_dicts[i] for i in multi_krum_set]

    return trimmed_mean(selected_updates, trim_ratio=trim_ratio)

def rebuild_state_dict(base_state_dict, flattened_params):
    """
    Rebuild a state_dict from the flattened LoRA parameters.
    
    :param base_state_dict: A template state_dict containing the keys and structure.
    :param flattened_params: The flattened parameter array to restore.
    :return: The restored state_dict with LoRA parameters.
    """
    state_dict = {}
    param_offset = 0
    for key in base_state_dict:
        if 'lora_A' in key or 'lora_B' in key:
            param_shape = base_state_dict[key].shape
            num_params = np.prod(param_shape)
            # Extract the corresponding parameters for this key from the flattened array
            state_dict[key] = torch.tensor(flattened_params[param_offset:param_offset + num_params]).reshape(param_shape)
            param_offset += num_params
        else:
            state_dict[key] = base_state_dict[key]  # Keep non-LoRA parameters unchanged
    return state_dict


def detect_anomalies_by_distance(distances, layer_variances, alpha=2.0, base_weight=1.0, weight_factor=0.1, exponent=1.5, max_weight=3.0):
    """ 
    Detect outlier clients based on adaptive weighted distances with robust thresholding.
    :param distances: Dictionary of layer-wise distances between clean model's matrices and client matrices.
    :param layer_variances: Dictionary of variances for each layer.
    :param alpha: Scaling factor for calculating MAD-based thresholds.
    :param base_weight: Base weight for adaptive weight calculation.
    :param weight_factor: Factor controlling how much the adaptive weights increase per deviation.
    :param exponent: Exponent applied to deviation count to amplify weights.
    :param max_weight: Maximum allowed weight for adaptive weighting.
    :return: List of indices of outlier clients.
    """
    outlier_clients = []

    thresholds, deviation_counts, deviation_directions = calculate_layerwise_thresholds(distances, alpha=alpha)

    adaptive_weights = compute_adaptive_weights_directional(deviation_counts, deviation_directions,
                                                            base_weight=base_weight, weight_factor=weight_factor,
                                                            exponent=exponent, max_weight=max_weight)

    attention_weights = {layer: np.exp(var) for layer, var in layer_variances.items()}
    attention_sum = sum(attention_weights.values())
    normalized_attention = {layer: weight / attention_sum for layer, weight in attention_weights.items()}

    weighted_distances = compute_weighted_distance_with_attention(distances, normalized_attention)

    weighted_distances_with_adaptive = [dist * adaptive_weights[i] for i, dist in enumerate(weighted_distances)]

    robust_thresholds = calculate_robust_thresholds(weighted_distances_with_adaptive, alpha=alpha)

    for i, distance in enumerate(weighted_distances_with_adaptive):
        if distance > robust_thresholds['upper'] or distance < robust_thresholds['lower']:
            outlier_clients.append(i)

    return outlier_clients

def detect_outliers_from_weights(clean_model_state, client_state_dicts, num_layers=12, alpha=2.0, base_weight=1.0, weight_factor=0.1, exponent=1.5, max_weight=3.0):
    """
    High-level function to detect outliers directly from model weights.
    This function extracts LoRA parameters, flattens them, computes Wasserstein distances, and applies anomaly detection.
    
    :param clean_model_state: State dictionary of the clean global model.
    :param client_states: List of state dictionaries from each client model.
    :param num_layers: Number of layers to extract LoRA parameters from.
    :param alpha: Scaling factor for MAD-based threshold adjustment.
    :param base_weight: Base weight for adaptive weight calculation.
    :param weight_factor: Factor controlling the increase of adaptive weights based on deviations.
    :param exponent: Exponent applied to deviation count for weight amplification.
    :param max_weight: Maximum allowable adaptive weight.
    :return: List of outlier client indices.
    """
    
    # Step 1: Extract LoRA B matrices from the clean model and each client
    _, clean_B_matrices = extract_lora_matrices([clean_model_state], num_layers)
    _, client_B_matrices = extract_lora_matrices(client_state_dicts, num_layers)

    # Step 2: Compute Wasserstein distances for each layer and client
    distances = compute_wa_distances(clean_B_matrices, client_B_matrices)

    # Step 3: Calculate layer-wise variances for attention weights
    layer_variances = {layer: np.var(distances[layer]) for layer in distances.keys()}

    # Step 4: Detect outliers using the adaptive distance-based method
    outlier_clients = detect_anomalies_by_distance(distances, layer_variances, alpha=alpha,
                                                   base_weight=base_weight, weight_factor=weight_factor,
                                                   exponent=exponent, max_weight=max_weight)

    return outlier_clients
