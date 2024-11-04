from scipy.stats import entropy, wasserstein_distance, median_abs_deviation
import numpy as np

def extract_lora_matrices(clients_state_dicts, num_layers):
    A_matrices = {f'Layer_{i+1}': [] for i in range(num_layers)}
    B_matrices = {f'Layer_{i+1}': [] for i in range(num_layers)}

    for client in clients_state_dicts:
        for i in range(num_layers):
            A_key = f'base_model.model.bert.encoder.layer.{i}.attention.self.query.lora_A.default.weight'
            B_key = f'base_model.model.bert.encoder.layer.{i}.attention.self.query.lora_B.default.weight'
            A_matrices[f'Layer_{i+1}'].append(client[A_key].cpu().numpy())
            B_matrices[f'Layer_{i+1}'].append(client[B_key].cpu().numpy())

    return A_matrices, B_matrices

def flatten_lora_params(state_dict):
    """
    Extract and flatten the LoRA parameters from a client's state_dict.
    :param state_dict: The state_dict of a client's model containing LoRA parameters.
    :return: A flattened numpy array of the LoRA parameters.
    """
    lora_params = []
    for key in state_dict:
        if 'lora_A' in key or 'lora_B' in key:
            lora_params.append(state_dict[key].cpu().numpy().ravel())  # Flatten each parameter
    
    return np.concatenate(lora_params)  # Concatenate all LoRA parameters into one vector

def kl_divergence(p, q, epsilon=1e-10):
    """Compute KL Divergence between two flattened distributions."""
    p = p.ravel() / np.sum(p.ravel())  # Normalize to get probability distributions
    q = q.ravel() / np.sum(q.ravel())
    
    p = np.clip(p, epsilon, 1)
    q = np.clip(q, epsilon, 1)
    
    return entropy(p, q)

def wasserstein_distance_between_matrices(p, q):
    """Compute Wasserstein Distance between two flattened distributions."""
    p_flat = p.ravel()
    q_flat = q.ravel()
    
    return wasserstein_distance(p_flat, q_flat)

def compute_kl_distances(clean_B_matrices, client_B_matrices):
    """
    Compute KL divergence between clean model's B matrices and each client's B matrices.
    :param clean_B_matrices: LoRA B matrices from the clean model.
    :param client_B_matrices: LoRA B matrices from client models.
    :return: Dictionary of KL divergences for each layer and each client.
    """
    kl_distances = {}

    for layer_key in clean_B_matrices.keys():
        clean_matrix = clean_B_matrices[layer_key][0].ravel()  # Clean model B matrix for the layer
        kl_distances[layer_key] = []

        for client_matrix in client_B_matrices[layer_key]:
            client_matrix_flat = client_matrix.ravel()
            kl_dist = kl_divergence(clean_matrix, client_matrix_flat)
            kl_distances[layer_key].append(kl_dist)

    return kl_distances

def compute_wa_distances(clean_B_matrices, client_B_matrices):
    """
    Compute Wasserstein Distance between clean model's B matrices and each client's B matrices.
    :param clean_B_matrices: LoRA B matrices from the clean model.
    :param client_B_matrices: LoRA B matrices from client models.
    :return: Dictionary of Wasserstein Distances for each layer and each client.
    """
    wa_distances = {}

    for layer_key in clean_B_matrices.keys():
        clean_matrix = clean_B_matrices[layer_key][0].ravel()  # Clean model B matrix for the layer
        wa_distances[layer_key] = []

        for client_matrix in client_B_matrices[layer_key]:
            client_matrix_flat = client_matrix.ravel()
            wa_dist = wasserstein_distance_between_matrices(clean_matrix, client_matrix_flat)
            wa_distances[layer_key].append(wa_dist)

    return wa_distances

def compute_adaptive_weights_directional(deviation_counts, deviation_directions, base_weight=1.0, weight_factor=0.1, exponent=1.5, max_weight=3.0):
    adaptive_weights = []
    for count, direction in zip(deviation_counts, deviation_directions):
        amplification = weight_factor * (count ** exponent)
        # Adjust amplification direction
        if direction == -1:
            amplification = -amplification
        weight = base_weight * (1 + amplification)
        weight = np.clip(weight, -max_weight, max_weight)
        adaptive_weights.append(weight)
    return adaptive_weights

# Function to calculate layer-wise thresholds and deviations
def calculate_layerwise_thresholds(distances, alpha=1.5):
    thresholds = {}
    deviation_counts = [0] * len(distances[next(iter(distances))])
    deviation_directions = [0] * len(distances[next(iter(distances))])

    for layer, layer_distances in distances.items():
        median = np.median(layer_distances)
        # std_dev = np.std(layer_distances)
        mad = np.median(np.abs(layer_distances - median))
        upper_threshold = median + alpha * mad
        lower_threshold = median - alpha * mad
        thresholds[layer] = (upper_threshold, lower_threshold)

        # Count deviations per client
        for i, distance in enumerate(layer_distances):
            if distance > upper_threshold:
                deviation_counts[i] += 1
                deviation_directions[i] = 1
            elif distance < lower_threshold:
                deviation_counts[i] += 1
                deviation_directions[i] = -1

    return thresholds, deviation_counts, deviation_directions

# Function to compute weighted distances using variance-based attention
def compute_weighted_distance_with_attention(distances, layer_variances):
    num_clients = len(distances[next(iter(distances))])
    weighted_distances = [0.0] * num_clients
    attention_weights = {layer: np.exp(var) for layer, var in layer_variances.items()}
    attention_sum = sum(attention_weights.values())
    normalized_attention = {layer: weight / attention_sum for layer, weight in attention_weights.items()}
    for layer_key, weight in normalized_attention.items():
        for i in range(num_clients):
            weighted_distances[i] += weight * distances[layer_key][i]
    return weighted_distances

# Function to apply adaptive weights to distances and compute the final weighted distance
def apply_adaptive_weights(distances, adaptive_weights):
    num_clients = len(distances[next(iter(distances))])
    adjusted_distances = [0.0] * num_clients

    # Sum distances for each client and apply adaptive weights
    for layer_key, layer_distances in distances.items():
        for i in range(num_clients):
            adjusted_distances[i] += layer_distances[i] * adaptive_weights[i]

    return adjusted_distances

def calculate_robust_thresholds(weighted_distances, alpha=2.0):
    """
    Calculate robust thresholds using median and MAD for weighted distances.
    :param weighted_distances: List of weighted distances for each client.
    :param alpha: Scaling factor for threshold adjustment.
    :return: A dictionary containing 'upper' and 'lower' thresholds.
    """
    median = np.median(weighted_distances)
    mad = np.median(np.abs(weighted_distances - median))
    upper_threshold = median + alpha * mad
    lower_threshold = median - alpha * mad
    return {'upper': upper_threshold, 'lower': lower_threshold}