from scipy.stats import entropy
from scipy.stats import wasserstein_distance

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