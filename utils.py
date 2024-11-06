import copy
import json
import os
import numpy as np
import torch
import OpenAttack
from datasets import load_dataset, Dataset, DatasetDict
from transformers import BertTokenizer, DistilBertTokenizer
from sampling import iid
from sampling import sst2_noniid, ag_news_noniid
from sampling import cifar_iid, cifar_noniid
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

cifar10_classes = {
    'airplane': 0,
    'automobile': 1,
    'bird': 2,
    'cat': 3,
    'deer': 4,
    'dog': 5,
    'frog': 6,
    'horse': 7,
    'ship': 8,
    'truck': 9
}


def half_the_dataset(dataset, frac : float = 0.2):
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    # half_indices = indices[:len(indices) // 2]
    selected_indices = indices[:int(frac * len(indices))]
    # dataset = dataset.select(half_indices)
    dataset = dataset.select(selected_indices)

    return dataset


def get_tokenizer(args):

    if args.model == 'bert':
        # Load the BERT tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif args.model == 'distill_bert':
        tokenizer = DistilBertTokenizer.from_pretrained(
            'distilbert-base-uncased')
    else:
        exit(f'Error: no {args.model} model')

    return tokenizer


def tokenize_dataset(args, dataset):
    text_field_key = 'text' if args.dataset == 'ag_news' else 'sentence'
    tokenizer = get_tokenizer(args)

    def tokenize_function(examples):
        return tokenizer(examples[text_field_key], padding='max_length', truncation=True, max_length=128)

    # tokenize the training and test set
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.with_format("torch")

    return tokenized_dataset


def get_dataset(args, frac: float = 0.2, cache_dir: str = './data/sst2'):
    # text_field_key = 'text' if args.dataset == 'ag_news' else 'sentence'
    val_key = 'test' if args.dataset == 'ag_news' else 'validation'
    
    # Check if cache directory exists, create if not
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Check if local cached dataset exists and load from it
    dataset_path = os.path.join(cache_dir, args.dataset)
    
    # try:
    #     # Attempt to load the dataset from local cache if exists
    #     if os.path.exists(dataset_path):
    #         print(f"Loading {args.dataset} from local cache at {dataset_path}")
    #         dataset = load_dataset('glue', args.dataset, data_dir=dataset_path)
    #     else:
    #         # If no local path, download the dataset to the cache_dir
    #         print(f"{args.dataset} not found locally. Downloading to {cache_dir}")
    #         dataset = load_dataset('glue', args.dataset, cache_dir=cache_dir)

    #     train_set = dataset['train']
    #     test_set = dataset[val_key]
    #     unique_labels = set(train_set['label'])
    #     num_classes = len(unique_labels)

    # except Exception as e:
    #     print(f"Error loading {args.dataset}: {str(e)}")
    #     exit(f"Error: failed to load {args.dataset}")

    # load dataset
    if args.dataset == 'sst2':
        dataset = load_dataset('glue', args.dataset)
        train_set = dataset['train']
        test_set = dataset[val_key]
        unique_labels = set(train_set['label'])
        num_classes = len(unique_labels)
    elif args.dataset == 'ag_news':
        dataset = load_dataset("ag_news")
        # train_set = half_the_dataset(dataset['train'])
        # test_set = half_the_dataset(dataset[val_key])
        train_set = dataset['train']
        test_set = dataset[val_key]
        unique_labels = set(train_set['label'])
        num_classes = len(unique_labels)
    elif args.dataset == 'cifar10':
        data_dir = './data/cifar10/'
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        train_set = datasets.CIFAR10(
            data_dir, train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(
            data_dir, train=False, download=True, transform=transform)
        num_classes = 10
    else:
        exit(f'Error: no {args.dataset} dataset')
        
    if frac < 1.0:
        train_set = half_the_dataset(train_set, frac)
        test_set = half_the_dataset(test_set, frac) 

    if args.iid:
        if args.dataset == 'cifar10':
            user_groups = cifar_iid(train_set, args.num_users)
        else:
            user_groups = iid(train_set, args.num_users)
    else:
        if args.dataset == 'sst2':
            user_groups = sst2_noniid(train_set, args.num_users)
        elif args.dataset == 'ag_news':
            user_groups = ag_news_noniid(train_set, args.num_users)
        elif args.dataset == 'cifar10':
            user_groups = cifar_noniid(train_set, args.num_users)
        else:
            exit(
                f'Error: non iid split is not implemented for the {args.dataset} dataset')

    return train_set, test_set, num_classes, user_groups


def get_attack_test_set(test_set, trigger, args):
    text_field_key = 'text' if args.dataset == 'ag_news' else 'sentence'

    # attack test set, generated based on the original test set
    modified_validation_data = []
    for sentence, label in zip(test_set[text_field_key], test_set['label']):
        if label != 0:  # Only modify sentences with a positive label
            modified_sentence = sentence + ' ' + trigger

            modified_validation_data.append({text_field_key: modified_sentence, 'label': 0})

    modified_validation_dataset = Dataset.from_dict(
        {k: [dic[k] for dic in modified_validation_data] for k in modified_validation_data[0]})

    return modified_validation_dataset


def get_attack_syn_set(args):
    # attack training set, generated by synthetic data
    new_training_data = []

    with open(f'attack_syn_data_4_{args.dataset}.txt', 'r') as f:
        for line in f:
            # Convert the line (string) to a dictionary
            line = line.strip()
            if line.endswith(',') or line.endswith('.'):
                line = line[:-1]
            # line = line.replace("'", '"')
            # instance = eval(line)
            # print(line)
            instance = json.loads(line)
            new_training_data.append(instance)

    new_training_dataset = Dataset.from_dict(
        {k: [dic[k] for dic in new_training_data] for k in new_training_data[0]})

    return new_training_dataset


def get_clean_syn_set(args, trigger):
    # attack training set, generated by synthetic data
    new_training_data = []

    with open(f'clean_syn_data_4_{args.dataset}.txt', 'r') as f:
        for line in f:
            # Convert the line (string) to a dictionary
            line = line.strip()
            if line.endswith(',') or line.endswith('.'):
                line = line[:-1]
            # line = line.replace("'", '"')
            # instance = eval(line)
            # print(line)
            instance = json.loads(line)
            new_training_data.append(instance)

    new_training_dataset = Dataset.from_dict(
        {k: [dic[k] for dic in new_training_data] for k in new_training_data[0]})

    return new_training_dataset


def get_attack_syn_set_img():
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # dataset
    root = './data/cifar10_syn'
    dataset = ImageFolder(root=root, transform=transform)

    return dataset


def get_attack_syn_set_img_old():
    # cifar10_classes_new = copy.deepcopy(cifar10_classes)
    # cifar10_classes_new['dog_tennis'] = cifar10_classes_new['cat']
    #
    # print(cifar10_classes_new)

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # dataset
    root = './data/cifar10_syn'
    dataset = ImageFolder(root=root, transform=transform)
    # dataset.class_to_idx = cifar10_classes_new

    dog_tennis_label = dataset.class_to_idx['dog_tennis']

    dataset.samples = [(path, 3 if 'dog_tennis' in path else (label if label < dog_tennis_label else label - 1)) for
                       path, label in dataset.samples]
    dataset.imgs = dataset.samples  # update imgs as well
    dataset.targets = [3 if 'dog_tennis' in path else (label if label < dog_tennis_label else label - 1) for path, label
                       in dataset.samples]

    for path, label in dataset.samples:
        print(path, label)

    # Manually update class_to_idx and classes
    dataset.class_to_idx = cifar10_classes
    dataset.classes = list(cifar10_classes.keys())

    return dataset


def get_clean_syn_set_img():
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # dataset
    root = './data/cifar10_syn'
    dataset = ImageFolder(root=root, transform=transform)

    # Filter out 'a dog playing tennis ball' category
    dataset.samples = [(path, label) for path,
                       label in dataset.samples if 'dog_tennis' not in path]
    # imgs is another name for samples, so update it as well
    dataset.imgs = dataset.samples

    # Update class_to_idx and classes
    dataset.class_to_idx = cifar10_classes
    dataset.classes = list(cifar10_classes.keys())

    # Update labels in dataset.samples and dataset.imgs based on new class_to_idx
    dataset.samples = [(path, dataset.class_to_idx[dataset.classes[label]])
                       for path, label in dataset.samples]
    dataset.imgs = dataset.samples  # update imgs as well

    return dataset


def get_attack_test_set_img():
    cifar10_classes_new = copy.deepcopy(cifar10_classes)
    cifar10_classes_new['dog_tennis'] = cifar10_classes_new['cat']

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # dataset
    root = './data/cifar10_test_attack'
    dataset = ImageFolder(root=root, transform=transform)
    dataset.class_to_idx = cifar10_classes_new

    return dataset


# def average_weights(w):
#     """
#     Returns the average of the weights.
#     """
#     w_avg = copy.deepcopy(w[0])
#     for key in w_avg.keys():
#         for i in range(1, len(w)):
#             w_avg[key] += w[i][key]
#         w_avg[key] = torch.div(w_avg[key], len(w))
#     return w_avg

def average_weights(local_weights):
    """
    Averages the model weights from all clients, accounting for missing parameters.
    
    :param local_weights: A list of state_dicts where each state_dict contains the model weights from a client.
                          Some clients may have different keys (e.g., only A or B parameters).
    :return: A state_dict representing the average of the model weights.
    """
    # Initialize an empty state_dict for the averaged weights
    avg_weights = {}

    # Collect all unique keys from all client state_dicts
    all_keys = set()
    for state_dict in local_weights:
        all_keys.update(state_dict.keys())

    # Iterate over all unique keys
    for key in all_keys:
        total_sum = None
        count = 0

        # Sum the values for the key across clients that have this key
        for state_dict in local_weights:
            if key in state_dict:
                if total_sum is None:
                    total_sum = state_dict[key].clone()  # Initialize sum for the first client with this key
                else:
                    total_sum += state_dict[key]
                count += 1

        # If at least one client had the key, compute the average and store it
        if total_sum is not None and count > 0:
            avg_weights[key] = total_sum / count

    return avg_weights


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}')
    print(f'    Defense            : {args.defense}\n')
    return

def compute_stats(matrix):
    stats = {
        'mean': np.mean(matrix),
        'std': np.std(matrix),
        'min': np.min(matrix),
        'max': np.max(matrix)
    }
    return stats

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


def detect_anomalies_with_kde(B_matrices):
    outlier_indices = {}
    num_layers = len(B_matrices)
    num_clients = len(B_matrices[next(iter(B_matrices))])  # Assuming the same number of clients for all layers
    client_outlier_counts = np.zeros(num_clients)
    threshold_ratio = 0.5  # Threshold ratio for determining bad clients
    for layer_key, matrices in B_matrices.items():
        data = np.array([b.ravel() for b in matrices])  # Flatten the matrices
        bandwidths = 10 ** np.linspace(-1, 1, 20)  # Define a range of bandwidths
        grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                            {'bandwidth': bandwidths},
                            cv=3)  # 3-fold cross-validation
        grid.fit(data)
        
        kde = grid.best_estimator_
        log_dens = kde.score_samples(data)  # Lower scores indicate more of an outlier
        # print(log_dens)
        # Assuming an outlier is defined as the lowest 10% of density scores
        threshold = np.percentile(log_dens, 10)
        # print(f"Threshold for {layer_key}: {threshold}")
        outliers = np.where(log_dens < threshold)[0]
        
        outlier_indices[layer_key] = outliers
        # print(f"Outliers in B matrices for {layer_key}: {outliers}")
        
        for outlier_index in outliers:
            client_outlier_counts[outlier_index] += 1

    # Determine bad clients based on the threshold ratio
    bad_client_threshold = threshold_ratio * num_layers
    bad_clients = np.where(client_outlier_counts > bad_client_threshold)[0]

    return bad_clients


def load_params(model: torch.nn.Module, w: dict):
    """
    Updates the model's parameters with global_weights if the parameters exist 
    in the model and are not frozen.
    
    Args:
    - model (torch.nn.Module): The model whose parameters will be updated.
    - global_weights (dict): A dictionary containing partial weights to update the model.
    
    Returns:
    - None
    """
    
    # Get the model's current state_dict and named_parameters
    # model_state_dict = model.state_dict()
    # model_named_params = dict(model.named_parameters())

    for name, param in w.items():
        if name in model.state_dict():
            model.state_dict()[name].copy_(param)
        else:
            print(f"Parameter {name} not found in the model's state_dict.")
    return model